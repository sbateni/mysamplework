import os
import json
import pickle
import argparse
from itertools import product

import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
from scipy import signal
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import acf, pacf
from termcolor import cprint

from Holdover_Utility import *

class Inference_Model():
    """
    base class for inference model
    """

    def __init__(self, **configData):
        """
        Arguments:
        configData: the config dict        """
        self.__dict__.update(configData)  # all configData become instance variables
        self.bufferLog = {}
    def sample_pre_processing(self, **modelData):
        """
        The exact same preprocessing done for training is done  on input data for sample prediction
        Arguments:
        modelData: the model dict
        """

        self.__dict__.update(modelData)  # all modelData become instance variables
        N_FEEDBACK = int(self.modelLog['nFeedbacks'])
        N_TEMP = int(self.modelLog['nTemps'])
        if N_FEEDBACK == 0 or N_TEMP == 0:
            self.rnnFeatures = 1
        else:
            self.rnnFeatures = 2
        lag = np.maximum(N_FEEDBACK, N_TEMP)

        if self.featureSelection is None:
            self.historyDepth = lag

        self.downsampleRatio = int(self.targetSamplingTime / self.currentSamplingTime)
        #------read enough samples from the end of the file for predicting one sample
        length = int((self.historyDepth + self.N_MARGIN) * self.downsampleRatio)
        readDataWhole = np.genfromtxt(self.inputFile, delimiter=',', skip_header=1)
        readDataWindow = readDataWhole[~np.isnan(readDataWhole).any(axis=1)][-length:,:]
        self.readData = readDataWindow

        # --------low-pass filtering and decimation----------------

        data1 = np.convolve(self.decimationFilterTaps, self.readData[:, 1], 'valid')[::-1][::self.downsampleRatio][::-1]
        data2 = np.convolve(self.decimationFilterTaps, self.readData[:, 2], 'valid')[::-1][::self.downsampleRatio][::-1]
        data = np.vstack((data1, data2)).T
        numData = data1.size
        self.timeStamps = self.readData[:, 0][29:][::-1][::self.downsampleRatio][::-1]
        # ------------linear de-trending----------------------------
        if self.detrending:
            linearRegPredict = self.trend.predict(self.timeStamps.reshape((numData, 1)))
            data[:, 1] = data[:, 1] - linearRegPredict[:, 0]
        self.rawData = np.copy(data)

        # -------Data preProcessing---------
        if self.preProcessing == 'filter':
            self.freqDrift = np.convolve(self.rawData[:, 1], self.firFilterTaps, mode='valid')
            self.temperature = np.convolve(self.firFilterTaps, self.rawData[:, 0], mode='valid')
        elif self.preProcessing == 'diff':
            self.bias = self.rawData[:, 1][-1]
            self.freqDrift = np.diff(self.rawData[:, 1])
            self.temperature = np.diff(self.rawData[:, 0])
        elif self.preProcessing == 'diff+filter':
            self.freqDrift = np.convolve(self.firFilterTaps, self.rawData[:, 1], mode='valid')
            self.bias = self.freqDrift[-1]
            self.freqDrift = np.diff(self.freqDrift)
            self.temperature = np.convolve(self.firFilterTaps, self.rawData[:, 0], mode='valid')
            self.temperature = np.diff(self.temperature)
        elif self.preProcessing is None:
            self.freqDrift = np.copy(self.rawData[:, 1])
            self.temperature = np.copy(self.rawData[:, 0])
        else:
            raise Exception('Error: unknown pre-processing')

        # -----normalizing data---------------

        if self.normalization:
            self.freqDrift = self.freqDrift - self.normMean
            self.freqDrift = self.freqDrift / self.normScale
            self.temperature = self.temperature - self.tempNormMean
            self.temperature = self.temperature / self.tempNormScale
        #------------------------preparing the input data for sample prediction------
        self.freqDrift = self.freqDrift[:-1]
        if os.path.exists('buffer.pickle')== False:
            self.bufferLog = {key: self.__dict__[key] for key in
                         ('freqDrift', 'temperature')}
        else:
            with open('buffer.pickle', 'rb') as pickleBuffer:
                self.bufferLog['freqDrift'] = pickle.load(pickleBuffer)
            self.freqDrift = self.bufferLog['freqDrift']

        self.inputSample = np.expand_dims(
            np.hstack((self.temperature[- N_TEMP:],
            self.freqDrift[-(self.historyDepth):])),axis=0)


    def sample_predict(self):
        """
        predicts one sample for any architecture

        """

        N_FEEDBACK = int(self.modelLog['nFeedbacks'])
        N_TEMP = int(self.modelLog['nTemps'])
        if self.architecture == 'MLP':
            self.samplePrediction, caches = mlp_forward_propagation(self.inputSample[:, self.modelLog['featureLags']].T, self.trainHistory['trainedParameters'],
                                                                 self.activation)
        if self.architecture == 'RNN':
            self.samplePrediction, _, _ = rnn_forward_propagation(
                np.reshape(self.inputSample[:, self.modelLog['featureLags']], (1, np.maximum(N_FEEDBACK, N_TEMP), self.rnnFeatures),
                           order='F'), self.trainHistory['trainedParameters'])
        elif self.architecture == 'LSTM':
            self.samplePrediction, _, _ = lstm_forward_propagation(
                np.reshape(self.inputSample[:, self.modelLog['featureLags']], (1, np.maximum(N_FEEDBACK, N_TEMP), self.rnnFeatures),
                           order='F'), self.trainHistory['trainedParameters'])
        self.bufferLog['freqDrift'] = np.roll(self.bufferLog['freqDrift'], -1)
        self.bufferLog['freqDrift'][-1] = self.samplePrediction[0]
        print('sample infered for trained model...', )
        with open('buffer.pickle', 'wb') as bufferFile:
            pickle.dump(self.bufferLog['freqDrift'], bufferFile)

    def sample_log(self):
        """
        process back the sample prediction and log it to output file
        """

        # ----de-normalizing prediction---------
        if self.normalization:
            self.samplePrediction = self.samplePrediction * self.normScale + self.normMean
        if 'diff' in self.preProcessing:
            if os.path.exists(self.outputFile):
                with open(self.outputFile, 'r') as outputFile:
                    self.bias = np.atleast_1d(np.loadtxt(outputFile))[-1]
            self.samplePrediction = self.samplePrediction + self.bias
        if self.detrending:
            sampleLinearReg = self.trend.predict((self.timeStamps[-1] + self.targetSamplingTime).reshape(1,-1))
            self.samplePrediction = self.samplePrediction + sampleLinearReg


        with open(self.outputFile, 'a') as outputFile:
            np.savetxt(outputFile, self.samplePrediction)

class Holdover_Model():
    """
    base class for holdover model
    """

    def __init__(self, *args, **configData):
        """
        Arguments:
        inputFile: The input data file containing frequency drifts and (probably) corresponding temperature inputs
        args: possible future use
        kwargs: the config dict
        """

        self.__dict__.update(configData) #all configData become instance variables
        self.validLog = {}
        self.modelLog = {}
        self.trainHistory = {}
        self.prediction = {}
        self.BayesianHyperparameters = []
        self.validLoss = []
        self.errors = []

    # ---------------parameter setting----------------------
    def model_tuning(self, tuningAlg='grid'):
        """
        main function for (grid and bayesian) tuning on top of train_model() and cross_validate_model() functions
        trains model for each set of hyperparameters and captures the  results
        Arguments:
        tuningAlg: grid or bayesian
        """
        # ------------------tuning and validation------------------------------------------
        hyperParameterDict = {}
        self.tuningAlg = tuningAlg
        if tuningAlg == 'grid':
            hyperParameterSets = product(*list(self.gridHyperparameters.values()))
        elif tuningAlg == 'Bayesian':
            hyperParameterSets = [list(self.BayesianNextHyperparameters)]
        for hyperParameterSet in hyperParameterSets:
            for counter, key in enumerate(self.gridHyperparameters.keys()):
                hyperParameterDict[key] = hyperParameterSet[counter]
                if key == 'nInputs':
                    hyperParameterDict['nTemps'] = hyperParameterSet[counter] * np.sign(self.nTemps)
                    hyperParameterDict['nFeedbacks'] = hyperParameterSet[counter] * np.sign(self.nFeedbacks)
            self.__dict__.update(hyperParameterDict)
            self.data_windowing()  # prepare windowed data for training

            if self.initMethod == 'pretrain' and tuningAlg == 'grid':  #---read pretrained parameters for initialization
                with open('preTrain/preTrainedModel.pickle', 'rb') as pickleLog:
                    self.initParameters = pickle.load(pickleLog)[str(hyperParameterSet)]['trainedParameters']

            print('training for hyperparameters:' + str(hyperParameterSet) + '...')
            if self.validationMode == 'CV':
                trainHistory, validationError, validPrediction, validOutput = self.cross_validate_model()
            elif self.validationMode == 'HO':
                trainHistory = self.train_model(self.laggedTrainData, self.laggedValidData)

            if self.architecture == 'RNN':
                trainPrediction, _, _ = rnn_forward_propagation(np.reshape(self.laggedTrainData[0], (
                self.laggedTrainData[0].shape[0], np.maximum(self.nTemps, self.nFeedbacks), self.rnnFeatures),
                                                                           order='F'),
                                                                trainHistory['trainedParameters'])
            elif self.architecture == 'LSTM':
                trainPrediction, _, _ = lstm_forward_propagation(np.reshape(self.laggedTrainData[0], (
                self.laggedTrainData[0].shape[0], np.maximum(self.nTemps, self.nFeedbacks), self.rnnFeatures),
                                                                            order='F'),
                                                                 trainHistory['trainedParameters'])
            elif self.architecture == 'MLP':
                trainPrediction, _ = mlp_forward_propagation(self.laggedTrainData[0].T,
                                                             trainHistory['trainedParameters'], self.activation)
            else:
                raise Exception('Error: unknown rnn architecture')
            predictions = [trainPrediction]
            actualOutputs = [self.trainData[1]]
            trainingError = metrics.mean_absolute_error(actualOutputs[0], predictions[0])
            errors = [trainingError]
            phaseErrors = []
            # ----------test and validation predictions and errors--------
            for i, data in enumerate([self.validData, self.testData]):
                if (self.validationMode == 'CV') and (i == 0):
                    predictions.append(validPrediction)
                    actualOutputs.append(validOutput)
                    errors.append(validationError)
                    continue
                dataPrediction, dataEstimation = self.predict(data[0], trainHistory['trainedParameters'])
                dataOutput = data[1].T

                # ----de-normalizing predictions---------
                if self.normalization:
                    dataPrediction = dataPrediction * self.normScale + self.normMean
                    dataEstimation = dataEstimation * self.normScale + self.normMean
                    dataOutput = dataOutput * self.normScale + self.normMean
                if 'diff' in self.preProcessing:
                    dataPrediction = np.cumsum(dataPrediction, axis=1)
                    dataOutput = np.cumsum(dataOutput, axis=1)

                mae = (metrics.mean_absolute_error(dataOutput, dataPrediction))
                predictions.append(dataPrediction.T)
                errors.append(mae)
                actualOutputs.append(dataOutput.T)

            phaseDiff, phaseDiffUncomp = self.phase_diff(actualOutputs[2], predictions[2])
            phaseError = abs(np.amax(phaseDiff) - np.amin(phaseDiff))
            phaseErrorUncomp = abs(np.amax(phaseDiffUncomp) - np.amin(phaseDiffUncomp))
            phaseErrors.append(phaseError)
            numEpoch = len(trainHistory['loss'])
            validationError = errors[1]
            # ------------logging results in a dict------------------------------
            self.validLog[str(hyperParameterSet)] = {
                                                     'training error(mae)': errors[0],
                                                     'validation error(mae)': errors[1], 'test error(mae)': errors[2],
                                                     'numElapsedEpochs': numEpoch, 'PktoPk phase error(ns)': phaseError,
                                                     'Uncomp phase error(ns)': phaseErrorUncomp, 'tuning': tuningAlg}
            self.modelLog[str(hyperParameterSet)] = {'nFeedbacks': self.nFeedbacks, 'nTemps': self.nTemps, 'featureLags':
            self.featureLags, 'historyDepth': self.historyDepth}
            self.trainHistory[str(hyperParameterSet)] = trainHistory
            self.prediction[str(hyperParameterSet)] = predictions
            self.errors.append(errors)
            # --------------------updating validation error history or bayesian tuning-----
            if (np.isnan(validationError) == False and np.isfinite(validationError) == True):
                self.BayesianHyperparameters.append(hyperParameterSet)
                np.random.seed(1)
                self.validLoss.append(np.minimum(validationError + np.random.randn() * 0.01, self.MIN_VALIDATION_ERROR))

            # --------------plotting graph results to a file---------------
            print('train error =', errors[0])
            print('valid error=', errors[1])
            print('test error=', errors[2])
            plt.figure()
            plt.suptitle(str(hyperParameterSet))
            plt.subplot(221)
            plt.xlabel('time(s)')
            plt.ylabel('frequency offset(ppb)')
            plt.plot(self.trainData[2], actualOutputs[0], 'b', label='actual_train')
            plt.plot(self.trainData[2], predictions[0], 'r', label='prediction_train')
            plt.grid(True)
            plt.legend()

            plt.subplot(222)
            plt.xlabel('time(s)')
            plt.ylabel('frequency offset(ppb)')
            plt.plot(np.arange(len(predictions[1])) * self.targetSamplingTime, actualOutputs[1], 'b',
                     label='actual_valid')
            plt.plot(np.arange(len(predictions[1])) * self.targetSamplingTime, predictions[1], 'r',
                     label='prediction_valid')
            plt.grid(True)
            plt.legend()

            plt.subplot(223)
            plt.xlabel('time(s)')
            plt.ylabel('frequency offset(ppb)')
            plt.plot(self.testData[2], predictions[2], 'r', label='prediction_test')
            plt.plot(self.testData[2], actualOutputs[2], 'b', label='actual_test')
            plt.grid(True)
            plt.legend()

            time_test = self.testData[2]
            plt.subplot(224)
            plt.plot(time_test[1:], phaseDiff, 'c', label='test data')
            plt.xlabel('time(s)')
            plt.ylabel('phase_error(ns)')
            plt.grid(True)
            plt.legend()
            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                                wspace=0.35)
            plt.savefig('figs/' +
                        str(hyperParameterSet) + '.png')
            plt.close('all')

    def model_log(self):

        """
        writes the logged data to json and pickle files and prints the final results
        """
        # --------------------validation accuracy analysis-----------------------

        errorArray = np.array(self.errors, dtype=np.float_)
        validationAccuracyIndex, validationMAE = self.valid_eval(errorArray[:, 1], errorArray[:, 2])
        print('-----------------validation accuracy--------------------------')
        print('validation accuracy index = ', validationAccuracyIndex)
        print('validation MAE = ', validationMAE)

        #        -------------------printing the results--------------------------------
        for key, value in sorted(self.validLog.items(), key=lambda item: item[1]['validation error(mae)']):
            self.selectedModel = key
            break
        print('-----------------selected model--------------------------')
        print('selected model : ' + str(self.architecture), self.selectedModel)
        print('selected model train error(mae) =', self.validLog[self.selectedModel]['training error(mae)'])
        print('selected model ' + self.validationMode + ' valid error(mae) =',
              self.validLog[self.selectedModel]['validation error(mae)'])
        print('selected model test error(mae) =', self.validLog[self.selectedModel]['test error(mae)'])
        print('selected model PktoPk phase error(ns) =', self.validLog[self.selectedModel]['PktoPk phase error(ns)'])
        # --------------------------logging to file--------------------------------------------------------------

        self.validLog['selectedModel'] = {'model': key, 'results': self.validLog[key]}
        self.validLog['VAI'] = validationAccuracyIndex
        with open('logs/log.json', 'a') as jsonLog:
            json.dump(self.validLog, jsonLog, indent=4)
        jsonLog.close()
        if self.preTrain:
            with open('preTrainedModel.pickle', 'wb') as pickleLog:
                pickle.dump(self.trainHistory, pickleLog)
        else:
            self.trainHistory = self.trainHistory[self.selectedModel]
            self.modelLog = self.modelLog[self.selectedModel]
            modelLog = {key: self.__dict__[key] for key in
                        ('modelLog', 'trainHistory', 'normScale', 'normMean','tempNormMean', 'tempNormScale',
                         'trend', 'firFilterTaps', 'decimationFilterTaps')}
            with open('model.pickle', 'wb') as pickleLog:
                pickle.dump(modelLog, pickleLog)
            #-------writing prediction to output file------
            self.prediction = self.prediction[self.selectedModel]
            np.savetxt(self.outputFile, self.prediction[2])


    def train_model(self, trainData, validData):
        """
        trains the holdover model

        Arguments:
        trainData -- tuple of (inputTrainData, outputTrainData, t_train) for training set
        validData -- tuple of (inputValidData, outputValidData, t_train) used for validation during training
        Returns:
        trainHistory -- python dictionary containing trained model parameters and training history(loss,..)
        """
        trainHistory = {}
        """
        inputTrainData -- input data, of shape (nInputs, number of examples)
        outputTrainData  -- True output data, of shape (1, number of examples)
        """

        inputTrainData = trainData[0]
        outputTrainData = trainData[1]
        inputValidData = validData[0]
        outputValidData = validData[1]

        costs = []
        parametersHistory = []
        optimizerCounter = 0  # initializing the counter required for adam update
        learningRateCounter = 0  # initializing the counter required for lerning rate decay
        # randomSeed = 0
        numTrainData = inputTrainData.shape[0]  # number of training data examples
        numValidData = inputValidData.shape[0]

        if self.architecture == 'RNN' or self.architecture == 'LSTM':
            inputTrainData = inputTrainData.reshape(numTrainData, np.maximum(self.nTemps, self.nFeedbacks),
                                                    self.rnnFeatures, order='F')
            inputValidData = inputValidData.reshape(numValidData, np.maximum(self.nTemps, self.nFeedbacks),
                                                    self.rnnFeatures, order='F')
        earlyStopCounter = 0  # initializing early stopping counter
        validLoss = []
        if self.architecture == 'MLP':
            layersDims = [inputTrainData.shape[1], int(self.nNodes), 1]
        else:
            layersDims = [inputTrainData.shape[2], int(self.nNodes), 1]

        randomSeed = self.randomSeed
        # read the pretrained params or randomly init them---
        if self.initMethod == 'pretrain' and self.tuningAlg == 'grid':
            trainableParameters = self.initParameters
        else:
            trainableParameters = initialize_parameters(layersDims, self.kernelInit, self.recurrentInit, self.architecture,
                                                    self.randomSeed)

        vAdam, sAdam = initialize_optimizer(trainableParameters, self.optimizer)
        # Optimization loop
        for i in range(self.numEpochs):
            gradCheckEpoch = True
            # Define the random minibatches. We increment the randomSeed to reshuffle differently the dataset after each epoch
            minibatches = random_mini_batches(inputTrainData, outputTrainData, self.batchSize, randomSeed)
            randomSeed = randomSeed + 1
            costTotal = 0

            if self.decay != 0.0:
                self.learningRate = self.learningRate / (1. + self.decay * float(learningRateCounter))
            learningRateCounter = learningRateCounter + 1

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                if self.architecture == 'RNN':
                    outputPred, recurrentOutput, caches = rnn_forward_propagation(minibatch_X, trainableParameters)
                elif self.architecture == 'LSTM':
                    outputPred, recurrentOutput, caches = lstm_forward_propagation(minibatch_X, trainableParameters)
                elif self.architecture == 'MLP':
                    outputPred, caches = mlp_forward_propagation(minibatch_X.T, trainableParameters, self.activation,
                                                                 self.dropProb, randomSeed)
                else:
                    raise Exception('Error: unknown rnn architecture')
                # Compute cost and add to the cost total
                costTotal += compute_cost(outputPred, minibatch_Y, self.costFunction)

                # Backward propagation
                if self.architecture == 'RNN':
                    grads = rnn_backward_propagation(minibatch_X, minibatch_Y, caches, trainableParameters,
                                                     self.costFunction)
                elif self.architecture == 'LSTM':
                    grads = lstm_backward_propagation(minibatch_X, minibatch_Y, caches, trainableParameters,
                                                      self.costFunction)
                elif self.architecture == 'MLP':
                    grads = mlp_backward_propagation(minibatch_X.T, minibatch_Y.T, caches, self.activation,
                                                     self.dropProb, self.costFunction)
                else:
                    raise Exception('Error: unknown rnn architecture')
                # -----------------gradient checking---------------------------------
                if self.gradChecking:
                    if (gradient_check(trainableParameters, grads, minibatch_X, minibatch_Y, self.architecture, self.activation)) == False:
                        gradCheckEpoch = False

                optimizerCounter = optimizerCounter + 1  # adam counter
                trainableParameters, vAdam, sAdam = update_parameters(self.optimizer, trainableParameters, grads, vAdam,
                                                                      sAdam,
                                                                      optimizerCounter, self.learningRate)
            costAvg = costTotal / numTrainData
            # ---------validation loss and early stopping based on validation loss
            if self.architecture == 'RNN':
                validPredonEpoch, _, _ = rnn_forward_propagation(inputValidData, trainableParameters)
            elif self.architecture == 'LSTM':
                validPredonEpoch, _, _ = lstm_forward_propagation(inputValidData, trainableParameters)
            elif self.architecture == 'MLP':
                validPredonEpoch, _ = mlp_forward_propagation(inputValidData.T, trainableParameters, self.activation)
            else:
                raise Exception('Error: unknown rnn architecture')
            validLoss.append(compute_cost(validPredonEpoch, outputValidData, self.costFunction) / numValidData)
            if validLoss[-1] >= validLoss[-earlyStopCounter - 1]:
                earlyStopCounter = earlyStopCounter + 1
            else:
                earlyStopCounter = 0
            print("Loss after epoch %i: %f" % (i + 1, costAvg), end="")
            print(".....valid_loss : %f" % (validLoss[-1]))
            if self.gradChecking:
                if gradCheckEpoch:
                    cprint("Grad Checking OK...", 'green')
                else:
                    cprint("Grad Checking failed ", 'red')
            costs.append(costAvg)
            parametersHistory.append(copy.deepcopy(trainableParameters))
            if costAvg < self.trainingStopLoss:
                print('Reached less than %f mse so cancelling training on epoch %i' % (self.trainingStopLoss, i))
                break
            if earlyStopCounter == self.earlyStopPatience:
                break
        if self.restoreBestWeights:
            trainedParameters = parametersHistory[-self.earlyStopPatience]
        else:
            trainedParameters = parametersHistory[-1]
        trainHistory['loss'] = costs
        trainHistory['validLoss'] = validLoss
        trainHistory['trainedParameters'] = trainedParameters

        return trainHistory

    def cross_validate_model(self):
        """
         Implements N-fold cross-validation
         Returns:
         trainHistory -- python dictionary containing trained model parameters and training history(loss,..)
         crossValidError -- cross validation error
         validPrediction -- cross validation predictions concatenated together
         validOutput -- actual output for validation data
         """

        nTrain = self.trainData[0].shape[0]
        lenCrossValid = int(nTrain / self.NUM_FOLD)
        trainIndices = np.arange(nTrain)
        folds = [np.arange(i * lenCrossValid, (i + 1) * lenCrossValid) for i in range(int(nTrain / lenCrossValid))]
        validPrediction = np.empty((lenCrossValid * self.NUM_FOLD, 1))
        validOutput = np.empty((lenCrossValid * self.NUM_FOLD, 1))
        crossValidErrors = []
        biasCrossValid = 0
        if self.crossValidMode == 'inc':
            trainHistory = self.train_model(self.laggedTrainData, self.laggedValidData)
        for fold in folds:
            inputTrainData = self.trainData[0][np.setdiff1d(trainIndices, fold), ...]
            outputTrainData = self.trainData[1][np.setdiff1d(trainIndices, fold), :]
            trainDataCrossValid = (inputTrainData, outputTrainData)
            inputValidData = self.trainData[0][fold, ...]
            outputValidData = self.trainData[1][fold, :]
            validDataCrossValid = (inputValidData, outputValidData)
            if self.crossValidMode == 'exc':
                trainHistory = self.train_model(trainDataCrossValid, validDataCrossValid)
            foldError, foldPrediction, foldOutput = self.error_calc(validDataCrossValid,
                                                                    trainHistory['trainedParameters'])
            validPrediction[fold] = foldPrediction.T + biasCrossValid
            validOutput[fold] = foldOutput.T + biasCrossValid
            if 'diff' in self.preProcessing:
                biasCrossValid = validOutput[fold[-1]]
            crossValidErrors.append(foldError)
        crossValidError = np.mean(np.array(crossValidErrors))
        if self.crossValidMode == 'exc':
            trainHistory = self.train_model(self.trainData, self.validData)

        return trainHistory, crossValidError, validPrediction, validOutput

    def data_preprocessing(self):
        """
        Reads raw data from file, do preProcessing and prepares training, validation and test sets
        """

        self.currentHourSamples = int(3600 / self.currentSamplingTime)  # one hour in samples
        self.currentTestSamples = self.testLength * self.currentHourSamples
        self.downsampleRatio = int(self.targetSamplingTime / self.currentSamplingTime)
        self.targetHourSamples = int(3600 / self.targetSamplingTime)
        self.targetTestSamples = self.testLength * self.targetHourSamples
        start = int(self.captureStart * self.currentHourSamples)
        length = int(self.captureLength * self.currentHourSamples)
        readDataWhole = np.genfromtxt(self.inputFile, delimiter=',', skip_header=1)
        readDataWindow = readDataWhole[~np.isnan(readDataWhole).any(axis=1)][start:start + length, :]
        self.readData = readDataWindow

        # --------low-pass filtering and decimation----------------
        self.decimationFilterTaps = signal.firwin(30, 1. / self.downsampleRatio, window='hamming')

        data1 = np.convolve(self.decimationFilterTaps, self.readData[:, 1], 'valid')[::-1][::self.downsampleRatio][::-1]
        data2 = np.convolve(self.decimationFilterTaps, self.readData[:, 2], 'valid')[::-1][::self.downsampleRatio][::-1]
        data = np.vstack((data1, data2)).T
        numData = data1.size
        self.timeStamps = self.readData[:, 0][29:][::-1][::self.downsampleRatio][::-1]
        # ------------linear de-trending----------------------------

        reg = LinearRegression().fit(self.timeStamps[:-self.targetTestSamples].reshape((numData - self.targetTestSamples, 1)),
            data[:-self.targetTestSamples, 1:])
        self.trendCoef = reg.coef_
        self.trendIntercept = reg.intercept_
        self.trend = reg
        if self.detrending:
            linearRegPredict = reg.predict(self.timeStamps.reshape((numData, 1)))
            data[:, 1] = data[:, 1] - linearRegPredict[:, 0]

        self.rawData = np.copy(data)

        # -------Data preProcessing---------
        self.firFilterTaps = signal.firwin(self.firFilterOrder, self.firFilterCutoff,
                                           window=('kaiser',
                                                   1))  # fir filter with cutoff = 0.1 for potential de-noising of input data
        if self.preProcessing == 'filter':
            self.freqDrift = np.convolve(self.rawData[:, 1], self.firFilterTaps, mode='valid')
            self.temperature = np.convolve(self.firFilterTaps, self.rawData[:, 0], mode='valid')
        elif self.preProcessing == 'diff':
            self.freqDrift = np.diff(self.rawData[:, 1])
            self.temperature = np.diff(self.rawData[:, 0])
        elif self.preProcessing == 'diff+filter':
            self.freqDrift = np.convolve(self.firFilterTaps, self.rawData[:, 1], mode='valid')
            self.freqDrift = np.diff(self.freqDrift)
            self.temperature = np.convolve(self.firFilterTaps, self.rawData[:, 0], mode='valid')
            self.temperature = np.diff(self.temperature)
        elif self.preProcessing is None:
            self.freqDrift = np.copy(self.rawData[:, 1])
            self.temperature = np.copy(self.rawData[:, 0])
        else:
            raise Exception('Error: unknown pre-processing')
        # ------------------------temp---------------------------
        plt.figure()
        plt.plot(np.arange(self.freqDrift.size) * self.targetSamplingTime, self.freqDrift, label='raw freq drift')
        plt.xlabel('time(s)')
        plt.legend()
        plt.grid(True)
        # plt.show()
        # -----normalizing data---------------
        self.normMean, self.normScale = [0, 1]
        if self.normalization:
            self.normMean = np.mean(self.freqDrift[:-self.targetTestSamples])
            self.freqDrift = self.freqDrift - self.normMean
            self.normScale = np.std(self.freqDrift[:-self.targetTestSamples])
            self.freqDrift = self.freqDrift / self.normScale
            self.tempNormMean = np.mean(self.freqDrift[:-self.targetTestSamples])
            self.temperature = self.temperature - self.tempNormMean
            self.tempNormScale = np.std(self.temperature[:-self.targetTestSamples])
            self.temperature = self.temperature / self.tempNormScale



        # -----aligning input and output ---------------
        delay = 0
        if self.alignment == True:
            if 'diff' in self.preProcessing:
                crossCorr = np.correlate(self.temperature, self.freqDrift, 'full')
                delay = (len(self.temperature) - 1) - np.argmax(abs(crossCorr))
            else:
                crossCorr = np.correlate(np.diff(self.temperature), np.diff(self.freqDrift), 'full')
                delay = (len(self.temperature) - 2) - np.argmax(abs(crossCorr))
            self.temperature = np.roll(self.temperature, delay, axis=0)  # aligning self.temperature and frequency data
            if delay < 0:
                self.freqDrift = np.delete(self.freqDrift, np.arange(len(self.temperature))[delay:])
                self.temperature = np.delete(self.temperature, np.arange(len(self.temperature))[delay:])
            if delay > 0:
                self.freqDrift = np.delete(self.freqDrift, np.arange(len(self.temperature))[:delay])
                self.temperature = np.delete(self.temperature, np.arange(len(self.temperature))[:delay])

        self.groupDelay = int(self.firFilterOrder / 2) - delay

    def data_windowing(self):
        """
        performs feature selection and windowing on preprocessed data according to number of temperature and feedback inputs
        """

        N_FEEDBACK = int(self.nFeedbacks)
        N_TEMP = int(self.nTemps)
        if N_FEEDBACK == 0 or N_TEMP == 0:
            self.rnnFeatures = 1
        else:
            self.rnnFeatures = 2
        lag = np.maximum(N_FEEDBACK, N_TEMP)

        if self.featureSelection is None:
            self.historyDepth = lag

        N_DATA = self.freqDrift.shape[0]
        N_TOTAL = N_DATA - self.historyDepth
        N_VALID = self.validLength * self.targetHourSamples
        N_TEST = self.targetTestSamples
        N_TRAIN = N_TOTAL - (N_TEST + N_VALID) - 2 * self.N_MARGIN
        timeProcess = np.arange(self.freqDrift.size) * self.targetSamplingTime
        timeProcess = np.expand_dims(timeProcess, axis=1)

        # ---------------Feature Selection-----------------------------
        partialAutocorr, partialConfint = pacf(self.freqDrift, nlags=self.historyDepth,
                                               alpha=0.05)  # partial autocorrelation function
        autocorr, confint = acf(self.freqDrift, nlags=self.historyDepth, alpha=0.05)  # autocorrelation function
        # only up to nSignificant features (above confidence levels) are selected. The rest up to Ninputs are selected as before
        if N_FEEDBACK != 0:
            if self.featureSelection == 'PAC':
                nSignificant = len(partialConfint[(partialConfint > 0).all(axis=1)]) - 1
                featureLags = (self.historyDepth - 1) - (
                np.argsort(partialAutocorr[1:])[-np.minimum(N_FEEDBACK, nSignificant):])
            elif self.featureSelection == 'PAC_ABS':
                nSignificant = len(
                    partialConfint[(partialConfint > 0).all(axis=1) | (partialConfint < 0).all(axis=1)]) - 1
                featureLags = (self.historyDepth - 1) - (
                np.argsort(abs(partialAutocorr))[-np.minimum(N_FEEDBACK, nSignificant):])
            elif self.featureSelection == 'AC':
                nSignificant = len(confint[(confint > 0).all(axis=1)]) - 1
                featureLags = (self.historyDepth - 1) - (
                np.argsort(abs(autocorr))[-np.minimum(N_FEEDBACK, nSignificant):])
            else:
                nSignificant = N_FEEDBACK
                featureLags = np.arange(self.historyDepth)[-N_FEEDBACK:]
            if N_FEEDBACK > nSignificant:
                remainLags = np.setdiff1d(np.arange(self.historyDepth), featureLags)[-(N_FEEDBACK - nSignificant):]
                featureLags = np.hstack((remainLags, featureLags))
        else:
            featureLags = np.array([], dtype=np.int64)

        self.featureLags = np.hstack((np.arange(N_TEMP, dtype=np.int64), N_TEMP + np.sort(featureLags)))
        # -------------------windowing training, validation and test data-------------------------

        windowedDataInput = np.expand_dims(
            np.hstack((self.temperature[self.historyDepth - N_TEMP + 1:self.historyDepth + 1],
                       self.freqDrift[:self.historyDepth + 1])),
            axis=0)

        for i in range(1, N_TOTAL):
            windowedData = np.expand_dims(
                np.hstack((self.temperature[self.historyDepth + i - N_TEMP + 1:self.historyDepth + i + 1],
                           self.freqDrift[i:i + self.historyDepth + 1])), axis=0)
            windowedDataInput = np.vstack((windowedDataInput, windowedData))

        inputTrainData, outputTrainData = windowedDataInput[:N_TRAIN, :-1], np.expand_dims(
            windowedDataInput[:N_TRAIN, -1], axis=1)
        inputValidData, outputValidData = windowedDataInput[
                                          -(N_TEST + N_VALID + self.N_MARGIN): -(
                                                  N_TEST + self.N_MARGIN),
                                          :-1], np.expand_dims(
            windowedDataInput[-(N_TEST + N_VALID + self.N_MARGIN): -(N_TEST + self.N_MARGIN), -1],
            axis=1)
        inputTestData, outputTestData = windowedDataInput[-N_TEST:, :-1], np.expand_dims(
            windowedDataInput[- N_TEST:, -1], axis=1)

        trainTime = timeProcess[: N_TRAIN] - timeProcess[0]
        validTime = timeProcess[-(N_TEST + N_VALID + self.N_MARGIN): -(N_TEST + self.N_MARGIN)] - timeProcess[
            -(N_TEST + N_VALID + self.N_MARGIN)]
        testTime = timeProcess[- N_TEST:] - timeProcess[- N_TEST]

        validSeparation = int(inputValidData.shape[0] / 2)
        self.trainData = (inputTrainData, outputTrainData, trainTime)
        self.validData = (inputValidData, outputValidData, validTime)
        self.testData = (inputTestData, outputTestData, testTime)

        self.laggedTrainData = (inputTrainData[:, self.featureLags], outputTrainData, trainTime)
        self.laggedValidData = (inputValidData[:, self.featureLags], outputValidData, validTime)

    def predict(self, inputData, trainedParameters):
        """
        Implements the open loop and closed loop prediction for holdover model

        Arguments:
        InputData -- input dataset to the model
        trainedParameters -- trained model object

        Returns:
        closedLoopPrediction --closed loop (autoregressive) prediction for the given dataset InputData(only first sample is used)
        openLoopPrediction -- open loop prediction for the given dataset InputData
        """
        nInputs = inputData.shape[1]
        numData = inputData.shape[0]
        N_FEEDBACK = int(self.nFeedbacks)
        N_TEMP = int(self.nTemps)
        if self.architecture == 'MLP':
            openLoopPrediction, caches = mlp_forward_propagation(inputData[:, self.featureLags].T, trainedParameters,
                                                                 self.activation)
        if self.architecture == 'RNN':
            openLoopPrediction, _, _ = rnn_forward_propagation(
                np.reshape(inputData[:, self.featureLags], (numData, np.maximum(N_FEEDBACK, N_TEMP), self.rnnFeatures),
                           order='F'), trainedParameters)
        elif self.architecture == 'LSTM':
            openLoopPrediction, _, _ = lstm_forward_propagation(
                np.reshape(inputData[:, self.featureLags], (numData, np.maximum(N_FEEDBACK, N_TEMP), self.rnnFeatures),
                           order='F'), trainedParameters)
        if nInputs == N_TEMP:
            return openLoopPrediction.T, openLoopPrediction.T
        closedLoopInput = np.copy(inputData[0:1, :])
        tempInput = inputData[:, :N_TEMP]
        closedLoopPrediction = np.zeros((1, numData))
        closedLoopFeedbackInput = closedLoopInput[:, N_TEMP:]
        for r in range(numData):
            closedLoopInput[0, :N_TEMP] = tempInput[r, :]

            if self.architecture == 'MLP':
                closedLoopPrediction[0, r], _ = mlp_forward_propagation(closedLoopInput[:, self.featureLags].T,
                                                                        trainedParameters, self.activation)
            if self.architecture == 'RNN':
                closedLoopPrediction[0, r], _, _ = rnn_forward_propagation(
                    np.reshape(closedLoopInput[:, self.featureLags],
                               (1, np.maximum(N_FEEDBACK, N_TEMP), self.rnnFeatures), order='F'), trainedParameters)
            elif self.architecture == 'LSTM':
                closedLoopPrediction[0, r], _, _ = lstm_forward_propagation(
                    np.reshape(closedLoopInput[:, self.featureLags],
                               (1, np.maximum(N_FEEDBACK, N_TEMP), self.rnnFeatures), order='F'), trainedParameters)

            closedLoopFeedbackInput = np.roll(closedLoopFeedbackInput, -1, axis=1)
            closedLoopFeedbackInput[0, -1] = closedLoopPrediction[0, r]
            closedLoopInput[:, N_TEMP:] = closedLoopFeedbackInput

        return closedLoopPrediction, openLoopPrediction.T

    def phase_diff(self, output, prediction):
        """
        Implements phase error calculation between prediction and output actual value
        Arguments:
        output -- actual output
        prediction -- predicted output
        time -- time stamps of the output
        Returns:
        phaseDiff: phase difference vs time
        """

        freqDiff = np.squeeze(output - prediction)
        time = np.arange(freqDiff.size) * self.targetSamplingTime
        freqDiffUncomp = np.squeeze(output - output[0])
        phaseDiff, phaseDiffUncomp = [], []
        for n in range(1, freqDiff.size):
            phaseDiff.append(np.trapz(freqDiff[:n + 1], time[:n + 1], axis=0))
            phaseDiffUncomp.append(np.trapz(freqDiffUncomp[:n + 1], time[:n + 1], axis=0))

        return phaseDiff, phaseDiffUncomp

    def valid_eval(self, validationErrors, predictionErrors):
        """
        Returns validation accuracy index and validation-test errors average MAE
        Arguments:
        validationErrors: validation errors for all models
        predictionErrors: test errors for all models
        Returns:
        accuracyIndex: validation accuracy index
        validationMAE: weighted average of MAE between test and validation errors
        """
        num = len(predictionErrors)
        beta = 10 / num
        maxIndex = np.sum(abs(np.arange(num) - np.flip(np.arange(num))) * np.exp(-np.arange(num) * beta))
        predictionIndex = np.argsort(predictionErrors)
        validationIndex = np.argsort(validationErrors)
        np.savetxt('index', np.vstack((predictionIndex, validationIndex)), fmt='%d')
        predValidCrossIndex = np.array([np.where(predictionIndex == i) for i in validationIndex]).reshape((num,))
        accuracyIndex = 1 - np.sum(
            abs(np.arange(num) - predValidCrossIndex) * np.exp(-np.arange(num) * beta)) / maxIndex
        validationMAE = sum(abs(predictionErrors[validationIndex] - validationErrors[validationIndex]) * np.exp(
            -np.arange(num) * beta)) / sum(np.exp(-np.arange(num) * beta))

        return accuracyIndex, validationMAE

    def error_calc(self, data, trainedParameters, predMode='prediction'):
        """
         Calculates rmse error between actual outputs and model predictions
         Arguments:
         data: tuple of (inputData, outputData, dataTime)
         trainableParameters: trained model trainableParameters
         predMode: 'prediction' for closed-loop prediction and 'estimation' for open-loop prediction
         Returns:
         error: rmse error between prediction and output
         prediction: model prediction for inputData
         dataOutput: actual output (processed back in case of diff or normalization )
        """

        dataPrediction, dataEstimation = self.predict(data[0], trainedParameters)
        dataOutput = data[1].T
        if self.normalization:
            dataPrediction = dataPrediction * self.normScale + self.normMean
            dataEstimation = dataEstimation * self.normScale + self.normMean
            dataOutput = dataOutput * self.normScale + self.normMean
        if 'diff' in self.preProcessing:
            dataPrediction = np.cumsum(dataPrediction, axis=1)
            dataEstimation = np.cumsum(dataEstimation, axis=1)
            dataOutput = np.cumsum(dataOutput, axis=1)

        maePrediction = (metrics.mean_absolute_error(dataOutput, dataPrediction))
        maeEstimation = (metrics.mean_absolute_error(dataOutput, dataEstimation))

        if predMode == 'prediction':
            error = maePrediction
            prediction = dataPrediction
        elif predMode == 'estimation':
            error = maeEstimation
            prediction = dataEstimation

        return error, prediction, dataOutput

    def bayesian_optimisation(self):
        """ bayesian_optimisation
        Uses Gaussian Processes to optimise the loss function `sample_loss`.
        Arguments:
        ----------
            self.nIterBayesian: integer.
                Number of iterations to run the search algorithm.
            sample_loss: function.
                Function to be optimised.
            paramPDFs: dictionary
                Distribution, Lower and upper bounds on the parameters of the function `sample_loss`.
            x0: array-like, shape = [n_pre_samples, n_params].
                Array of initial points to sample the loss function for. If None, randomly
                samples from the loss function.
            n_pre_samples: integer.
                If x0 is None, samples `n_pre_samples` initial points from the loss function.
            gp_params: dictionary.
                Dictionary of parameters to pass on to the underlying Gaussian Process.
            random_search: integer.
                Flag that indicates whether to perform random search or L-BFGS-B optimisation
                over the acquisition function.
            alpha: double.
                Variance of the error term of the GP.
            epsilon: double.
                Precision tolerance for floats.
        """

        print('bayesian tuning started...')
        randomSeed = 0

        # Create the GP
        if self.BayesianGpParams is not None:
            model = gp.GaussianProcessRegressor(**self.BayesianGpParams)
        else:
            kernel = gp.kernels.Matern()
            model = gp.GaussianProcessRegressor(kernel=kernel,
                                                alpha=1e-5,
                                                n_restarts_optimizer=10,
                                                normalize_y=True)

        for n in range(self.nIterBayesian):
            xp = np.array(self.BayesianHyperparameters)
            yp = np.array(self.validLoss)

            model.fit(xp, yp)

            # Sample next hyperparameter

            self.BayesianNextHyperparameters = sample_next_hyperparameter(expected_improvement, model, yp,
                                                                          greater_is_better=False,
                                                                          paramPDFs=self.bayesianHyperparamPDFs,
                                                                          n_restarts=100, randomSeed=randomSeed)

            randomSeed = randomSeed + 1
            # Sample loss for new set of parameters
            self.model_tuning('Bayesian')


def main():
    # ------------updating config values. It can also be done directly in config.json-----
    configData = {}
    configData['nTemps'] = 0 # number of feedback inputs
    configData['nFeedbacks'] = 10  # number of feedback inputs
    configData['nNodes'] = [30, 40, 50]  # number of hidden layer nodes
    configData['learningRate'] = [0.01, 0.001]  # training learning rate
    configData['decay'] = [1e-4, 1e-6]  # learning rate decay
    configData['numEpochs'] = 250  # umber of training epochs
    configData['batchSize'] = 32  # mini-batch size
    configData['preTrain'] = False # whether it is a pretrain or main training session
    configData['initMethod'] = 'random' #'random' for random initialization and 'pretrain' for using pretrained weights
    configData['kernelInit'] = 'uniform'  # input/output weights random initialization method
    configData['recurrentInit'] = 'orthogonal' #state weights random initialization method
    configData['featureSelection'] = None #PAC' for partial autocorrelation, 'PAC_ABS' for absolute value of partial
    # autocorrelation,'AC'or (absolute value of) autocorrelation, and 'none' for no feature selection
    configData['historyDepth'] = 500 #number of samples used for feature selection
    configData['activation'] = 'relu' #MLP activation function
    configData['optimizer'] = 'adam'  # solver
    configData['trainingStopLoss'] = 1e-8  # mse threshold to stop training
    configData['earlyStopPatience'] = 10  # number of epochs waited before early stopping
    configData['restoreBestWeights'] = False
    configData['architecture'] = 'MLP'  # chosse between 'RNN', 'LSTM', and 'MLP'
    configData['dropProb'] = 0.0  # dropout probabilty
    configData['preProcessing'] = 'filter' #None for no processing,  'filter' for low-pass filtering of input data,
    # 'diff' fordifferential input to MLP,'diff+filter' for low-pass filtering differential data
    configData['firFilterOrder'] = 10
    configData['firFilterCutoff'] = 0.01
    configData['normalization'] = True #(boolean): True if data is going to be normalized, False if not
    configData['detrending'] = True #(boolean): True if data is going to be de-trended, False if not
    configData['gradChecking'] = False#(boolean): True if gradChecking wanted
    configData['alignment'] = False   #(boolean): True if temp and freq alignment is needed
    configData['randomSeed'] = 0
    configData['costFunction'] = 'mse'
    configData['validationMode'] = 'CV'  # 'CV' for cross validation and 'HO' for Holdout
    configData['crossValidMode'] = 'inc'  # 'inc' for inclusive CV and 'exc' for exclusive CV
    configData['NUM_FOLD'] = 10  # number of folds for cross validation
    configData['validLength'] = 24 #validation period in hours
    configData['testLength'] = 24 # test period in hours
    configData['nIterBayesian'] = 18 #bayesian tuning iterations
    configData['BayesianGpParams'] = None
    configData['currentSamplingTime'] = 1 * 6  # sampling time of the data in file
    configData['targetSamplingTime'] = 300  # target sampling time for synthetic or file data
    configData['MIN_VALIDATION_ERROR'] = 100000  # a large number to initiate the search
    configData['N_MARGIN'] = 100 #number of margin samples between training, valid and test data
    configData['captureStart'] = 0 #start point of reading input file
    configData['captureLength'] = 30 * 24 #length of reading input file in hours
    #dict of hyperparameters for grid tuning
    configData['gridHyperparameters'] = {'nNodes': configData['nNodes'], 'learningRate': configData['learningRate'],
                                         'decay': configData['decay'],'nInputs': [max(configData['nTemps'], configData['nFeedbacks'])]}
    #hyperparameters' PDF for bayesian tuning
    configData['bayesianHyperparamPDFs'] = {0: ['integer-uniform', 10, 100, 'nNodes'], 1: ['log-uniform', -6, -1, 'lr'],
                                            2: ['log-uniform', -6, -1, 'decay'], 3: ['integer-uniform', 1, 50, 'nInputs']}

    with open('config.json', 'w') as configFile:
        json.dump(configData, configFile, indent=4)

    #--------------------------command line arguments------------------------------------------------------------------------


    parser = argparse.ArgumentParser(description='main method.')
    parser.add_argument('--input', type=str, help='Input filename. First column is sample time, second is temperature, and third column is freqdrift', default='input.csv')
    parser.add_argument('--output', type=str, help='Output filename. One column predicted Freqdrift', default='output.csv')
    parser.add_argument('--config', type=str, help='Config filename', default='config.json')
    parser.add_argument('--mode', type=str, help='t for training, i for inference, p for pretrain', default='t')
    args = parser.parse_args()


    #------------reading config file--------------------------------------------
    # read the config file
    with open(args.config, 'r') as configFile:
        configReadData = json.load(configFile)


    #---------create an object based on args and config data-------------------

    
    currentPath = os.getcwd() + '/'
    configReadData['inputFile'] = currentPath+args.input
    configReadData['outputFile'] = currentPath+args.output
    phaseErrors={}
    if args.mode == 't':
        configReadData['NUM_FOLD'] = int(configReadData['captureLength']/24 - 2)
        myModel = Holdover_Model(**configReadData)
        # write config data also to log
        if os.path.exists(currentPath +'logs') == False:
            os.mkdir('logs')
        if os.path.exists(currentPath + 'figs') == False:
            os.mkdir('figs')
        with open('logs/log.json', 'w') as logFile:
            json.dump(configReadData, logFile, indent=4)
        myModel.data_preprocessing()
        myModel.model_tuning('grid')
        myModel.bayesian_optimisation()
        myModel.model_log()


    elif args.mode == 'i':
        with open('model.pickle', 'rb') as modelFile:
            modelReadData = pickle.load(modelFile)

        myModel = Inference_Model(**configReadData)
        myModel.sample_pre_processing(**modelReadData)
        myModel.sample_predict()
        myModel.sample_log()

    elif args.mode == 'p':
        #---move the path to preTrain folder and log everything there---
        if os.path.exists(currentPath +'preTrain') == False:
            os.mkdir('preTrain')
        os.chdir('preTrain')
        currentPath = os.getcwd() + '/'
        configReadData['preTrain'] = True
        if os.path.exists(currentPath + 'logs') == False:
            os.mkdir('logs')
        if os.path.exists(currentPath + 'figs') == False:
            os.mkdir('figs')
        with open('logs/log.json', 'w') as logFile:
            json.dump(configReadData, logFile, indent=4)
        myModel = Holdover_Model(**configReadData)
        myModel.data_preprocessing()
        myModel.model_tuning('grid')
        myModel.model_log()
if __name__ == '__main__':
    main()

