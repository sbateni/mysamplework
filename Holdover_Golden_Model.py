import os, shutil, sys
import json
import pickle
import argparse
import multiprocessing
from itertools import product

import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
from scipy import signal, interpolate
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import acf, pacf
from termcolor import cprint
import time
from Holdover_Utility import *

import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Holdover_Inference_Model():
    """
    base class for inference model
    """

    def __init__(self, **configData):
        """
        Arguments:
        configData: (dict) the config dict read from  config.json
        self.inputFile: (str)the path to input csv file with columns(timestamp, temperature, splitXO, freq drift)
        self.modelLog : (dict) containing {'nFeedbacks': self.nFeedbacks, 'nTemps': self.nTemps,'nSplits': self.nSplits, 'featureLags':
            self.featureLags, 'historyDepth': self.historyDepth}  for selected model
        self.inputData : (str)'temp' for temperature and 'split' for splitXO. also could be 'temp+split' or 'split+split'
        self.rnnFeatures : (int) number of rnn features (each of freq, temp and split counts for one)
        self.featureSelection : (str)PAC' for partial autocorrelation, 'PAC_ABS' for absolute value of partial
            autocorrelation,'AC' for (absolute value of) autocorrelation, and 'none' for no feature selection
        self.currentSamplingTime : (float)sampling time of the data in input file
        self.targetSamplingTime : float)target sampling time
        self.historyDepth :(int)max number of previous samples used for feature selection
        self.N_MARGIN_INFER : (int) number of extra (target)samples read for every inference
        self.decimationFilterTaps :  (nd.array) taps of decimation FIR filter
        self.trend : (linear regression object) for inference data (calculated in Holdover_Driver.inference_detrend())
        self.architecture : (str) architecture of NN
        self.trainHistory :(dict) dictionary containing trained model parameters and training history(loss,..)  for selected model
        self.sampleInput :  (nd.array) input data (temp+split+freq drift) for the sample being predicted
        self.samplePrediction : (float) prediction(inference) for one sample
        self.bufferLog : (dict)buffer to keep track of history of freq drift and coresponding timestamps in successive runs. It will be written to buffer.pickle each time and read back next time
           outputs:
        self.resultDir + '/buffer.pickle': (str) pickle file containing bufferLog dictionary
        self.timeStamps : (nd.array)timestamps of input data
        self.linearRegPredict : (obj) linear regression of input data
        self.biasPredict : (float) the value should be added to prediction if in diff mode
        self.freqDrift : (nd.array)normalized history of freq drift used for sample inference (last historeDepth valuus)
        self.temperature : (nd.array)normalized temperature used for sample inference (last n_temp values)
        self.split : (nd.array) normalized split xo used for sample inference (last n_split values)
        self.sampleActualOutput : (float) actual output for sample being predicted
        self.sampleTimeStamp : (int) timestamp of sample being predicted
        self.sampleInputData : (nd.array) input data (temp+split) for the sample being predicted(for logging and plotting purpose)
        self.sampleInput : (nd.array)input data (temp+split+freq drift) for the sample being predicted
        self.normalization : (str)'standard' for standardization , 'norm' for normalization
        self.samplePrediction :(float) prediction(inference) for one sample
        self.featureType :(str)None for regular features,'diff' for differential features
        self.normMean :(float) mean of freq drift training data
        self.normScale : (float)scale of freq drift training data
        self.outputFile : (str) csv file with columns (time of day, time stamp(s), actual output(ppb), prediction(ppb), input data)

              """
        self.__dict__.update(configData)  # all configData become instance variables
        self.bufferLog = {} #buffer to kjeep track of history of freq drift in successive runs. It will be written to buffer.pickle each time and read back next time

    def run(self):
         """
         runner function of model. runs the model in order of pre_processing, prediction or inference and logs the results
         inputs :
         self.resultDir + '/model.pickle' :pickle file containing self.trainHistory and self.modellog for selected model. refer to sample_pre_processing() comments for details
         """

         with open(self.resultDir + '/model.pickle', 'rb') as modelFile:
             modelReadData = pickle.load(modelFile)

         self.sample_pre_processing(**modelReadData)
         self.sample_predict()
         self.sample_log()


    def sample_pre_processing(self, **modelData):
        """
        
        reads the necessary amount of data for one sample prediction from self.inputfile, performs pre-processing(resampling, decimation, feature engineering, and normalization) on data

        Arguments:
        modelData: the model dict containing trained parameters and other mathematical parameters of model
        containing  ('modelLog', 'trainHistory', 'normScale', 'normMean','tempNormMean', 'tempNormScale',
                         'splitNormMean', 'splitNormScale', 'trend', 'decimationFilterTaps')} for selected model
        """

        self.__dict__.update(modelData)  # all modelData become instance variables
        N_FEEDBACK = int(self.modelLog['nFeedbacks'])
        N_TEMP = int(self.modelLog['nTemps'])
        if self.inputData in ['split+temp','split+split']:
            N_SPLIT = N_TEMP
        if self.inputData not in ['split+temp','split+split']:
            self.rnnFeatures = 2
        else:
            self.rnnFeatures = 3

        if N_FEEDBACK == 0 or N_TEMP == 0:
            self.rnnFeatures = self.rnnFeatures - 1
        lag = np.maximum(N_FEEDBACK, N_TEMP)

        if self.featureSelection is None:
            self.historyDepth = lag

        self.downsampleRatio = int(self.targetSamplingTime / self.currentSamplingTime)
        #------read enough samples from the end of the file for predicting one sample
        length = int((self.historyDepth + self.N_MARGIN_INFER) * self.downsampleRatio)
        readDataWhole = np.genfromtxt(self.inputFile, delimiter=',', skip_header=1)
        readDataWindow = readDataWhole[~np.isnan(readDataWhole).any(axis=1)][-length:,:]

        # ---fixing the sampling times---
        actualTimeStamps = readDataWindow[:, 0]
        desiredTimeStamps = np.arange(actualTimeStamps[-1], actualTimeStamps[0], -self.currentSamplingTime)[::-1]
        tempSampleFixed = np.interp(desiredTimeStamps, actualTimeStamps, readDataWindow[:, 1])
        splitSampleFixed = np.interp(desiredTimeStamps, actualTimeStamps, readDataWindow[:, 2])
        freqSampleFixed = np.interp(desiredTimeStamps, actualTimeStamps, readDataWindow[:, 3])

        if self.inputData == 'temp':
            readData = np.array([desiredTimeStamps, tempSampleFixed, freqSampleFixed]).T
        elif self.inputData == 'split':
            readData = np.array([desiredTimeStamps, splitSampleFixed, freqSampleFixed]).T
        elif self.inputData == 'split+temp':
            readData = np.array([desiredTimeStamps, tempSampleFixed, freqSampleFixed, splitSampleFixed]).T
        elif self.inputData == 'split+split':
            readData = np.array([desiredTimeStamps, splitSampleFixed, freqSampleFixed, splitSampleFixed]).T

        # ----perform iir post filtering
        if self.postFilter:
            zi = signal.lfilter_zi(self.iirNum, self.iirDen)
            filteredData, _ = signal.lfilter(self.iirNum, self.iirDen, readData, axis=0,
                                             zi=np.dot(np.expand_dims(zi, axis=1), readData[0:1, :]))
            self.readData = filteredData
        else:
            self.readData = readData

        # --keep unfiltered data to compare with predictions
        self.rawFreq = np.convolve(readData[:, 2], self.decimationFilterTaps, 'valid')[::-1][
                       ::self.downsampleRatio][::-1]
        # --------low-pass filtering and decimation----------------

        data1 = np.convolve(self.readData[:, 1], self.decimationFilterTaps, 'valid')[::-1][::self.downsampleRatio][::-1]
        data2 = np.convolve(self.readData[:, 2], self.decimationFilterTaps, 'valid')[::-1][::self.downsampleRatio][::-1]
        data = np.vstack((data1, data2)).T
        numData = data1.size
        self.timeStamps = self.readData[:, 0][self.decimationFirFilterOrder - 1:][::-1][::self.downsampleRatio][::-1]
        if self.inputData in ['split+temp','split+split']:
            data3 = np.convolve(self.readData[:, 3], self.decimationFilterTaps, 'valid')[::-1][::self.downsampleRatio][
                    ::-1]
            data = np.vstack((data1, data2, data3)).T
        self.rawData = np.copy(data)
        # ---feature engineering (diff or detrend)
        # ------------linear de-trending----------------------------
        if self.detrending:
            self.linearRegPredict = self.trend.predict(self.timeStamps.reshape((numData, 1)))
            data[:, 1] = data[:, 1] - self.linearRegPredict[:, 0]


        # -------Data featureType---------


        if 'diff' in self.featureType:
            self.biasPredict, self.biasActual = self.rawData[:, 1][-2], self.rawData[:, 1][-2]
            self.freqDrift = np.diff(data[:, 1])
            self.temperature = np.diff(data[:, 0])
            self.timeStamps = np.delete(self.timeStamps, 0)
            if self.inputData in ['split+temp','split+split']:
                self.split = np.diff(data[:, 2])

        else:
            self.freqDrift = np.copy(data[:, 1])
            self.temperature = np.copy(data[:, 0])
            if self.inputData in ['split+temp','split+split']:
                self.split = np.copy(data[:, 2])

        # -----normalizing data---------------
        if self.normalization:
            self.freqDrift = self.freqDrift - self.normMean
            self.freqDrift = self.freqDrift / self.normScale
            self.temperature = self.temperature - self.tempNormMean
            self.temperature = self.temperature / self.tempNormScale
            if self.inputData in ['split+temp','split+split']:
                self.split = self.split - self.splitNormMean
                self.split = self.split / self.splitNormScale
        #------------------------preparing the input data for sample prediction------
        self.freqDrift = self.freqDrift[:-1]
        self.sampleActualOutput = self.rawFreq[-1]
        self.sampleTimeStamp = self.timeStamps[-1]
        self.sampleInputData = self.rawData[-1,0]
        if self.inputData in ['split+temp','split+split']:
            self.sampleInputData = np.hstack((self.rawData[-1, 0], self.rawData[-1, 2]))

        if os.path.exists(self.resultDir + '/buffer.pickle')== False:
            self.bufferLog = {key: self.__dict__[key] for key in
                         ('freqDrift', 'timeStamps')}
        else:
            with open(self.resultDir + '/buffer.pickle', 'rb') as pickleBuffer:
                self.bufferLog = pickle.load(pickleBuffer)
            # fixing the history inputs on precise timestamps
            lastTimeStamps = self.bufferLog['timeStamps']
            self.bufferLog['freqDrift'] = np.interp(self.timeStamps[:-1], lastTimeStamps, self.bufferLog['freqDrift'])
            self.freqDrift = self.bufferLog['freqDrift']
        self.bufferLog['timeStamps'] = self.timeStamps[1:]
        if self.inputData not in ['split+temp','split+split']:
            self.sampleInput = np.expand_dims(
                np.hstack((self.temperature[- N_TEMP:],
                self.freqDrift[-(self.historyDepth):])),axis=0)
        else:
            self.sampleInput = np.expand_dims(
                np.hstack((self.temperature[- N_TEMP:],self.split[- N_SPLIT:],
                           self.freqDrift[-(self.historyDepth):])), axis=0)


    def sample_predict(self):
        """
        predicts one sample of inference using self.sampleInput data prepared in sample_pre_processing

        inputs:

        """

        N_FEEDBACK = int(self.modelLog['nFeedbacks'])
        N_TEMP = int(self.modelLog['nTemps'])
        if self.architecture == 'MLP':
            self.samplePrediction, caches = mlp_forward_propagation(self.sampleInput[:, self.modelLog['featureLags']].T, self.trainHistory['trainedParameters'],
                                                                 self.activation)
        if self.architecture == 'RNN':
            self.samplePrediction, _, _ = rnn_forward_propagation(
                np.reshape(self.sampleInput[:, self.modelLog['featureLags']], (1, np.maximum(N_FEEDBACK, N_TEMP), self.rnnFeatures),
                           order='F'), self.trainHistory['trainedParameters'])
        elif self.architecture == 'LSTM':
            self.samplePrediction, _, _ = lstm_forward_propagation(
                np.reshape(self.sampleInput[:, self.modelLog['featureLags']], (1, np.maximum(N_FEEDBACK, N_TEMP), self.rnnFeatures),
                           order='F'), self.trainHistory['trainedParameters'])
        #--update buffer.pickle with new prediction----------------------------
        self.bufferLog['freqDrift'] = np.roll(self.bufferLog['freqDrift'], -1)
        self.bufferLog['freqDrift'][-1] = self.samplePrediction[0]
        print('sample infered for time stamp = ', self.sampleTimeStamp)
        with open(self.resultDir +'/buffer.pickle', 'wb') as bufferFile:
            pickle.dump(self.bufferLog, bufferFile)
        print(f"autoregressive inputs for next of {self.sampleTimeStamp/3600} are {self.bufferLog['freqDrift']}")

    def sample_log(self):
        """
        process back the sample prediction and log it to output file, updates buffer.pickle with new prediction



        """

        # ----de-normalizing prediction---------
        if self.normalization:
            self.samplePrediction = self.samplePrediction * self.normScale + self.normMean
        #---reverting feature engineering (diff or detrend)
        if 'diff' in self.featureType:
            if os.path.exists(self.outputFile):
                with open(self.outputFile, 'r') as outputFile:
                    biasReadBack = np.atleast_2d(np.loadtxt(outputFile, dtype=np.float_, delimiter=',', skiprows=1))
                    self.biasPredict = biasReadBack[-1,3]
                    self.biasActual = biasReadBack[-1,2]

            self.samplePrediction = self.samplePrediction + self.biasPredict

        if self.detrending:
            sampleLinearReg = self.trend.predict(self.timeStamps[-1].reshape(1, -1))
            self.samplePrediction = self.samplePrediction + sampleLinearReg


        # ---writing [tod, timestamp, sample actual output, sample prediction] to the output file
        with open(self.outputFile, 'a') as outputFile:
            if self.inputData not in ['split+temp','split+split']:
                if os.stat(self.outputFile).st_size == 0:
                    np.savetxt(outputFile, [np.array(
                        [time.time(), self.sampleTimeStamp, self.sampleActualOutput, self.samplePrediction,
                         self.sampleInputData])],
                               delimiter=',',
                               header='time of day, time stamp(s), actual output(ppb), prediction(ppb), input data')
                else:
                    np.savetxt(outputFile, [np.array(
                        [time.time(), self.sampleTimeStamp, self.sampleActualOutput, self.samplePrediction,
                         self.sampleInputData])], delimiter=',')
            else:
                if os.stat(self.outputFile).st_size == 0:
                    np.savetxt(outputFile, [np.array(
                        [time.time(), self.sampleTimeStamp, self.sampleActualOutput, self.samplePrediction,
                         self.sampleInputData[0], self.sampleInputData[1]])],
                               delimiter=',',
                               header='time of day, time stamp(s), actual output(ppb), prediction(ppb), temp, split')
                else:
                    np.savetxt(outputFile, [np.array(
                        [time.time(), self.sampleTimeStamp, self.sampleActualOutput, self.samplePrediction,
                         self.sampleInputData[0],self.sampleInputData[1]])], delimiter=',')
        print(f'prediction for {self.sampleTimeStamp/3600} is {self.samplePrediction}')

class Holdover_Train_Model():
    """
    base class for holdover model
    """

    def __init__(self, *args, **configData):
        """
        Arguments:
        configData: the config dict defined in Holdover_Driver class init and written to config.json
        args: possible future use

        self.gridHyperparameters : (dictionary) containing the grid tuning hyperparameters. see configData for more details
        self.BayesianNextHyperparameters : (nd.array) of hyperparameters chosen by Bayesian tuning, the order is same as self.gridHyperparameters
        self.nTemps : (int)number of temperature and/or split inputs
        self.nFeedbacks : (int)number of feedback(autoregressive) inputs
        self.featureLags: (list) lags (indices) of selected feature in [0,self.historyDepth]
        self.inputData : (str)'temp' for temperature and 'split' for splitXO. also could be 'temp+split' or 'split+split'
        self.detrendLength : (float list) length for piecewise detrending in hours
        self.detrending: (boolean): True if data is going to be de-trended, False if not
        self.initMethod : (boolean)whether it is a pre-training(True) or main training(False) session
        self.resultDir: (str)directory containing train results
        self.architecture : (str)architecture of NN
        self.laggedTrainData: (tuple) of (inputTrainData, outputTrainData, timestamps) after feature selection
        self.laggedValidData : (tuple) of (inputValidData, outputValidData, timestamps) after feature selection
        self.rnnFeatures : (int)number of rnn features (each of freq, temp and split counts for one)
        self.trainData : (tuple)of (inputTrainData, outputTrainData, timestamps) before feature selection
        self.ValidData: (tuple) of (inputValidData, outputValidData, timestamps) before feature selection
        self.testData : (tuple) of (inputTestData, outputTestData, timestamps) before feature selection
        self.validationMode : (str)'CV' for cross validation and 'HO' for Holdout
        self.normalization: (str) 'standard' for standardization , 'norm' for normalization
        self.normMean : (float) mean(standard) or min(normalize) of training freq data after possible detrending or diff
        self.normScale: (float)std(standard) or max - min (normalize) of training freq data after possible detrending or diff
        self.tempNormMean:(float) mean(standard) or min(normalize) of training temp data
        self.tempNormScale: (float)mean(standard) or min(normalize) of training temp data
        self.activation: (str)MLP activation function . Could be 'relu', 'tanh' or 'sigmoid'
        self.linearRegPredict: (nd.array)linear regression line(s) for freq data
        self.featureType: (str)'none' for regular features,'diff' for differential features
        self.targetTestSamples: (int)number of test samples in target sampling rate
        self.bayesianHyperparamPDFs :(dict) dictionary containing hyperparameters' PDF for bayesian tuning
        self.paramMean: (nd.array) mean of hyper parameters to map them to [-1,1]
        self.paramScale:(nd.array) scale of hyper parameters to map them to [-1,1]
        self.confidenceData: (list) of tuples of (input,output,timestamp) for all confidence windows
        self.numGridTunings:(int) number of gird tuning iterations
        self.phaseDiff: (nd.array) of prediction phase error
        self.phaseDiffUncomp: (nd.array) of uncompensated phase error
        self.numEpoch: (int)number of elapsed epoch
        self.validLog: (dict) containing {
                                                     'training error(mae)': errors[0],
                                                     'validation error(mae)': errors[1], 'test error(mae)': errors[2],
                                                     'numElapsedEpochs': self.numEpoch, 'numMiniBatch':numMinibatch, 'number of trained weights': totalNumWeights, 'PktoPk phase error(ns)': phaseError,
                                                     'Uncomp phase error(ns)': phaseErrorUncomp, 'tuning': tuningAlg,
                                                     'Temp Phase error(ns)':phaseErrorTemp}  for each set of hyperparameters
        self.modelLog: (dict) containing {'nFeedbacks': self.nFeedbacks, 'nTemps': self.nTemps,'nSplits': self.nSplits, 'featureLags':
            self.featureLags, 'historyDepth': self.historyDepth}  for all sets of hyperparameters
        self.trainHistory: (dict) of trainhistory  for all sets of hyperparameters
        self.actualOutput: (dict) of actual outputs  for all sets of hyperparameters
        self.confidenceDataDict: (dict) of self.confidenceData  for all sets of hyperparameters
        self.BayesianHyperparameters: (list) of normalized hyperparameters sets (will be converted to 2-D array of (hyperParameters,number of tuning iterations))
        self.validLoss: (list) of validations (will be converted  to 1-D of shape (number of tuning iterations,))

        self.numGridTunings:(int) number of gird tuning iterations
        self.startTime: (int) start time of training
        self.outputFile: (str) csv file containing 'time stamp(s), actual output(ppb), prediction(ppb)' for selected model
        self.logDir +'/log.json':(str) json file containing self.validlog dict
        self.resultDir +'/model.pickle' :(str) pickle file containing self.trainHistory and self.modellog for selected model
        self.selectedModel: (str)hyperparams for the chosen model

        self.nNodes : (int)number of hidden nodes in MLP and number of hidden states in RNN or LSTM
        self.randomSeed: (int)random generator initial seed
        self.initMethod:  (str)'random' for random initialization and 'pretrain' for using pretrained weights
        self.kernelInit :(str) input-> output weights random initialization method. 'uniform', 'normal',
             or 'orthogonal'(only for RNN and LSTM)
        self.recurrentInit :(str) state weights random initialization method (only for RNN and LSTM).
             'uniform', 'normal', or 'orthogonal'
        self.optimizer : (str) solver(adam)
        self.learningRate : (float)learning rate for gradient descent backpropagation algorithm
        self.decay: (float)decay of learning rate for gradient descent backpropagation algorithm
        self.batchSize: (int)mini-batch size
        self.dropProb : (float) dropout probabilty for MLP
        self.costFunction: (str)training cost function. 'mse' or 'mae'. Also'huber' only for MLP
        self.gradChecking : (boolean): True if gradChecking wanted
        self.earlyStopPatience : (int)number of epochs waited before early stopping of training
        self.trainingStopLoss : (float) threshold to stop training
        self.NUM_FOLD: (int)number of folds for cross validation
        self.inputFile: (str)the path to input csv file with columns(timestamp, temperature, splitXO, freq drift)
        self.testLength : (float) length of test period in hours
        self.validLength : (float) length of validation period in hours
        self.decimationFirFilterOrder : (int)order of low-pass FIR filter used for decimation
        self.decimationFilterTaps :(nd.array) taps of decimation FIR filter
        self.N_MARGIN : (int)number of margin samples between training, valid and test data

        self.downsampleRatio : (float)ratio of down sampling(decimation)
        self.targetHourSamples : (int)number of samples in an hour with target sampling time
        self.targetTestSamples :(int) number of samples in a day with target sampling time

        self.linearRegPredicts :(nd.array) 2-D array of shape (len(data), len(self.detrendLength) containing linear reg line(s) for all detrendLenghts
        self.freqDrifts : (nd.array)2-D array of shape (len(data), len(self.detrendLength) containing normalized freq drift data for all detrendLenghts
        self.temperature :(nd.array) normalized temperature data
        self.timeStamps :(nd.array) timestamps of data taken from input file
        self.split : (nd.array)normalized split data
        self.freqBias :(float) first value of freq drift before diff mode
        self.tempBias :(float) first value of temp before diff mode
        self.splitBias : (float) first value of split before diff mode
        self.normMeans :(nd.array) means used for normalizing self.freqdrifts
        self.normScales : (nd.array)scales used for normalizing self.freqdrifts
        self.normMeanDict : (dict) dictionary of norm means for all models
        self.normScaleDict : (dict) dictionary of norm scales for all models
        self.splitNormMean :(float) mean of normalizing split
        self.splitNormScale : (float) scale of normalizing split
        self.detrendIndice : (int)selected indice for detrend length list
        self.confidenceData : (list) of tuples of (input,output,timestamp) for all confidence windows

        """
        print('training started....')
        self.startTime = time.process_time()
        self.__dict__.update(configData) #all configData become instance variables
        # ----------create log and fig folders
        self.logDir = self.resultDir + '/logs'
        self.figDir = self.resultDir + '/figs'
        if os.path.exists(self.logDir) == True:
            shutil.rmtree(self.logDir)
        os.mkdir(self.logDir)
        if os.path.exists(self.figDir) == True:
            shutil.rmtree(self.figDir)
        os.mkdir(self.figDir)
        self.nSplits = self.nTemps
        self.validLog = {}
        self.modelLog = {}
        self.trainHistory = {}
        self.prediction = {}
        self.actualOutput = {}
        self.confidenceDataDict = {}
        self.confidenceLogDict = {}
        self.normMeanDict = {}
        self.normScaleDict={}
        self.BayesianHyperparameters = []
        self.validLoss = []
        self.errors = []
        self.modelCounter = 0
        self.sumEpoch = 0
        self.trend = None
        if self.inputData in ['split+temp', 'split+split']:
            self.inputLabels = self.inputData.split('+')
        #----------mean and scale of hyperparameters for normalization
        nParams = len(self.bayesianHyperparamPDFs.keys())
        self.paramMean = np.zeros((nParams,))
        self.paramScale = np.ones((nParams,))
        for counter,key in enumerate(list(self.bayesianHyperparamPDFs.keys())):
            self.paramMean[counter] = 1. / 2 * (
                        self.bayesianHyperparamPDFs[key][1] + self.bayesianHyperparamPDFs[key][2])
            self.paramScale[counter] = self.bayesianHyperparamPDFs[key][2] - self.paramMean[counter]



    def model_tune(self, tuningAlg='grid'):
        """
        main function for (grid and bayesian) tuning on top of model_train() and model_cross_validate() functions
        trains model for each set of hyperparameters and captures the  results
        Arguments:
        tuningAlg: grid or bayesian


        """
        # ------------------tuning and validation------------------------------------------
        hyperParameterDict = {}
        self.tuningAlg = tuningAlg
        if tuningAlg == 'grid':
            hyperParameterSets = product(*list(self.gridHyperparameters.values()))
            self.numGridTunings = np.prod([len(x) for x in list(self.gridHyperparameters.values())])
        elif tuningAlg == 'Bayesian':
            hyperParameterSets = [list(self.BayesianNextHyperparameters)]
        for hyperParameterSet in hyperParameterSets:
            dictKey = '['
            for counter, key in enumerate(self.gridHyperparameters.keys()):
                hyperParameterDict[key] = hyperParameterSet[counter]
                if key == 'nInputs':
                    hyperParameterDict['nTemps'] = hyperParameterSet[counter] * np.sign(self.nTemps)
                    hyperParameterDict['nFeedbacks'] = hyperParameterSet[counter] * np.sign(self.nFeedbacks)
                    hyperParameterDict['nSplits'] = 0
                    if self.inputData in ['split+temp','split+split']:
                        hyperParameterDict['nSplits'] =  hyperParameterDict['nTemps']
                #---formating the model params for printing
                if isinstance(hyperParameterSet[counter], int):
                    if key == 'detrendIndice':
                        dictKey = dictKey + "{:<4d}, ".format(self.detrendLength[hyperParameterSet[counter]])
                    else:
                        dictKey = dictKey + "{:<3d}, ".format(hyperParameterSet[counter])
                elif isinstance(hyperParameterSet[counter], float):
                    dictKey = dictKey + "{:.2e}, ".format(hyperParameterSet[counter])
            dictKey = dictKey.rstrip(', ') + ']'
            self.__dict__.update(hyperParameterDict)
            self.data_windowing()  # prepare windowed data for training

            if self.initMethod == 'pretrain':  #---read pretrained parameters for initialization
                with open(self.resultDir +'/preTrainedModel.pickle', 'rb') as pickleLog:
                    self.initParameters = pickle.load(pickleLog)['trainedParameters']
            self.modelCounter = self.modelCounter+1
            dictKey = str(self.modelCounter)+dictKey
            print('training for model '+str(self.modelCounter)+':'+self.architecture+' hyperparameters:' + dictKey + '...')
            if self.validationMode == 'CV':
                trainHistory, validationError, validPrediction, validOutput = self.model_cross_validate()
            elif self.validationMode == 'HO':
                trainHistory = self.model_train(self.laggedTrainData, self.laggedValidData)

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
                dataPrediction, dataEstimation = predict(self, data[0], trainHistory['trainedParameters'])
                dataOutput = data[1].T

                # ----de-normalizing, cumsum and de-trending predictions---------
                if self.normalization:
                    dataPrediction = dataPrediction * self.normScale + self.normMean
                    dataEstimation = dataEstimation * self.normScale + self.normMean
                    dataOutput = dataOutput * self.normScale + self.normMean
                    if i == 1:
                        #temperature input scaled to output
                        tempPrediction = np.expand_dims(data[0][:,self.nTemps - 1] * self.normScale + self.normMean, axis=1)
                        self.tempInput = np.expand_dims(data[0][:, self.nTemps - 1] * self.tempNormScale + self.tempNormMean,axis=1)
                if 'diff' in self.featureType:
                    biasIndice = np.argwhere(self.timeStamps == data[2][0])
                    dataPrediction = np.cumsum(dataPrediction, axis=1) + self.freqBias[biasIndice]
                    dataOutput = np.cumsum(dataOutput, axis=1) + self.freqBias[biasIndice]
                    if i==1:
                        tempPrediction = np.cumsum(tempPrediction, axis=0) + self.freqBias[biasIndice]
                        self.tempInput = np.cumsum(self.tempInput, axis=0) + self.tempBias[biasIndice]
                if self.detrending and i==1:
                    dataPrediction = dataPrediction + self.linearRegPredict[-self.targetTestSamples:].reshape(1,self.targetTestSamples)
                    dataOutput = dataOutput + self.linearRegPredict[-self.targetTestSamples:].reshape(1,self.targetTestSamples)
                    tempPrediction = tempPrediction + self.linearRegPredict[-self.targetTestSamples:].reshape(tempPrediction.size, 1)

                mae = (metrics.mean_absolute_error(dataOutput, dataPrediction))
                predictions.append(dataPrediction.T)
                errors.append(mae)
                actualOutputs.append(dataOutput.T)

            self.phaseDiff, self.phaseDiffUncomp = phase_diff(actualOutputs[2], predictions[2], self.targetSamplingTime)
            phaseDiffTemp, _ = phase_diff(actualOutputs[2], tempPrediction, self.targetSamplingTime)
            phaseErrorTemp = abs(np.amax(phaseDiffTemp) - np.amin(phaseDiffTemp))
            phaseError = abs(np.amax(self.phaseDiff) - np.amin(self.phaseDiff))
            phaseErrorUncomp = abs(np.amax(self.phaseDiffUncomp) - np.amin(self.phaseDiffUncomp))
            phaseErrors.append(phaseError)
            self.numEpoch = len(trainHistory['loss'])
            numMinibatch = int(self.trainData[0].shape[0] / self.batchSize)
            totalNumWeights = 0
            for key,value in trainHistory['trainedParameters'].items():
                totalNumWeights = totalNumWeights + value.size
            validationError = errors[1]
            # ------------logging results in a dict------------------------------
            self.sumEpoch = self.sumEpoch + self.numEpoch
            self.validLog[dictKey] = {
                                                     'training error(mae)': errors[0],
                                                     'validation error(mae)': errors[1], 'test error(mae)': errors[2],
                                                     'numElapsedEpochs': self.numEpoch, 'numMiniBatch':numMinibatch, 'number of trained weights': totalNumWeights, 'PktoPk phase error(ns)': phaseError,
                                                     'Uncomp phase error(ns)': phaseErrorUncomp, 'tuning': tuningAlg,
                                                     'Temp Phase error(ns)':phaseErrorTemp}
            self.modelLog[dictKey] = {'nFeedbacks': self.nFeedbacks, 'nTemps': self.nTemps,'nSplits': self.nSplits, 'featureLags':
            self.featureLags, 'historyDepth': self.historyDepth}
            self.trainHistory[dictKey] = trainHistory
            self.prediction[dictKey] = predictions
            self.actualOutput[dictKey] = actualOutputs
            self.errors.append(errors)
            self.confidenceDataDict[dictKey] = self.confidenceData
            self.normMeanDict[dictKey] = self.normMean
            self.normScaleDict[dictKey] = self.normScale
            # --------------------updating validation error history for bayesian tuning-----
            if (np.isnan(validationError) == False and np.isfinite(validationError) == True):
                normHyperParameterSet = []
                for counter, key in enumerate(list(self.bayesianHyperparamPDFs.keys())):
                    #normalizing all bayesian hyperparameters uniformly to [-1,1]
                    if self.bayesianHyperparamPDFs[key][0] == 'log-uniform':
                        normHyperParameterSet.append(
                            (np.log10(hyperParameterSet[counter]) - self.paramMean[counter]) / self.paramScale[counter])
                    elif self.bayesianHyperparamPDFs[key][0] == 'integer-uniform':
                        normHyperParameterSet.append((hyperParameterSet[counter] - self.paramMean[counter]) / self.paramScale[counter])
                self.BayesianHyperparameters.append(normHyperParameterSet)
                self.validLoss.append(np.minimum(validationError, self.MIN_VALIDATION_ERROR))
                
                print('train error =', errors[0])
                print('valid error=', errors[1])
                print('test error=', errors[2])
                self.plot_train(dictKey)

    def plot_train(self, dictKey):
        """
        plot training plots for one model
        :param dictKey: dict key of the model
               """

        # --------------plotting graph results to a file---------------
        if self.plot :
           
            plt.figure(figsize=(15, 7))
            plt.suptitle(self.oscillator+' '+self.architecture + ': ' + dictKey[dictKey.find('['):], fontweight='bold')
            plt.subplot(231)
            plt.xlabel('time(hours)')
            plt.title('pre-process: '+self.featureType + ', ' +'detrend: '+str(self.detrending))
            plt.ylabel('frequency offset(normalized) ')
            plt.plot(self.trainTimeStamps / 3600,  self.actualOutput[dictKey][0], 'b', label='actual_train')
            plt.plot(self.trainTimeStamps / 3600, self.prediction[dictKey][0], 'r', label='prediction_train')
            plt.grid(True)
            plt.legend(fontsize='x-small')

            plt.subplot(233)
            plt.xlabel('epochs')
            plt.title('patience(epochs): ' + str(
                    self.earlyStopPatience) + ', ' + 'maxEpochs: ' + str(self.numEpochs))
            plt.ylabel('mse')
            plt.plot(np.arange(1, self.numEpoch+1), self.trainHistory[dictKey]['loss'], 'b', label='training loss')
            plt.plot(np.arange(1, self.numEpoch+1), self.trainHistory[dictKey]['validLoss'], 'r', label='holdout valid loss')
            plt.grid(True)
            plt.legend(fontsize='x-small')

            plt.subplot(232)
            plt.title('pre-process: '+self.featureType + ', ' + 'detrend: ' + str(self.detrending))
            plt.xlabel('time(hours)')
            plt.ylabel('frequency offset(normalized)')
            plt.plot(self.validTimeStamps/3600, self.trainHistory[dictKey]['validPred'], 'r', label='prediction_valid for training')
            plt.plot(self.validTimeStamps/3600, self.validData[1], 'b', label='actual_valid for training')
            plt.grid(True)
            plt.legend(fontsize='x-small')

            plt.subplot(234)
            plt.xlabel('time(hours)')
            plt.ylabel('frequency offset(ppb)')
            if self.validationMode == 'CV':
                plt.title('number of folds for cross-valid: ' + str(self.NUM_FOLD))
                plt.plot(self.trainTimeStamps[:self.actualOutput[dictKey][1].size]/3600, self.actualOutput[dictKey][1], 'b',
                         label='actual_cross valid')
                plt.plot(self.trainTimeStamps[:self.prediction[dictKey][1].size]/3600, self.prediction[dictKey][1], 'r',
                         label='prediction_cross valid')
            elif self.validationMode == 'HO':
                plt.plot(self.validTimeStamps/3600, self.actualOutput[dictKey][1], 'b',
                         label='actual_valid')
                plt.plot(self.validTimeStamps/3600, self.prediction[dictKey][1], 'r',
                         label='prediction_valid')
            plt.grid(True)
            plt.legend(fontsize='x-small')

            ax1 = plt.subplot(235)
            ax1.set_xlabel('time(hours)')
            ax1.set_ylabel('frequency offset(ppb)')
            ax1.plot(self.testData[2] / 3600, self.prediction[dictKey][2], 'r', label='prediction_test')
            ax1.plot(self.testData[2] / 3600, self.actualOutput[dictKey][2], 'b', label='actual_test')
            ax1.tick_params(axis='y')
            ax1.legend(fontsize='x-small', loc=2)
            ax1.grid(True)
            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('temp sensor value', color=color, fontsize='small')
            ax2.plot(self.testData[2] / 3600, self.tempInput, 'g', label=self.inputData)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.yaxis.set_tick_params(labelsize='small')
            plt.tight_layout()
            ax2.legend(fontsize='x-small', loc=1)

            time_test = self.testData[2] / 3600
            ax3 = plt.subplot(236)
            ax3.plot(time_test[1:], self.phaseDiff, 'r', label='pred phase error')
            ax3.plot(time_test[1:], self.phaseDiffUncomp, 'b', label='Uncomp phase error')
            ax3.set_xlabel('time(hours)')
            ax3.set_ylabel('phase_error(ns)', fontsize='small')
            ax3.yaxis.set_tick_params(labelsize='small')
            ax3.grid(True)
            ax3.legend(fontsize='x-small')
            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                                wspace=0.35)
            plt.savefig(self.figDir +'/' + self.architecture +
                        dictKey + '.png')
            plt.close('all')

    def model_log(self):

        """
        writes the logged data to json and selected model info to  pickle files and prints the final results


        """
        # --------------------validation accuracy analysis-----------------------

        #--creating label for figure names to save
        if self.detrending == True:
            self.figLabel = self.architecture+'_detrend_'+self.inputData
        elif self.featureType == 'diff':
            self.figLabel = self.architecture + '_diff_'+self.inputData
        else:
            self.figLabel = self.architecture + '_' + self.inputData
        errorArray = np.array(self.errors, dtype=np.float_)
        validationAccuracyIndex, validationMAE = self.valid_eval(errorArray[:, 1], errorArray[:, 2])
        #--plot the validation errors-----
        plt.xlabel('parameter sets')
        plt.ylabel('mae')
        plt.plot(np.arange(1,self.numGridTunings+1), errorArray[:self.numGridTunings, 1],'b',label='grid tuning')
        plt.plot(np.arange(self.numGridTunings + 1, errorArray.shape[0]+1), errorArray[self.numGridTunings:, 1], 'r', label='bayesian tuning')
        plt.grid(True)
        plt.legend(fontsize='x-small')
        plt.savefig(self.figDir+'/'+self.figLabel+'_tuning.png')

        print('-----------------validation accuracy--------------------------')
        print('validation accuracy index = ', validationAccuracyIndex)
        print('validation MAE = ', validationMAE)

        #        -------------------printing the results--------------------------------
        for key, value in sorted(self.validLog.items(), key=lambda item: item[1]['validation error(mae)']):
            self.selectedModel = key
            break
        print('-----------------selected model--------------------------')
        print('selected model : ' + str(self.architecture), self.selectedModel)
        print('selected model train error(mae) :', self.validLog[self.selectedModel]['training error(mae)'])
        print('selected model ' + self.validationMode + ' valid error(mae) :',
              self.validLog[self.selectedModel]['validation error(mae)'])
        print('selected model test error(mae) :', self.validLog[self.selectedModel]['test error(mae)'])
        print('selected model PktoPk phase error(ns) :', self.validLog[self.selectedModel]['PktoPk phase error(ns)'])
        print('----------------------------------------------------------')
        # --------------------------logging to file--------------------------------------------------------------

        self.validLog['selectedModel'] = {'model': key, 'results': self.validLog[key]}
        self.validLog['VAI'] = validationAccuracyIndex
        self.validLog['elapsed cpu time'] = time.process_time()-self.startTime
        self.validLog['total epochs'] = self.sumEpoch
        with open(self.logDir +'/log.json', 'a') as jsonLog:
            json.dump(self.validLog, jsonLog, indent=4)

        self.trainHistory = self.trainHistory[self.selectedModel]
        self.modelLog = self.modelLog[self.selectedModel]
        self.normMean = self.normMeanDict[self.selectedModel]
        self.normScale = self.normScaleDict[self.selectedModel]
        #----log trained weights for next training
        if self.fixWeightShape:
            with open(self.resultDir + '/preTrainedModel.pickle', 'wb') as pickleLog:
                pickle.dump(self.trainHistory, pickleLog)
        #---log training params for inference
        modelLog = {key: self.__dict__[key] for key in
                    ('modelLog', 'trainHistory', 'normScale', 'normMean','tempNormMean', 'tempNormScale',
                     'splitNormMean', 'splitNormScale', 'trend', 'decimationFilterTaps','iirNum','iirDen')}
        with open(self.resultDir +'/model.pickle', 'wb') as pickleLog:
            pickle.dump(modelLog, pickleLog)
        #-------writing prediction to output file------
        self.prediction = self.prediction[self.selectedModel]
        self.actualOutput = self.actualOutput[self.selectedModel]
        np.savetxt(self.outputFile, np.concatenate((np.expand_dims(self.testTimeStamps, axis=1), self.actualOutput[2],
        self.prediction[2]), axis = 1), delimiter =',', header = 'time stamp(s), actual output(ppb), prediction(ppb)')


    def model_train(self, trainData, validData):
        """
        trains the holdover model for one set of hyperparameters

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
        if self.initMethod == 'pretrain':
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
        trainHistory['validPred'] = validPredonEpoch
        trainHistory['trainedParameters'] = trainedParameters

        return trainHistory

    def model_cross_validate(self):
        """
         Implements N-fold cross-validation

         Returns:
         trainHistory -- python dictionary containing trained model parameters and training history(loss,..)
         crossValidError -- cross validation error
         validPrediction -- cross validation predictions concatenated together
         validOutput -- actual output for validation data concatenated together
         """

        nTrain = self.trainData[0].shape[0]
        lenCrossValid = int(nTrain / self.NUM_FOLD)
        trainIndices = np.arange(nTrain)
        folds = [np.arange(i * lenCrossValid, (i + 1) * lenCrossValid) for i in range(int(nTrain / lenCrossValid))]
        validPrediction = np.empty((lenCrossValid * self.NUM_FOLD, 1))
        validOutput = np.empty((lenCrossValid * self.NUM_FOLD, 1))
        crossValidErrors = []
        biasCrossValid = 0
        if self.crossValidMode == 'inc': #--train on whole data and then predict for folds in closed-loop mode
            trainHistory = self.model_train(self.laggedTrainData, self.laggedValidData)
        for fold in folds:
            inputTrainData = self.trainData[0][np.setdiff1d(trainIndices, fold), ...]
            outputTrainData = self.trainData[1][np.setdiff1d(trainIndices, fold), :]
            trainDataCrossValid = (inputTrainData, outputTrainData)
            inputValidData = self.trainData[0][fold, ...]
            outputValidData = self.trainData[1][fold, :]
            timeStampsValidData = self.trainData[2][fold]
            validDataCrossValid = (inputValidData, outputValidData, timeStampsValidData)
            if self.crossValidMode == 'exc': #--conventional n-fold cross-validation. not used in our product
                trainHistory = self.model_train(trainDataCrossValid, validDataCrossValid)
            foldError, foldPrediction, foldOutput, _ = evaluate(self, validDataCrossValid,
                                                                    trainHistory['trainedParameters'])
            validPrediction[fold] = foldPrediction.T + biasCrossValid
            validOutput[fold] = foldOutput.T + biasCrossValid
            crossValidErrors.append(foldError)
        crossValidError = np.mean(np.array(crossValidErrors))
        if self.crossValidMode == 'exc':
            trainHistory = self.model_train(self.trainData, self.validData)

        return trainHistory, crossValidError, validPrediction, validOutput

    def data_pre_processing(self):
        """
        Reads raw data from file, do feature engineering(diff  or  detrending) and prepares training, validation and test sets

        """




        self.currentHourSamples = int(3600 / self.currentSamplingTime)  # one hour in samples
        self.currentTestSamples = int(self.testLength * self.currentHourSamples)
        self.downsampleRatio = int(self.targetSamplingTime / self.currentSamplingTime)
        self.targetHourSamples = int(3600 / self.targetSamplingTime)
        self.targetTestSamples = int(self.testLength * self.targetHourSamples)
        start = int(self.captureStart * self.currentHourSamples)
        length = int(self.captureLength * self.currentHourSamples)
        readDataWhole = np.genfromtxt(self.inputFile, delimiter=',', skip_header=1)
        readDataWindow = readDataWhole[~np.isnan(readDataWhole).any(axis=1)][-length:,:]

        # ---fixing the sampling times---
        actualTimeStamps = readDataWindow[:, 0]
        desiredTimeStamps = np.arange(actualTimeStamps[-1], actualTimeStamps[0], -self.currentSamplingTime)[::-1]
        tempSampleFixed = np.interp(desiredTimeStamps, actualTimeStamps, readDataWindow[:, 1])
        splitSampleFixed = np.interp(desiredTimeStamps, actualTimeStamps, readDataWindow[:, 2])
        freqSampleFixed = np.interp(desiredTimeStamps, actualTimeStamps, readDataWindow[:, 3])
        f = interpolate.interp1d(actualTimeStamps, readDataWindow[:, 4], kind='previous')
        outagePartition = f(desiredTimeStamps)
        if self.inputData == 'temp':
            readData = np.array([desiredTimeStamps, tempSampleFixed, freqSampleFixed]).T
        elif self.inputData == 'split':
            readData = np.array([desiredTimeStamps, splitSampleFixed, freqSampleFixed]).T
        elif self.inputData == 'split+temp':
            readData = np.array([desiredTimeStamps, tempSampleFixed, freqSampleFixed, splitSampleFixed]).T
        elif self.inputData == 'split+split':
            tempSampleFixed = splitSampleFixed  + np.random.randn(len(splitSampleFixed )) * np.std(splitSampleFixed) * self.splitNoiseStd
            readData = np.array([desiredTimeStamps, tempSampleFixed, freqSampleFixed, splitSampleFixed]).T
        # ---------------filter parameter setting----------------------
        self.decimationFilterTaps = signal.firwin(self.decimationFirFilterOrder, 1. / self.downsampleRatio,
                                                  window='hamming')

        Fs = 1 / self.currentSamplingTime
        Wn = self.postFilterBandwidth / (Fs / 2)
        self.iirNum, self.iirDen = signal.bessel(self.iirPostFilterOrder, Wn, btype='lowpass', analog=False)

        # ----perform iir post filtering
        if self.postFilter:
            zi = signal.lfilter_zi(self.iirNum, self.iirDen)
            filteredData, _ = signal.lfilter(self.iirNum, self.iirDen, readData, axis=0,
                                             zi=np.dot(np.expand_dims(zi, axis=1), readData[0:1, :]))
            self.readData = filteredData
        else:
            self.readData = readData
        #--keep unfiltered data to compare with predictions
        self.rawFreq = np.convolve(readData[:, 2], self.decimationFilterTaps, 'valid')[::-1][::self.downsampleRatio][::-1]

        self.NUM_FOLD = int((self.readData.shape[0] / self.currentHourSamples - (
                    self.testLength + self.validLength)) / self.testLength)
        # --------low-pass filtering and decimation----------------



        data1 = np.convolve(self.readData[:, 1], self.decimationFilterTaps,'valid')[::-1][::self.downsampleRatio][::-1]
        data2 = np.convolve(self.readData[:, 2], self.decimationFilterTaps, 'valid')[::-1][::self.downsampleRatio][::-1]
        data = np.vstack((data1, data2)).T
        if self.inputData in ['split+temp','split+split']:
            data3 = np.convolve(self.readData[:, 3], self.decimationFilterTaps, 'valid')[::-1][::self.downsampleRatio][::-1]
            data = np.vstack((data1, data2, data3)).T

        numData = data1.size
        self.timeStamps = self.readData[:, 0][self.decimationFirFilterOrder-1:][::-1][::self.downsampleRatio][::-1]
        self.outagePartition = outagePartition[self.decimationFirFilterOrder-1:][::-1][::self.downsampleRatio][::-1]
        self.rawData = np.copy(data)
        self.plot_data()

        # ------------linear de-trending----------------------------

        if self.detrending:
            detrendDatas = np.zeros((numData, len(self.detrendLength)))
            self.linearRegPredicts = np.zeros((numData, len(self.detrendLength)))
            for counter,detrendLength in enumerate(self.detrendLength):
                detrendData, linearRegPredict , detrendError, lastTrend = self.piecewise_detrend(data[:,1:2], detrendLength)
                detrendDatas[:,counter] = detrendData
                self.linearRegPredicts[:,counter] = np.squeeze(linearRegPredict)
                if counter == 0:
                    minDetrendError = detrendError
                if detrendError <= minDetrendError :
                    bestDetrendData = detrendData
                    self.linearRegPredict = linearRegPredict
                    self.reg = lastTrend
                    minDetrendError = detrendError
                    trendIndice = counter

            print(f'best fit detrend length is {self.detrendLength[trendIndice]} hours')

            self.trendCoef = self.reg.coef_
            self.trendIntercept = self.reg.intercept_
            self.trend = self.reg
            if self.plot:
                plt.figure()
                plt.title('best fit training trend')
                plt.plot(self.timeStamps/3600, self.rawData[:,1], label='data')
                plt.plot(self.timeStamps/3600, self.linearRegPredict, label='linear trend')
                plt.xlabel('time(hours)')
                plt.ylabel('ppb')
                plt.grid(True)
                plt.savefig(self.figDir + '/trendData.png')  #
            # -------Data featureType---------
        if 'diff' in self.featureType:
            self.freqBias = np.copy(self.rawData[:, 1])
            self.tempBias = np.copy(self.rawData[:, 0])
            self.freqDrifts = np.diff(data[:, 1])
            self.temperature = np.diff(data[:, 0])
            self.timeStamps = np.delete(self.timeStamps, 0)
            if self.inputData in ['split+temp','split+split']:
                self.splitBias = np.copy(self.rawData[:, 2])
                self.split = np.diff(data[:, 2])
        else:
            self.freqDrifts = np.copy(detrendDatas)
            self.temperature = np.copy(data[:, 0])
            if self.inputData in ['split+temp','split+split']:
                self.split = np.copy(data[:, 2])

        # -----normalizing data---------------
        self.normMean, self.normScale = [0, 1]
        self.splitNormMean, self.splitNormScale = [0,1]
        self.tempNormMean, self.tempNormScale = [0, 1]
        if self.normalization == 'standard':
            self.normMeans = np.mean(self.freqDrifts[:-(self.targetTestSamples+self.N_MARGIN)], axis = 0)
            self.normScales = np.std(self.freqDrifts[:-(self.targetTestSamples + self.N_MARGIN)], axis =0)
            self.tempNormMean = np.mean(self.temperature[:-(self.targetTestSamples + self.N_MARGIN)])
            self.tempNormScale = np.std(self.temperature[:-(self.targetTestSamples + self.N_MARGIN)])
            if self.inputData in ['split+temp', 'split+split']:
                self.splitNormMean = np.mean(self.split[:-(self.targetTestSamples + self.N_MARGIN)])
                self.splitNormScale = np.std(self.split[:-(self.targetTestSamples + self.N_MARGIN)])
        elif self.normalization == 'norm':
            self.normMeans = np.amin(self.freqDrifts[:-(self.targetTestSamples + self.N_MARGIN)], axis=0)
            self.normScales = np.amax(self.freqDrifts[:-(self.targetTestSamples + self.N_MARGIN)], axis=0) - self.normMeans
            self.tempNormMean = np.amin(self.temperature[:-(self.targetTestSamples + self.N_MARGIN)])
            self.tempNormScale = np.amax(self.temperature[:-(self.targetTestSamples + self.N_MARGIN)]) - self.tempNormMean
            if self.inputData in ['split+temp', 'split+split']:
                self.splitNormMean = np.amin(self.split[:-(self.targetTestSamples + self.N_MARGIN)])
                self.splitNormScale = np.amax(self.split[:-(self.targetTestSamples + self.N_MARGIN)]) - self.splitNormMean

        self.freqDrifts= self.freqDrifts - self.normMeans
        self.freqDrifts = self.freqDrifts / self.normScales
        self.temperature = self.temperature - self.tempNormMean
        self.temperature = self.temperature / self.tempNormScale
        if self.inputData in ['split+temp','split+split']:
            self.split = self.split - self.splitNormMean
            self.split = self.split / self.splitNormScale

    def plot_data(self):
        """
        plot raw data plots
        """
        if self.plot:

            if self.inputData not in ['split+temp','split+split']:
                fig, ax1 = plt.subplots(figsize=(15, 7))
                color = 'tab:blue'
                ax1.set_xlabel('time(hours)')
                ax1.set_ylabel('ppb', color=color)
                ax1.plot(self.timeStamps / 3600, (self.rawData[:,1]),
                         label='raw data', color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.legend(loc=2)
                ax1.set_title(self.oscillator)
                ax1.grid(True)
                ax2 = ax1.twinx()
                color = 'tab:orange'
                ax2.set_ylabel(self.inputData+' value', color=color)
                ax2.plot(self.timeStamps / 3600, (data1),
                         label=self.inputData, color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.legend(loc=1)
                fig.tight_layout()
            else:
                plt.figure(figsize=(15, 7))
                ax1 = plt.subplot(211)
                ax2 = plt.subplot(212)
                ax22 = ax2.twinx()
                color = 'tab:blue'

                ax1.set_ylabel('ppb', color=color)
                ax1.plot(self.timeStamps / 3600, (self.rawData[:,1]),
                         label='raw data', color=color)
                ax1.plot(self.timeStamps[self.outagePartition !=0 ] / 3600, (self.rawData[self.outagePartition !=0 , 1]),'r.',
                         label='outage compensated', alpha = 0.05)
                ax1.legend()
                ax1.set_title(self.oscillator)
                ax1.grid(True)

                color = 'tab:orange'
                ax2.set_ylabel('temp value', color=color)
                ax2.set_xlabel('time(hours)')
                ax2.plot(self.timeStamps / 3600, (self.rawData[:,0]),
                         label=self.inputLabels[1], color=color)
                ax2.plot(self.timeStamps[self.outagePartition != 0] / 3600,
                         (self.rawData[self.outagePartition != 0, 0]), 'r.', alpha=0.05)
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.legend(loc=2)
                ax2.grid(True)
                color = 'tab:green'
                ax22.set_ylabel('split value', color=color)
                ax22.plot(self.timeStamps / 3600, (self.rawData[:,2]),
                         label=self.inputLabels[0], color=color)
                ax22.plot(self.timeStamps[self.outagePartition != 0] / 3600,
                         (self.rawData[self.outagePartition != 0, 2]), 'r.', alpha=0.05)
                ax22.tick_params(axis='y', labelcolor=color)
                ax22.legend(loc=1)
            plt.savefig(self.figDir+'/rawData_'+self.inputData+'.png') #raw data saved in fig folder
            plt.close('all')

    def piecewise_detrend(self, data, detrendLength):
        """
        perform partial deternding on freq data
        :param data: (nd.array) data to be detrended
        :param detrendLength: (float)  detrending length
        :return:
        copyData[:,0] :(nd.array) detrended data,
        lineraRegPredict : (nd.array) detrend lines,
        error: (float) mse error of regression,
        lastReg: (Linear Regression obj) last part trend object
        """
        copyData = np.copy(data)
        numData = copyData.shape[0]
        lastIndice= self.targetTestSamples+self.N_MARGIN
        if self.partialDetrending:
            numDetrendFolds = int((self.timeStamps[-lastIndice]-self.timeStamps[0])/(detrendLength*3600))
            if numDetrendFolds ==0:
                numDetrendFolds=1
        else:
            numDetrendFolds = 1
        linearRegPredict=[]
        for f in range(numDetrendFolds):
            lowIndice, highIndice = int(f*detrendLength*3600/self.targetSamplingTime), int((f+1)*detrendLength*3600/self.targetSamplingTime)
            if f == numDetrendFolds-1: # first part starts from beginning of data
                highIndice = numData - lastIndice

            reg = LinearRegression().fit(self.timeStamps[-highIndice-lastIndice:-lowIndice-lastIndice].reshape((highIndice-lowIndice, 1)), copyData[-highIndice-lastIndice:-lowIndice-lastIndice])
            if f == 0: #last detrend also predicts trend of test and conf windows
                lastReg = reg
                partialPredict = reg.predict(self.timeStamps[-highIndice - lastIndice:].reshape((lastIndice+highIndice, 1)))
                copyData[-highIndice - lastIndice:] = copyData[-highIndice - lastIndice:] - partialPredict
            else:
                partialPredict = reg.predict(self.timeStamps[-highIndice-lastIndice:-lowIndice-lastIndice].reshape((highIndice-lowIndice, 1)))
                copyData[-highIndice - lastIndice:-lowIndice - lastIndice] = copyData[-highIndice - lastIndice:-lowIndice - lastIndice] - partialPredict
            linearRegPredict[:0] = partialPredict
        error = np.sum(copyData[:-lastIndice] **2)
        linearRegPredict = np.array(linearRegPredict)
        return copyData[:,0], linearRegPredict, error, lastReg

    def data_windowing(self):
        """
        performs feature selection and windowing on preprocessed data according to number of temperature/split and feedback inputs


        """
        # select input freq with selected detrend length or for diff model in this run
        if 'diff' in self.featureType:
            self.freqDrift = self.freqDrifts
            self.normMean = self.normMeans
            self.normScale = self.normScales
        else:
            self.freqDrift = self.freqDrifts[:, self.detrendIndice]
            self.linearRegPredict = self.linearRegPredicts[:, self.detrendIndice]
            self.normMean = self.normMeans[self.detrendIndice]
            self.normScale = self.normScales[self.detrendIndice]

        N_FEEDBACK = int(self.nFeedbacks)
        N_TEMP = int(self.nTemps)
        if self.inputData in ['split+temp','split+split']:
            N_SPLIT = N_TEMP

        if self.inputData not in ['split+temp','split+split']:
            self.rnnFeatures = 2
        else:
            self.rnnFeatures = 3

        if N_FEEDBACK == 0 or N_TEMP == 0:
            self.rnnFeatures = self.rnnFeatures-1

        lag = np.maximum(N_FEEDBACK, N_TEMP)

        if self.featureSelection is None:
            self.historyDepth = lag

        N_DATA = self.freqDrift.shape[0]
        N_TOTAL = N_DATA - self.historyDepth
        N_VALID = int(self.validLength * self.targetHourSamples)
        N_TEST = self.targetTestSamples
        N_TRAIN = N_TOTAL - (N_TEST + N_VALID) - 2 * self.N_MARGIN

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
        if self.inputData not in ['split+temp','split+split']:
            self.featureLags = np.hstack((np.arange(N_TEMP, dtype=np.int64), N_TEMP + np.sort(featureLags)))
        else:
            self.featureLags = np.hstack((np.arange(N_TEMP, dtype=np.int64),np.arange(N_SPLIT, dtype=np.int64) + N_TEMP, N_TEMP + N_SPLIT + np.sort(featureLags)))
        # -------------------windowing training, validation and test data-------------------------
        if self.inputData not in ['split+temp','split+split']:
            windowedDataInput = np.expand_dims(
                np.hstack((self.temperature[self.historyDepth - N_TEMP + 1:self.historyDepth + 1],
                           self.freqDrift[:self.historyDepth + 1])),
                axis=0)
        else:
            windowedDataInput = np.expand_dims(
                np.hstack((self.temperature[self.historyDepth - N_TEMP + 1:self.historyDepth + 1],self.split[self.historyDepth - N_SPLIT + 1:self.historyDepth + 1],
                           self.freqDrift[:self.historyDepth + 1])),
                axis=0)
        for i in range(1, N_TOTAL):
            if self.inputData not in ['split+temp','split+split']:
                windowedData = np.expand_dims(
                    np.hstack((self.temperature[self.historyDepth + i - N_TEMP + 1:self.historyDepth + i + 1],
                               self.freqDrift[i:i + self.historyDepth + 1])), axis=0)
            else:
                windowedData = np.expand_dims(
                    np.hstack((self.temperature[self.historyDepth + i - N_TEMP + 1:self.historyDepth + i + 1],self.split[self.historyDepth + i - N_SPLIT + 1:self.historyDepth + i + 1],
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
        #---preparing data for confidence calculation-------
        confidenceWinLen = int(self.N_MARGIN / self. nConfidenceWin)
        self.confidenceData = [(windowedDataInput[-N_TEST:, :-1], np.expand_dims(
            windowedDataInput[- N_TEST:, -1], axis=1), self.timeStamps[- N_TEST:])]
        for n in range(1, self.nConfidenceWin): #confidence wins start from test day and slide back confidenceWinLen samples each time
            self.confidenceData.append((windowedDataInput[-N_TEST-n*confidenceWinLen:-n*confidenceWinLen, :-1],
            np.expand_dims(windowedDataInput[-N_TEST-n*confidenceWinLen:-n*confidenceWinLen, -1], axis=1), self.timeStamps[-N_TEST-n*confidenceWinLen:-n*confidenceWinLen]))
        self.trainTimeStamps = self.timeStamps[self.historyDepth:self.historyDepth+N_TRAIN]
        self.validTimeStamps = self.timeStamps[-(N_TEST + N_VALID + self.N_MARGIN): -(N_TEST + self.N_MARGIN)]
        self.testTimeStamps  = self.timeStamps[- N_TEST:]

        validSeparation = int(inputValidData.shape[0] / 2)
        self.trainData = (inputTrainData, outputTrainData, self.trainTimeStamps)
        self.validData = (inputValidData, outputValidData, self.validTimeStamps)
        self.testData = (self.confidenceData[0][0], self.confidenceData[0][1], self.testTimeStamps)

        self.laggedTrainData = (inputTrainData[:, self.featureLags], outputTrainData, self.trainTimeStamps)
        self.laggedValidData = (inputValidData[:, self.featureLags], outputValidData, self.validTimeStamps)


    def valid_eval(self, validationErrors, predictionErrors):
        """
        Returns validation accuracy index and validation-test errors average MAE
        Arguments:
        validationErrors: (list) validation errors for all models
        predictionErrors: (list)test errors for all models
        Returns:
        accuracyIndex: (float) validation accuracy index
        validationMAE:(float)  weighted average of MAE between test and validation errors
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



    def bayesian_optimisation(self):
        """ bayesian_optimisation
        iteratively fits GP to existing points of validation loss as a function of hyper parameters and performs bayesian tuning on this function
         inputs:
        ----------
            self.nIterBayesian: (int)
                Number of iterations to run the search algorithm.
            paramPDFs: (dict) dictionary
                Distribution, Lower and upper bounds on the parameters of the function `sample_loss`.
            self.BayesianHyperparameters: (list) of normalized hyperparameters sets (will be converted to 2-D array of (number of tuning iterations, number of hyperParameters,))
            array-like, shape = [n_pre_samples, n_params].
                Array of initial points to sample the valid loss function for
            self.validLoss :(list) function to be optimized. list of validations (will be converted  to 1-D of shape (number of tuning iterations,))
            self.BayesianGpParams: (dict)dictionary.
                Dictionary of parameters to pass on to the underlying Gaussian Process.
           self.bayesianHyperparamPDFs : (dict)dictionary containing hyperparameters' PDF for bayesian tuning
           self.nIterBayesian : (int)number of bayessian tuning iterations
           self.paramScale : (float) mean of hyper parameters to map them to [-1,1]
           self.paramMean : (float) scale of hyper parameters to map them to [-1,1]
        outputs:
        self.BayesianNextHyperparameters : next set of hyperparameters chosen at the end of each iteration
        """

        print('bayesian tuning started...')
        randomSeed = 0

        # Create the GP
        if self.BayesianGpParams is not None:
            model = gp.GaussianProcessRegressor(**self.BayesianGpParams)
        else:
            kernel = gp.kernels.Matern()
            model = gp.GaussianProcessRegressor(kernel=kernel)

        for n in range(self.nIterBayesian):
            xp = np.array(self.BayesianHyperparameters)
            yp = np.array(self.validLoss)

            model.fit(xp, yp)

            # Sample next hyperparameter

            nextHyperparameters = sample_next_hyperparameter(expected_improvement, model, yp,
                                                                          greater_is_better=False,
                                                                          paramPDFs=self.bayesianHyperparamPDFs,
                                                                          n_restarts=100, randomSeed=randomSeed,BayesianParam = self.BayesianParam)

            if np.any(np.linalg.norm((nextHyperparameters - xp), axis=1) <= 1e-7):
                print('too close hyperparameters ...randomly chosen')
                nextHyperparameters = np.random.uniform(-1, 1, xp.shape[1])
            #-------------de-normalizing selected hyperparameters---------
            self.BayesianNextHyperparameters = []
            for counter, key in enumerate(list(self.bayesianHyperparamPDFs.keys())):
                if self.bayesianHyperparamPDFs[key][0] == 'log-uniform':
                    self.BayesianNextHyperparameters.append(
                        10 ** (nextHyperparameters[counter] * self.paramScale[counter] + self.paramMean[counter]))
                elif self.bayesianHyperparamPDFs[key][0] == 'integer-uniform':
                    nextIntegerHyperparameter = int(np.floor(nextHyperparameters[counter] * self.paramScale[counter] + self.paramMean[counter]))
                    if nextIntegerHyperparameter == self.bayesianHyperparamPDFs[key][2]:
                        nextIntegerHyperparameter = nextIntegerHyperparameter - 1
                    self.BayesianNextHyperparameters.append(nextIntegerHyperparameter)
            randomSeed = randomSeed + 1
            # do training and compute valid loss for new set of parameters
            self.model_tune('Bayesian')


    def run(self):

        """
        runner function of model. runs training in order of pre_processing,grid tuning, bayesian tuning, and logging
        """

        self.data_pre_processing()
        self.model_tune('grid')
        self.bayesian_optimisation()
        self.model_log()


def predict(self, inputData, trainedParameters):
    """
    Implements the open loop and closed loop prediction for a model with trainedParameters and inputData

    Arguments:
    self: holdover_train_model or confidence_model obj
    InputData -- (nd.array) input dataset to the model
    trainedParameters -- (dict) trained model parameters

    Returns:
    closedLoopPrediction --(nd.array)closed loop (autoregressive) prediction for the given dataset InputData(only first sample is used)
    openLoopPrediction -- (nd.array) open loop prediction for the given dataset InputData
    """
    nInputs = inputData.shape[1]
    numData = inputData.shape[0]
    N_FEEDBACK = int(self.nFeedbacks)
    N_TEMP = int(self.nTemps)
    N_SPLIT = 0
    if self.inputData in ['split+temp', 'split+split']:
        N_SPLIT = N_TEMP
    #calculate open-loop predictions
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
    # if there is no feedback return openloop predictions
    if nInputs == N_TEMP + N_SPLIT:
        return openLoopPrediction.T, openLoopPrediction.T
    closedLoopInput = np.copy(inputData[0:1, :])
    tempInput = inputData[:, :N_TEMP + N_SPLIT]
    closedLoopPrediction = np.zeros((1, numData))
    closedLoopFeedbackInput = closedLoopInput[:, N_TEMP + N_SPLIT:]
    # calculate cloosed-loop prediction for each output sample
    #it's based predicting one sample, shifting the input one sample to left, and replacing last sample by new prediction
    for r in range(numData):
        closedLoopInput[0, :N_TEMP + N_SPLIT] = tempInput[r, :]

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
        closedLoopInput[:, N_TEMP + N_SPLIT:] = closedLoopFeedbackInput

    return closedLoopPrediction, openLoopPrediction.T

def evaluate(self, data, trainedParameters, predMode='prediction'):
        """
         Calculates rmse error between actual outputs and model predictions
         Arguments:
         self: holdover_train_model or confidence_model obj
         data: (tuple) of (inputData, outputData, dataTime)
         trainableParameters: (dict)trained model parameters
         predMode:(str) 'prediction' for closed-loop prediction and 'estimation' for open-loop prediction
         Returns:
         error: (float)rmse error between prediction and output
         prediction: (nd.array)model prediction for inputData
         dataOutput: (nd.array) actual output (processed back in case of diff or normalization )
        """

        dataPrediction, dataEstimation = predict(self, data[0], trainedParameters)
        dataOutput = data[1].T
        if self.inputData not in  ['split+temp','split+split']:
            dataInput = np.expand_dims(data[0][:, self.nTemps - 1] , axis=1)
        else:
            dataInput = np.vstack((data[0][:, self.nTemps - 1],data[0][:, self.nTemps + self.nSplits - 1])).T
        if self.normalization: # de-normalize output and prediction
            dataPrediction = dataPrediction * self.normScale + self.normMean
            dataEstimation = dataEstimation * self.normScale + self.normMean
            dataOutput = dataOutput * self.normScale + self.normMean
            if self.inputData not in  ['split+temp','split+split']:
                dataInput =  dataInput * self.tempNormScale + self.tempNormMean
            else:
                dataInput[:,0] = dataInput[:,0] * self.tempNormScale + self.tempNormMean
                dataInput[:,1] = dataInput[:,1] * self.splitNormScale + self.splitNormMean
        #---revert feature engineering(detrend or diff)
        if 'diff' in self.featureType:
            biasIndice = np.argwhere(self.timeStamps == data[2][0])
            dataPrediction = np.cumsum(dataPrediction, axis=1) + self.freqBias[biasIndice]
            dataEstimation = np.cumsum(dataEstimation, axis=1) + self.freqBias[biasIndice]
            dataOutput = np.cumsum(dataOutput, axis=1) + self.freqBias[biasIndice]
            if self.inputData not in  ['split+temp','split+split']:
                dataInput = np.cumsum(dataInput, axis=0) + self.tempBias[biasIndice]
            else:
                dataInput = np.cumsum(dataInput, axis=0) + [self.tempBias[biasIndice][0][0],self.splitBias[biasIndice][0][0]]
        if self.detrending:
            lowTrendIndice = np.argwhere(self.timeStamps == data[2][0])[0][0]
            highTrendIndice = np.argwhere(self.timeStamps == data[2][-1])[0][0] + 1
            dataPrediction = dataPrediction + self.linearRegPredict[lowTrendIndice:highTrendIndice].T
            dataEstimation = dataEstimation + self.linearRegPredict[lowTrendIndice:highTrendIndice].T
            dataOutput = dataOutput + self.linearRegPredict[lowTrendIndice:highTrendIndice].T

        #---compute rmse errors
        maePrediction = (metrics.mean_absolute_error(dataOutput, dataPrediction))
        maeEstimation = (metrics.mean_absolute_error(dataOutput, dataEstimation))

        if predMode == 'prediction':
            error = maePrediction
            prediction = dataPrediction
        elif predMode == 'estimation':
            error = maeEstimation
            prediction = dataEstimation

        return error, prediction, dataOutput, dataInput

def phase_diff(actual, prediction, samplingTime, time = []):
    """
    Implements phase error calculation between prediction and output actual arrays
    Arguments:
    actual -- (nd.array) actual output
    prediction -- (nd.array)predicted output
    samplingTime -- (float) sampling time
    time -- (nd.array)time stamps of the output
    Returns:
    phaseDiff:(list) prediction phase error vs time
    phaseDiffUncomp:(list) uncomp phase error vs time
    """

    freqDiff = np.squeeze(actual - prediction)
    if time == []:
        time = np.arange(freqDiff.size) * samplingTime
    if actual.ndim > 1:
        actual = np.squeeze(actual)
    freqDiffUncomp = actual -actual[0]
    phaseDiff, phaseDiffUncomp = [], []
    #--for each sample of output and prediction, compute the phase error
    for n in range(1, freqDiff.size):
        phaseDiff.append(np.trapz(freqDiff[:n + 1], time[:n + 1], axis=0))
        phaseDiffUncomp.append(np.trapz(freqDiffUncomp[:n + 1], time[:n + 1], axis=0))

    return phaseDiff, phaseDiffUncomp






