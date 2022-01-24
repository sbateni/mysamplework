/* __dev_AiHoTrainDriveReentrant */
/**
The most top function in training which drives a single training session for one training config set.
Fuctions is reentrant in the sense that it supports  pausing, resuming and stopping.
\param[in]  config  pointer to training config structure.
\param[in] inputFileS pointer to input file structure.
\param[in] outageData pointer to outage data matrix. If not NULL it will be inserted in data read from inputFileS.
\param[out] outputFileS pointer to training output structure.
\param[in/out] trainContext pointer to training reentrant context structure containing states and temp data.
\retval  __DEV_AIHO_OK          Success.
\retval __DEV_AIHO_ERR_INVALID_POINTER    Pointer not created/NULL

*******************************************************************************/
AIHOE __dev_AiHoTrainDriveReentrant(const __dev_AiHoTrainConfigS* config, __dev_AiHoInputFileS* inputFileS, __dev_AiHoTrainFileS* outputFileS, __dev_AiHoMatrixS* outageData,
                                     __dev_AiHoTrainReentContextS* trainContext)   
{ 
    __dev_AiHoTrainDriveReentDataS* pRun = NULL;
    __dev_AiHoModelTuneReentDataS* pRunNext = NULL;
    __dev_BooleanE externalStop = __DEV_FALSE;
    AIHOE_NEW();
    AIHOE_GATE_PTR(trainContext); 
    if (AIHOE_OK())
    {         
        pRun = &trainContext->trainDriveRunData;
        pRunNext = &trainContext->modelTuneRunData;
        externalStop = (pRun->state == __DEV_AIHO_TRAIN_STATE_STOP) ? __DEV_TRUE : __DEV_FALSE;       
    }
    if (AIHOE_OK() && pRun->state == __DEV_AIHO_TRAIN_STATE_START)
    {        
        /* checking only in START state for time optimization*/
        AIHOE_GATE_PTR(inputFileS);
        AIHOE_GATE(__dev_AiHoCheckTrainFileS(outputFileS));
        AIHOE_GATE(__dev_AiHoCheckTrainConfigS(config));
 
        if (AIHOE_OK())
        {     
            OS_MEMSET(&pRun->input, 0, sizeof(__dev_AiHoMatrixS));   
            OS_MEMSET(&pRun->inputS, 0, sizeof(__dev_AiHoInputS)); 
            OS_MEMSET(&pRun->trainInputS, 0, sizeof(__dev_AiHoTrainInputS)); 
            OS_MEMSET(&pRun->preprocessConfig, 0, sizeof(__dev_AiHoConfigS)); 
            OS_MEMSET(&pRun->model, 0, sizeof( __dev_AiHoTrainingModelS)); 
            inputFileS->numRows = (Uint32T)(config->captureLength * __DEV_AIHO_SECONDS_IN_HOUR / config->currentSamplingTime);
        }   
        AIHOE_GATE(__dev_AiHoMatrixCreate(&pRun->input, inputFileS->numCols, inputFileS->numRows)); 
        AIHOE_GATE(__dev_AiHoReadInputCsvFile(inputFileS, &pRun->input));
        
        if (AIHOE_OK())
        {        
            pRun->inputS.length = inputFileS->numRows;
            pRun->inputS.stackLength = pRun->inputS.length;
            
            AIHOE_REPLACE(__dev_AiHoCreateInputS(&pRun->inputS));        
        }  
            
        AIHOE_GATE(__dev_AiHoUpdateInputStruct(inputFileS, &pRun->input, &pRun->inputS, outageData));
        AIHOE_FINALLY(__dev_AiHoMatrixDestroy(&pRun->input));
        
        if (AIHOE_OK())
        {        
            memcpy(&pRun->preprocessConfig, config, (__DEV_AIHO_TRAINED_MODEL_NUM_CONFIG_PARAMS) * sizeof(float));       
            AIHOE_REPLACE(__dev_AiHoDataPreProcessing(&pRun->preprocessConfig, &config->iirFilterInitBuffer, &config->iirFilterCoeffs, config->decimationFilterTaps, &pRun->inputS)); 
        }
       
        if (AIHOE_OK())
        { 
            pRun->trainInputS.length = pRun->inputS.length;
            pRun->trainInputS.stackLength = pRun->inputS.stackLength;  
            pRun->trainInputS.featureType = config->featureType;
            pRun->trainInputS.numDetrendLengths = config->numDetrendLengths;
            if (config->featureType == __DEV_AIHO_FEATURE_TYPE_DIFF)
            {
                pRun->trainInputS.numDetrendLengths = 0;
            }
            AIHOE_REPLACE(__dev_AiHoCreateTrainInputS(&pRun->trainInputS));               
        }      

        AIHOE_GATE(__dev_AiHoCopyInputToTrainInputS(&pRun->inputS, &pRun->trainInputS));
        AIHOE_GATE(__dev_AiHoTrainPreProcessing(config, &pRun->trainInputS)); 
        
        if (AIHOE_OK())
        {
            __DEV_AIHO_TRACE_TRAINING_INFO("__dev_AiHoTrainDrive: START successfull...RUNNING now...",0,0,0,0,0,0); 
            pRun->state = __DEV_AIHO_TRAIN_STATE_RUNNING;
            pRunNext->state = __DEV_AIHO_TRAIN_STATE_START;
        }            
        
    }
    if (AIHOE_OK() && pRun->state == __DEV_AIHO_TRAIN_STATE_RUNNING)
    {        
        AIHOE_GATE(__dev_AiHoModelTuneReentrant(config, &pRun->trainInputS, &pRun->model, outputFileS->logFile, trainContext));
        if (AIHOE_OK() && pRunNext->state == __DEV_AIHO_TRAIN_STATE_DONE)
        {
            pRun->state = __DEV_AIHO_TRAIN_STATE_DONE;                                  
        }
    }
    if (AIHOE_OK() && pRun->state == __DEV_AIHO_TRAIN_STATE_STOP)
    {       
        __DEV_AIHO_TRACE_TRAINING_INFO("__dev_AiHoTrainDrive: STOP...",0,0,0,0,0,0);
        /* one last run for cleaning up all nested functions*/
        AIHOE_GATE(__dev_AiHoModelTuneReentrant(config, &pRun->trainInputS, &pRun->model, outputFileS->logFile, trainContext));
        if (AIHOE_OK())
        { 
            pRun->state = __DEV_AIHO_TRAIN_STATE_DONE;
        }
    }
    if (AIHOE_OK() && pRun->state == __DEV_AIHO_TRAIN_STATE_DONE)
    {       
        
        __dev_AiHoTrainedModelS selectedModel; 
        OS_MEMSET(&selectedModel, 0, sizeof(selectedModel));
         /* select best model if at least one model is trained*/
        if (pRunNext->elapsedIters > 0)
        {
       
            AIHOE_GATE(__dev_AiHoModelValidSelect(pRun->model.crossValidations, pRun->model.trainHistorys, &selectedModel, pRunNext->elapsedIters)); 
             /* also add input norm factors to slected model*/
            if (AIHOE_OK())
            { 
                selectedModel.tempNormMean = pRun->trainInputS.tempNormMean;
                selectedModel.tempNormScale = pRun->trainInputS.tempNormScale;
                selectedModel.splitNormMean = pRun->trainInputS.splitNormMean;
                selectedModel.splitNormScale = pRun->trainInputS.splitNormScale;
            }
            AIHOE_GATE(__dev_AiHoModelLog(config, &selectedModel, &pRun->trainInputS, outputFileS)); 
        }       
      
        /*delete structures*/
        AIHOE_FINALLY(__dev_AiHoDestroyTrainingModelS(&pRun->model)); 
        AIHOE_FINALLY(__dev_AiHoDestroyTrainedParamS(&selectedModel.trainedParams)); 
        AIHOE_FINALLY(__dev_AiHoDestroyTrainInputS(&pRun->trainInputS)); 
        AIHOE_FINALLY(__dev_AiHoDestroyInputS(&pRun->inputS));

        if (AIHOE_OK())
        {
            if (externalStop)
            { 
                __DEV_AIHO_TRACE_TRAINING_INFO("__dev_AiHoTrainDrive: DONE...by external STOP",0,0,0,0,0,0); 
            }
            else
            {
                __DEV_AIHO_TRACE_TRAINING_INFO("__dev_AiHoTrainDrive: DONE...by finishing iters",0,0,0,0,0,0); 
            }             
            __DEV_AIHO_TRACE_TRAINING_DEBUG("__dev_AiHoTrainDrive: training done and logged successfully",0,0,0,0,0,0); 
        } 
    }        
    AIHOE_RETURN();    
    
}