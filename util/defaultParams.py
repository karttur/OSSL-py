'''
Created on 7 Aug 2023

@author: thomasgumbricht
'''

from os import path, makedirs

from util.jsonIO import DumpAnyJson

def CampaignParams():
    """ Default campaign parameters for all OSSL processing
    
        :returns: parameter dictionary
        
        :rtype: dict
    """
    
    campaignD = {'campaignId': 'OSSL-region-etc','campaignShortId':'OSSL-xyz'}
    
    campaignD['campaignType'] = 'laboratory'
    
    campaignD['theme'] = 'soil'
    
    campaignD['product'] = 'diffuse reflectance'
    
    campaignD['units'] = 'fraction'
        
    campaignD['geoRegion'] = "Sweden"
    
    return campaignD
    

def StandardParams(): 
    """ Default standard parameters for all OSSL processing
    
        :returns: parameter dictionary
        
        :rtype: dict
    """   
    
    paramD = {}
    
    paramD['verbose'] = 1
    
    paramD['id'] = "auto"
    
    paramD['name'] = "auto"
    
    paramD['userId'] = "youruserid - any for now"
    
    paramD['importVersion'] = "OSSL-202308"
    
    return paramD

def StandardXspectreParams(): 
    """ Default standard parameters for all OSSL processing
    
        :returns: parameter dictionary
        
        :rtype: dict
    """   
    
    paramD = {}
    
    paramD['verbose'] = 1
    
    paramD['id'] = "auto"
    
    paramD['name'] = "auto"
    
    paramD['userId'] = "youruserid - any for now"
    
    paramD['importVersion'] = "importxspectrev080"
    
    return paramD

def ImportParams():
    """ Default template parameters for importing OSSL csv data
    
        :returns: parameter dictionary
        
        :rtype: dict
    """
    
    paramD = StandardParams()
    
    paramD['campaign'] = CampaignParams()
    
    paramD['rootFP'] = '/path/to/folder/with/ossl/download'
    
    paramD['soilSample'] = {'minDepth':0,'maxDepth':20}
      
    paramD['visnir'] = {}
    
    paramD['visnir']['apply'] = True
    
    paramD['visnir']['subFP'] = 'visnir'
    
    paramD['visnir']['beginWaveLength'] = 460
    
    paramD['visnir']['endWaveLength'] = 1050
        
    paramD['visnir']['inputBandWidth'] = 2
           
    paramD['visnir']['outputBandWidth'] = 10
    
    paramD['mir'] = {}
    
    paramD['mir']['apply'] = True
    
    paramD['mir']['subFP'] = 'mir'
    
    paramD['mir']['beginWaveLength'] = 2500
    
    paramD['mir']['endWaveLength'] = 8000
        
    paramD['mir']['inputBandWidth'] = 2
           
    paramD['mir']['outputBandWidth'] = 10
    
    paramD['neon'] = {}
    
    paramD['neon']['apply'] = True
    
    paramD['neon']['subFP'] = 'neon'
    
    paramD['neon']['beginWaveLength'] = 1350
    
    paramD['neon']['endWaveLength'] = 2550
        
    paramD['neon']['inputBandWidth'] = 2
           
    paramD['neon']['outputBandWidth'] = 10
        
    ''' USDA oriented input data
    paramD['labData'] = ['c.tot_usda.a622_w.pct','ca.ext_usda.a1059_mg.kg',
                'caco3_usda.a54_w.pct','oc_usda.c729_w.pct', 'cec_usda.a723_cmolc.kg',
                'clay.tot_usda.a334_w.pct','fe.ext_usda.a1064_mg.kg']
    
    paramD['input']['labDataRange'] = {'oc_usda.calc_wpct':{'min':0,'max':10}}
    
    '''
    ''' LUCAS oriented input data'''
    paramD['labData'] = ['caco3_usda.a54_w.pct',
      'cec_usda.a723_cmolc.kg',
      'cf_usda.c236_w.pct',
      'clay.tot_usda.a334_w.pct',
      'ec_usda.a364_ds.m',
      'k.ext_usda.a725_cmolc.kg',
      'n.tot_usda.a623_w.pct',
      'oc_usda.c729_w.pct',
      'p.ext_usda.a274_mg.kg',
      'ph.cacl2_usda.a481_index',
      'ph.h2o_usda.a268_index',
      'sand.tot_usda.c60_w.pct',
      'silt.tot_usda.c62_w.pct']
    
    ''' LabDataRange - example'''
    
    paramD['labDataRange'] = {}
    
    paramD['labDataRange']['caco3_usda.a54_w.pct'] = {
        "min": 0,
        "max": 10}

    return (paramD)

def ImportXspectreParams():
    """ Default template parameters for importing OSSL csv data
    
        :returns: parameter dictionary
        
        :rtype: dict
    """
    
    paramD = StandardXspectreParams()
    
    paramD['campaign'] = CampaignParams()
    
    paramD['rootFP'] = '/path/to/folder/with/ossl/download'
    
    paramD['whiteReference'] = 'whiteRef.csv'
    
    paramD['whiteReferenceFactor'] = 1.0
        
    paramD['soilSample'] = {'minDepth':0,'maxDepth':100}
      
    paramD['xspectrolum'] = {}
    
    paramD['xspectrolum']['apply'] = True
    
    paramD['xspectrolum']['subFP'] = 'visnir'
    
    paramD['xspectrolum']['beginWaveLength'] = 460
    
    paramD['xspectrolum']['endWaveLength'] = 1050
                   
    paramD['xspectrolum']['outputBandWidth'] = 10
    
    paramD['mode'] = 'default'
    
    paramD['version'] = ''
    
    paramD['prepcode'] = ''
    
    paramD['scan'] = ''
    
    paramD['getlist'] = ''
    
    paramD['listPath'] = ''
    
    paramD['pattern'] = ''
            
    
    ''' LUCAS oriented input data
    paramD['labData'] = ['caco3_usda.a54_w.pct',
      'cec_usda.a723_cmolc.kg',
      'cf_usda.c236_w.pct',
      'clay.tot_usda.a334_w.pct',
      'ec_usda.a364_ds.m',
      'k.ext_usda.a725_cmolc.kg',
      'n.tot_usda.a623_w.pct',
      'oc_usda.c729_w.pct',
      'p.ext_usda.a274_mg.kg',
      'ph.cacl2_usda.a481_index',
      'ph.h2o_usda.a268_index',
      'sand.tot_usda.c60_w.pct',
      'silt.tot_usda.c62_w.pct']
    '''
    ''' LabDataRange - example
    
    paramD['labDataRange'] = {}
    
    paramD['labDataRange']['caco3_usda.a54_w.pct'] = {
        "min": 0,
        "max": 10}
    '''
    return (paramD)

def PlotParams():
    ''' Default parameters for plotting soil spectral library data
    
        :returns: parameter dictionary
        :rtype: dict
    '''
    
    
    paramD = StandardParams()
    
    paramD['campaign'] = CampaignParams()
    
    paramD['input'] = {}
    
    paramD['input']['jsonSpectraDataFilePath'] = 'path/to/jsonfile/with/spectraldata.json'
                   
    paramD['plot'] = {}
                
    paramD['plot']['colorRamp'] = "jet"
    
    paramD['plot']['maxSpectra'] = 100
    
    paramD['plot']['figSize'] = {'x':0,'y':0}
    
    paramD['plot']['legend'] = False
    
    paramD['plot']['tightLayout'] = False
    
    paramD['plot']['supTitle'] = "auto"
    
    paramD['plot']['screenDraw'] = True
    
    paramD['plot']['savePng'] = True

    paramD['plot']['xLim'] = {}
    
    paramD['plot']['xLim']['xMin'] = 0
    
    paramD['plot']['xLim']['xMax'] = 0
    
    paramD['plot']['raw'] = {}
    
    paramD['plot']['raw']['apply'] = True
    
    paramD['plot']['raw']['axisLabel'] = {}
    
    paramD['plot']['raw']['axisLabel']['x'] = "Wavelength (nm)"
    
    paramD['plot']['raw']['axisLabel']['y'] = "reflectance"
    
    paramD['plot']['raw']['yLim'] = {}
    
    paramD['plot']['raw']['yLim']['yMin'] = 0
    
    paramD['plot']['raw']['yLim']['yMax'] = 0
    
    paramD['plot']['raw']['title'] = {}
    
    paramD['plot']['raw']['title']['x'] = 0
    
    paramD['plot']['raw']['title']['y'] = 0
    
    paramD['plot']['raw']['title']['title'] = "Original spectra"
    
    paramD['plot']['raw']['text'] = {}
    
    paramD['plot']['raw']['text']['x'] = 0.02
    
    paramD['plot']['raw']['text']['y'] = 0.8
    
    paramD['plot']['raw']['text']['bandWidth'] = True
    
    paramD['plot']['raw']['text']['samples'] = True
    
    paramD['plot']['raw']['text']['skipStep'] = True
    
    paramD['plot']['raw']['text']['text'] = ""
    
    paramD['plot']['derivatives'] = {}
    
    paramD['plot']['derivatives']['apply'] = True
    
    paramD['plot']['derivatives']['axisLabel'] = {}
    
    paramD['plot']['derivatives']['axisLabel']['x'] = "Wavelength (nm)"
    
    paramD['plot']['derivatives']['axisLabel']['y'] = "Derivative"
    
    paramD['plot']['derivatives']['yLim'] = {}
    
    paramD['plot']['derivatives']['yLim']['yMin'] = 0
    
    paramD['plot']['derivatives']['yLim']['yMax'] = 0
    
    paramD['plot']['derivatives']['title'] = {}
    
    paramD['plot']['derivatives']['title']['x'] = 0
    
    paramD['plot']['derivatives']['title']['y'] = 0
    
    paramD['plot']['derivatives']['title']['title'] = "Derivatives"
    
    paramD['plot']['derivatives']['text'] = {}
    
    paramD['plot']['derivatives']['text']['x'] = 0.6
    
    paramD['plot']['derivatives']['text']['y'] = 0.8
    
    paramD['plot']['derivatives']['text']['bandWidth'] = True
    
    paramD['plot']['derivatives']['text']['samples'] = True
    
    paramD['plot']['derivatives']['text']['skipStep'] = True
    
    paramD['plot']['derivatives']['text']['text'] = ""
    
    paramD['plot']['scatter'] = {}
    
    paramD['plot']['scatter']['size'] = 50
            
    return (paramD)

def SoilLineExtractParams():
    ''' Default parameters for soilline extraction from soil spectral library data
    
        :returns: parameter dictionary
        :rtype: dict
    '''
    
    paramD = StandardParams()
    
    paramD['campaign'] = CampaignParams()
    
    paramD['input'] = {}
    
    paramD['input']['jsonSpectraDataFilePath'] = 'path/to/jsonfile/with/spectraldata.json'
        
    paramD['input']['yband'] = 1040
        
    paramD['model'] = {}

    paramD['model']['regressor'] = 'OLS'
    
    # lowerEdgePriority excluded in this version    
    #paramD['model']['lowerEdgePriority'] = False
    
    paramD['model']['distillStd'] = 1.0
       
    paramD['plot'] = {}
    
    paramD['plot']['minMaxSoilLines'] = True
    
    paramD['plot']['rawRegressionSoilLines'] = True
    
    paramD['plot']['distilledRegressionSoilLines'] = True
            
    paramD['plot']['colorRamp'] = "jet"
    
    paramD['plot']['maxScatterBands'] = 6
    
    paramD['plot']['figSize'] = {'x':0,'y':0}
    
    paramD['plot']['legend'] = False
    
    paramD['plot']['tightLayout'] = False
    
    paramD['plot']['scatter'] = {'size':50}
    
    paramD['plot']['endMembers'] = {}
    
    paramD['plot']['endMembers']['lightSoil'] = {'size':150,'color':'lightgrey'}
    
    paramD['plot']['endMembers']['darkSoil'] = {'size':150,'color':'black'}
        
    paramD['plot']['text'] = {'x':0.6,'y':0.2}
    
    paramD['plot']['text']['bandWidth'] = True
    
    paramD['plot']['text']['samples'] = True
    
    paramD['plot']['text']['text'] = ''
    
    paramD['figure'] = {} 
    
    paramD['figure']['minMaxSoilLines'] = True
    
    paramD['figure']['rawRegressionSoilLines'] = True
    
    paramD['figure']['distilledRegressionSoilLines'] = True
    
    return (paramD) 

def MLmodelParams():
    ''' Default parameters for soilline extraction from soil spectral library data
    
        :returns: parameter dictionary
        :rtype: dict
    '''
    
    paramD = StandardParams()
    
    paramD['campaign'] = CampaignParams()
    
    paramD['input'] = {}
    
    paramD['input']['jsonSpectraDataFilePath'] = 'path/to/jsonfile/with/spectraldata.json'
    
    paramD['input']['jsonSpectraParamsFilePath'] = 'path/to/jsonfile/with/spectralparams.json'
    
    paramD['input']['hyperParameterRandomTuning'] = 'path/to/jsonfile/with/hyperparam/tuning.json'
    
    ''' LUCAS oriented targetFeatures data'''
    paramD['targetFeatures'] = ['caco3_usda.a54_w.pct',
      'cec_usda.a723_cmolc.kg',
      'cf_usda.c236_w.pct',
      'clay.tot_usda.a334_w.pct',
      'ec_usda.a364_ds.m',
      'k.ext_usda.a725_cmolc.kg',
      'n.tot_usda.a623_w.pct',
      'oc_usda.c729_w.pct',
      'p.ext_usda.a274_mg.kg',
      'ph.cacl2_usda.a481_index',
      'ph.h2o_usda.a268_index',
      'sand.tot_usda.c60_w.pct',
      'silt.tot_usda.c62_w.pct']
    
    paramD['targetFeatureSymbols'] = {'caco3_usda.a54_w.pct':{'color': 'orange', 'size':50}}
    
    paramD['derivatives'] = {'apply':False, 'join':False}
    
    paramD['removeOutliers'] = {}
    
    paramD['removeOutliers']['comment'] = "removes sample outliers based on spectra only - globally applied as preprocess"
    
    paramD['removeOutliers']['apply'] = True
    
    paramD['removeOutliers']['detectorMethodList'] = ["iforest (isolationforest)",
                                                      "ee (eenvelope,ellipticenvelope)",
                                                      "lof (lofactor,localoutlierfactor)",
                                                      "1csvm (1c-svm, oneclasssvm)"]
    
    paramD['removeOutliers']['detector'] = "1csvm"
    
    paramD['removeOutliers']['contamination'] = 0.1
    
    paramD['manualFeatureSelection'] = {}
    
    paramD['manualFeatureSelection']['comment'] = "Manual feature selection overrides other selection alternatives"
   
    paramD['manualFeatureSelection'] ['apply'] = False
    
    paramD['manualFeatureSelection']['spectra'] = [ "A", "B", "C"],
    
    paramD['manualFeatureSelection']['derivatives'] = {}
    
    paramD['manualFeatureSelection']['derivatives']['firstWaveLength'] = ['A','D']
    
    paramD['manualFeatureSelection']['derivatives']['lastWaveLength'] = ['C','F']
    
    paramD['globalFeatureSelection'] = {}
    
    paramD['globalFeatureSelection']['comment'] ="removes spectra with variance below given thresholds - globally applied as preprocess",
    
    paramD['globalFeatureSelection']['apply'] = False
    
    paramD['globalFeatureSelection']['varianceThreshold'] = {'threshold': 0.025}
    
    
    paramD['modelFeatureSelection'] = {}
    
    paramD['modelFeatureSelection']['comment'] = 'feature selection using model data',
    
    paramD['modelFeatureSelection']['apply'] = False
    
    paramD['modelFeatureSelection']['univariateSelection'] = {}
    
    paramD['modelFeatureSelection']['univariateSelection']['apply'] = False
    
    paramD['modelFeatureSelection']['univariateSelection']['SelectKBest'] = {}
    
    paramD['modelFeatureSelection']['univariateSelection']['SelectKBest']['apply'] = False
    
    paramD['modelFeatureSelection']['univariateSelection']['SelectKBest']['n_features'] = 5
    
    
    paramD['modelFeatureSelection']['univariateSelection']['SelectPercentile'] = {}
    
    paramD['modelFeatureSelection']['univariateSelection']['SelectPercentile']['implemented'] = False
    
    paramD['modelFeatureSelection']['univariateSelection']['SelectPercentile']['apply'] = False
    
    paramD['modelFeatureSelection']['univariateSelection']['SelectPercentile']['percentile'] = 10
    

    paramD['modelFeatureSelection']['univariateSelection']['genericUnivariateSelect'] = {}
    
    paramD['modelFeatureSelection']['univariateSelection']['genericUnivariateSelect']['implemented'] = False
    
    paramD['modelFeatureSelection']['univariateSelection']['genericUnivariateSelect']['apply'] = False
    
    paramD['modelFeatureSelection']['univariateSelection']['genericUnivariateSelect']['hyperParameters'] = {}
    
    
    paramD['modelFeatureSelection']['RFE'] = {}
    
    paramD['modelFeatureSelection']['RFE']['apply'] = True
    
    paramD['modelFeatureSelection']['RFE']['CV'] = True
    
    paramD['modelFeatureSelection']['RFE']['n_features_to_select'] = 5
    
    paramD['modelFeatureSelection']['RFE']['step'] = 1
    
    
    paramD['featureAgglomeration'] = {}
    
    paramD['featureAgglomeration']['apply'] = False
    
    paramD['featureAgglomeration']['agglomerativeClustering'] = {}
    
    paramD['featureAgglomeration']['agglomerativeClustering']['apply'] = False 
    
    paramD['featureAgglomeration']['agglomerativeClustering']['implemented'] = False
    
    paramD['featureAgglomeration']['wardClustering'] = {}
    
    paramD['featureAgglomeration']['wardClustering']['apply'] = False 
    
    paramD['featureAgglomeration']['wardClustering']['n_cluster'] = 0
    
    paramD['featureAgglomeration']['wardClustering']['affinity'] = 'euclidean'
    
    paramD['featureAgglomeration']['wardClustering']['tuneWardClustering'] = {}
    
    paramD['featureAgglomeration']['wardClustering']['tuneWardClustering']['apply'] = False 
    
    paramD['featureAgglomeration']['wardClustering']['tuneWardClustering']['kfolds'] = 3 

    paramD['featureAgglomeration']['wardClustering']['tuneWardClustering']['clusters'] = [2,
            3,4,5,6,7,8,9,10,11,12]
        
    paramD['hyperParameterTuning'] = {}
    
    paramD['hyperParameterTuning']['apply'] = False
                                       
    paramD['hyperParameterTuning']['fraction'] = 0.5
    
    paramD['hyperParameterTuning']['nIterSearch'] = 6
    
    paramD['hyperParameterTuning']['n_top'] = 3
    
    paramD['hyperParameterTuning']['randomTuning'] = {}
    
    paramD['hyperParameterTuning']['randomTuning']['apply'] = False
    
    paramD['hyperParameterTuning']['exhaustiveTuning'] = {}
    
    paramD['hyperParameterTuning']['exhaustiveTuning']['apply'] = False
    
    paramD['featureImportance'] = {}
    
    paramD['featureImportance']['apply'] = True
                                       
    paramD['featureImportance']['reportMaxFeatures'] = 12
    
    paramD['featureImportance']['permutationRepeats'] = 10
            
    paramD['modelling'] = {}

    paramD['modelling']['apply'] = True
    
    paramD['regressionModels'] = {}
    
    paramD['regressionModels']['OLS'] = {}
    
    paramD['regressionModels']['OLS']['apply'] = False
    
    paramD['regressionModels']['OLS']['hyperParams'] = {}
    
    paramD['regressionModels']['OLS']['hyperParams']['fit_intercept'] = False
  
    paramD['regressionModels']['TheilSen'] = {}
    
    paramD['regressionModels']['TheilSen']['apply'] = False
    
    paramD['regressionModels']['TheilSen']['hyperParams'] = {}

    paramD['regressionModels']['Huber'] = {}
    
    paramD['regressionModels']['Huber']['apply'] = False
    
    paramD['regressionModels']['Huber']['hyperParams'] = {}
    
    paramD['regressionModels']['KnnRegr'] = {}
    
    paramD['regressionModels']['KnnRegr']['apply'] = False
    
    paramD['regressionModels']['KnnRegr']['hyperParams'] = {}
    
    paramD['regressionModels']['DecTreeRegr'] = {}
    
    paramD['regressionModels']['DecTreeRegr']['apply'] = False
    
    paramD['regressionModels']['DecTreeRegr']['hyperParams'] = {}
    
    paramD['regressionModels']['SVR'] = {}
    
    paramD['regressionModels']['SVR']['apply'] = False
    
    paramD['regressionModels']['SVR']['hyperParams'] = {}
    
    paramD['regressionModels']['SVR']['hyperParams']['kernel'] = 'linear'
    
    paramD['regressionModels']['SVR']['hyperParams']['C'] = 1.5
    
    paramD['regressionModels']['SVR']['hyperParams']['epsilon'] = 0.05
    
    paramD['regressionModels']['RandForRegr'] = {}
    
    paramD['regressionModels']['RandForRegr']['apply'] = False
    
    paramD['regressionModels']['RandForRegr']['hyperParams'] = {}
    
    paramD['regressionModels']['RandForRegr']['hyperParams']['n_estimators'] = 30
    
    
    paramD['regressionModels']['MLP'] = {}
    
    paramD['regressionModels']['MLP']['apply'] = False
    
    paramD['regressionModels']['MLP']['hyperParams'] = {}
    
    paramD['regressionModels']['MLP']['hyperParams']['hidden_layer_sizes'] = [100,100]
    
    paramD['regressionModels']['MLP']['hyperParams']['max_iter'] = 200
    
    paramD['regressionModels']['MLP']['hyperParams']['tol'] = 0.001
    
    paramD['regressionModels']['MLP']['hyperParams']['epsilon'] = 1e-8
    
    paramD['modelTests'] = {}
    
    paramD['modelTests']['trainTest'] = {}
    
    paramD['modelTests']['trainTest']['apply'] = False
    
    paramD['modelTests']['trainTest']['testSize'] = 0.3
    
    paramD['modelTests']['trainTest']['plot'] = True

    paramD['modelTests']['trainTest']['marker'] = 's'
    
    
    paramD['modelTests']['Kfold'] = {}
    
    paramD['modelTests']['Kfold']['apply'] = False
    
    paramD['modelTests']['Kfold']['folds'] = 10
    
    paramD['modelTests']['Kfold']['plot'] = True

    paramD['modelTests']['Kfold']['marker'] = '.'
    
       
    paramD['plot'] = {}
    
    paramD['plot']['apply'] = True
    
    paramD['plot']['subPlots'] = {}
    
    paramD['plot']['subPlots']['singles'] = {}
    
    paramD['plot']['subPlots']['singles']['apply'] = True
    
    paramD['plot']['subPlots']['singles']['regressor'] = False
    
    paramD['plot']['subPlots']['singles']['targetFeature'] = False
    
    paramD['plot']['subPlots']['singles']['hyperParameters'] = False
    
    paramD['plot']['subPlots']['singles']['modelTests'] = False
    
    paramD['plot']['subPlots']['rows'] = {}
    
    paramD['plot']['subPlots']['rows']['apply'] = True
    
    paramD['plot']['subPlots']['rows']['regressor'] = False
    
    paramD['plot']['subPlots']['rows']['targetFeature'] = False
    
    paramD['plot']['subPlots']['rows']['hyperParameters'] = False
    
    paramD['plot']['subPlots']['rows']['modelTests'] = False
    
    paramD['plot']['subPlots']['columns'] = {}
    
    paramD['plot']['subPlots']['columns']['apply'] = True
    
    paramD['plot']['subPlots']['columns']['regressor'] = False
    
    paramD['plot']['subPlots']['columns']['targetFeature'] = False
    
    paramD['plot']['subPlots']['columns']['hyperParameters'] = False
    
    paramD['plot']['subPlots']['columns']['modelTests'] = False
    
    paramD['plot']['subPlots']['doubles'] = {}
    
    paramD['plot']['subPlots']['doubles']['apply'] = True
    
    paramD['plot']['subPlots']['doubles']['columns'] = "regressor, targetFeature, hyperParameters or modelTest"
    
    paramD['plot']['subPlots']['doubles']['rows'] = "regressor, targetFeature, hyperParameters or modelTest"
    


    
    paramD['plot']['figSize'] = {'x':0,'y':0}
    
    paramD['plot']['legend'] = False
    
    paramD['plot']['tightLayout'] = False
    
    paramD['plot']['scatter'] = {'size':50}
    
    
        
    paramD['plot']['text'] = {'x':0.6,'y':0.2}
    
    paramD['plot']['text']['bandWidth'] = True
    
    paramD['plot']['text']['samples'] = True
    
    paramD['plot']['text']['text'] = ''
    
    paramD['figure'] = {} 
    
    paramD['figure']['apply'] = True
    
    
    return (paramD) 

def CreateArrangeParamJson(jsonFP, projFN, processstep):
    """ Create the default json parameters file structure, only to create template if lacking
    
        :param str dstrootFP: directory path 
        
        :param str jsonpath: subfolder under directory path 
    """
    
    def ExitMsgMsg(flag):
        
        if flag:
            
            exitstr = 'json parameter file already exists: %s\n' %(jsonFPN)
        
        else:
        
            exitstr = 'json parameter file created: %s\n' %(jsonFPN)
        
        exitstr += ' Edit the json file for your project and rename it to reflect the commands.\n' 
        
        exitstr += ' Add the path of the edited file to your project file (%s).\n' %(projFN)
        
        exitstr += ' Then set createjsonparams to False in the main section and rerun script.'
        
        exit(exitstr)
       
    if processstep.lower() in ['import','arrange']:
    
        # Get the default import params
        paramD = ImportParams()
        
        # Set the json FPN
        jsonFPN = path.join(jsonFP, 'template_import_ossl-spectra.json')
        
    if processstep.lower() in ['importxspectre','arrangexspectre']:
    
        # Get the default import params
        paramD = ImportXspectreParams()
        
        # Set the json FPN
        jsonFPN = path.join(jsonFP, 'template_import_xspectre-spectra.json')
        
    elif processstep.lower() in ['soilline','extract-soilline']:
    
        # Get the default import params
        paramD = SoilLineExtractParams()
        
        # Set the json FPN
        jsonFPN = path.join(jsonFP, 'template_soilline_ossl-spectra.json')
        
    elif processstep.lower() in ['plot']:
    
        # Get the default import params
        paramD = PlotParams()
        
        # Set the json FPN
        jsonFPN = path.join(jsonFP, 'template_plot_ossl-spectra.json')
        
    elif processstep.lower() in ['model','mlmodel']:
    
        # Get the default import params
        paramD = MLmodelParams()
        
        # Set the json FPN
        jsonFPN = path.join(jsonFP, 'template_model_ossl-spectra.json')
    
    if path.exists(jsonFPN):
        
        ExitMsgMsg(True)
    
    DumpAnyJson(paramD,jsonFPN)
    
    ExitMsgMsg(False)
    
def CheckMakeDocPaths(docpath,arrangeddatafolder, jsonpath, sourcedatafolder=False):
    """ Create the default json parameters file structure, only to create template if lacking
    
        :param str dstrootFP: directory path 
        
        :param str jsonpath: subfolder under directory path 
    """

    if not path.exists(docpath):
        
        exitstr = "The docpath does not exists: %s" %(docpath)
        
        exit(exitstr)
        
    if sourcedatafolder:
        
        srcFP = path.join(path.dirname(__file__),docpath,sourcedatafolder)
            
        if not path.exists(srcFP):
            
            exitstr = "The source data path to the original OSSL data does not exists:\n %s" %(srcFP)
            
            exit(exitstr)
        
    dstRootFP = path.join(path.dirname(__file__),docpath,arrangeddatafolder)
        
    if not path.exists(dstRootFP):
        
        makedirs(dstRootFP)
        
    jsonFP = path.join(dstRootFP,jsonpath)
    
    if not path.exists(jsonFP):
        
        makedirs(jsonFP)
        
    return dstRootFP, jsonFP

def ReadProjectFile(dstRootFP,projFN, jsonFP):
           
    projFPN = path.join(dstRootFP,projFN)

    if not path.exists(projFPN):

        exitstr = 'EXITING, project file missing: %s.' %(projFPN)
        
        exit( exitstr )

    infostr = 'Processing %s' %(projFPN)

    print (infostr)
    
    # Open and read the text file linking to all json files defining the project
    with open(projFPN) as f:

        jsonL = f.readlines()

    # Clean the list of json objects from comments and whithespace etc
    jsonProcessObjectL = [path.join(jsonFP,x.strip())  for x in jsonL if len(x) > 10 and x[0] != '#']
    
    return jsonProcessObjectL