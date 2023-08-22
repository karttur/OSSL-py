'''
Created on 3 Aug 2023

@author: thomasgumbricht

Notes
-----
The module plot.py:

    requires that you have soil spectra data organised as json files in xSpectre format. 
     
    The script takes 3 string parameters as input:
    
        - docpath: the full path to a folder that must contain the txt file as given by the "projFN" parameter
        - projFN: the name of an existing txt files that sequentially lists json parameter files to run
        - jsonpath: the relative path (vis-a-vis "docpath") where the json parameter files (listed in "projFN") are 
    
    The parameter files must list approximately 40 parameters in a precise nested json structure with dictionaries and lists.
    You can create a template json parameter file by running "def CreateParamJson" (just uncomment under "def SetupProcesses",
    this creates a template json parameter file called "extract_soillines.json" in the path given as the parameter "docpath".
    
    With an edited json parameter file the script reads the spectral data in xSpectre´s json format.
    The script first run the stand alone "def SetupProcesses" that reads the txt file "projFN" and 
    then sequentialy run the json parameter files listed. 
    
    Each soilline extraction (i.e. each json parameter file) is run as a separate instance of the class "SoilLine". 
    
    Each soilline extract process result in 2 json files, containg 1) the extacted soillines and 2) soil
    spectral endmembers for darksoil and lightsoil. The names of the destination files cannot be set by the 
    user, they are defaulted as follows:
    
    soillines result files:
    
        "rootFP"#visnirjson#visnir_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"_result-soillines.json
            
    endmember result files:
    
        "rootFP"#visnirjson#visnir_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"_result-endmembers.json
        
    If requested the script also produced png images showing the raw and/or final soillines:
    
        "rootFP"#visnirjson#visnir_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"_raw-soillines.png
        "rootFP"#visnirjson#visnir_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"_final-soillines.png
        
'''

# Standard library imports

import os

from copy import deepcopy

from math import ceil

# Third party imports

import tempfile

import json

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import randint as sp_randint
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from joblib import Memory

from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Outlier detection
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Feature selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE, RFECV

from sklearn.inspection import permutation_importance

# Package application imports

def PlotParams():
    ''' Default parameters for plotting soil spectral library data
    
        :returns: parameter dictionary
        :rtype: dict
    '''
    
    paramD = {}
    
    paramD['verbose'] = 1
    
    paramD['userid'] = "youruserid - any for now"
    
    paramD['importversion'] = "OSSL-202308"
    
    paramD['campaign'] = {'campaignshortid':'OSSL'}
    
    paramD['campaign']['campaigntype'] = 'laboratory'
    
    paramD['campaign']['theme'] = 'soil'
    
    paramD['campaign']['product'] = 'diffuse reflectance'
    
    paramD['campaign']['units'] = 'fraction'
        
    paramD['campaign']['georegion'] = "Sweden"
    
    paramD['input'] = {}
    
    paramD['input']['jsonSpectraDataFilePath'] = 'path/to(jsonfile/with/spectraldata.json'
    
    paramD['input']['jsonSpectraParamsFilePath'] = 'path/to(jsonfile/with/spectralparams.json'
        
    paramD['input']['bandjump'] = 1
        
    paramD['plot'] = {}
    
    paramD['plot']['raw'] = True
    
    paramD['plot']['derivative'] = True
            
    paramD['plot']['colorramp'] = "jet"
    
    paramD['plot']['maxspectra'] = 100
    
    paramD['plot']['figsize'] = {'x':0,'y':0}
    
    paramD['plot']['legend'] = False
    
    paramD['plot']['tight_layout'] = False
    
    paramD['plot']['scatter'] = {'size':50}
            
    paramD['plot']['text'] = {'x':0.1,'y':0.9}
    
    paramD['plot']['text']['bandwidth'] = True
    
    paramD['plot']['text']['samples'] = True
    
    paramD['plot']['text']['text'] = ''
    
    paramD['figure'] = {} 
    
    paramD['figure']['raw'] = True
    
    paramD['figure']['derivative'] = True
        
    paramD['xylim'] = {}
    
    paramD['xylim']['xmin'] = 15
    
    paramD['xylim']['xmax'] = 80
    
    paramD['xylim']['ymin'] = 45
    
    paramD['xylim']['ymax'] = 80
        
    return (paramD)
   
def CreateParamJson(docpath):
    """ Create the default json parameters file structure, only to create template if lacking
    
        :param str dstrootFP: directory path 
        
        :param str jsonpath: subfolder under directory path 
    """
    
    # Get the default params
    paramD = PlotParams()
    
    # Set the json FPN
    jsonFPN = os.path.join(docpath, 'template_plot_spectra.json')
    
    if os.path.exists(jsonFPN):
        
        return (True, jsonFPN)
    
    # Dump the paramD as a json object   
    jsonF = open(jsonFPN, "w")
  
    json.dump(paramD, jsonF, indent = 2)
  
    jsonF.close()
    
    return (False, jsonFPN)
    
def ReadModelJson(jsonFPN):
    """ Read the parameters for modeling
    
    :param jsonFPN: path to json file
    :type jsonFPN: str
    
    :return paramD: parameters
    :rtype: dict
   """
    
    with open(jsonFPN) as jsonF:
    
        paramD = json.load(jsonF)
        
    return (paramD)
    
class RegressionModels:
    
    '''Machinelearning using regression models
    '''
    def __init__(self):
        '''creates an empty instance of RegressionMode
        '''  

        self.modelSelectD = {}
        
        self.modelRetaindD = {}
        
        self.modD = {}
        
        #Create a list to hold retained columns
        self.retainD = {}
            
        self.retainPrintD = {}
        
        self.tunedModD = {}

    def _ExtractDataFrame(self):
        ''' Extract the original dataframe to X (covariate) array and y (predict) column
        '''

        # Extract the target feature
        self.y = self.abundanceDf[self.targetFeature]
        
        # Append the target array to the self.spectraDF dataframe      
        self.spectraDF['target'] = self.y
        
        # define the list of covariates to use
        self.columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]
        
        # extract the covariate columns as X
        self.X = self.spectraDF[self.columnsX]
        
        # Drop the added target column from the dataframe 
        self.spectraDF = self.spectraDF.drop('target', axis=1)
        
        # Remove all samples where the targetfeature is NaN
        self.X = self.X[~np.isnan(self.X).any(axis=1)]
        
        # Drop the added target column from self.X 
        self.X = self.X.drop('target', axis=1)
             
        # Then also delete NaN from self.y
        self.y = self.y[~np.isnan(self.y)]
        
    def _SetTargetFeatureSymbol(self):
        '''
        '''
        
        self.featureSymbolColor = 'black'
        
        self.featureSymbolMarker = '.'
        
        self.featureSymbolSize = 12
        
        if hasattr(self, 'targetFeatureSymbols'):
            
            if hasattr(self.targetFeatureSymbols, self.targetFeature):
                
                symbol = getattr(self.targetFeatureSymbols, self.targetFeature)
                
                if hasattr(symbol, 'color'):
                
                    self.featureSymbolColor = getattr(symbol, 'color')
                    
                if hasattr(symbol, 'marker'):
                
                    self.featureSymbolMarker = getattr(symbol, 'marker')
                    
                if hasattr(symbol, 'size'):
                
                    self.featureSymbolSize = getattr(symbol, 'size')
        
    def _PlotRegr(self, obs, pred, suptitle, title, txtstr,  txtstrHyperParams, color='black'):
        '''
        '''

        fig, ax = plt.subplots()
        ax.scatter(obs, pred, edgecolors=(0, 0, 0),  color=self.featureSymbolColor,
                   s=self.featureSymbolSize, marker=self.featureSymbolMarker)
        ax.plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'k--', lw=1)
        ax.set_xlabel('Observations')
        ax.set_ylabel('Predictions')
        plt.suptitle(suptitle)
        plt.title(title)
        plt.text(obs.min(), (obs.max()-obs.min())*0.8, txtstr,  wrap=True)
        
        #plt.text(obs.max()-((obs.max()-obs.min())*0.3), (obs.min()+obs.max())*0.1, txtstrHyperParams,  wrap=True)
        
        plt.show()
    
    def _RegModelSelectSet(self):
        """ Set the regressors to evaluate
        """
        
        self.regressorModels = []

        if hasattr(self.regressionModels, 'OLS') and self.regressionModels.OLS.apply:
            
            self.regressorModels.append(('OLS', linear_model.LinearRegression(**self.jsonparamsD['regressionModels']['OLS']['hyperparams'])))
            
            self.modelSelectD['OLS'] = []
            
        if hasattr(self.regressionModels, 'TheilSen') and self.regressionModels.TheilSen.apply:
            
            self.regressorModels.append(('TheilSen', linear_model.TheilSenRegressor(**self.jsonparamsD['regressionModels']['OLS']['hyperparams'])))
            
            self.modelSelectD['TheilSen'] = []
            
        if hasattr(self.regressionModels, 'Huber') and self.regressionModels.Huber.apply:
            
            self.regressorModels.append(('Huber', linear_model.HuberRegressor(**self.jsonparamsD['regressionModels']['OLS']['hyperparams'])))
            
            self.modelSelectD['Huber'] = []
            
        if hasattr(self.regressionModels, 'KnnRegr') and self.regressionModels.KnnRegr.apply:
            self.regressorModels.append(('KnnRegr', KNeighborsRegressor( **self.jsonparamsD['regressionModels']['KnnRegr']['hyperparams'])))
            self.modelSelectD['KnnRegr'] = []
            
        if hasattr(self.regressionModels, 'DecTreeRegr') and self.regressionModels.DecTreeRegr.apply:
            self.regressorModels.append(('DecTreeRegr', DecisionTreeRegressor(**self.jsonparamsD['regressionModels']['DecTreeRegr']['hyperparams'])))
            self.modelSelectD['DecTreeRegr'] = []
            
        if hasattr(self.regressionModels, 'SVR') and self.regressionModels.SVR.apply:
            self.regressorModels.append(('SVR', SVR(**self.jsonparamsD['regressionModels']['SVR']['hyperparams'])))
            self.modelSelectD['SVR'] = []
            
        if hasattr(self.regressionModels, 'RandForRegr') and self.regressionModels.RandForRegr.apply:
            self.regressorModels.append(('RandForRegr', RandomForestRegressor( **self.jsonparamsD['regressionModels']['RandForRegr']['hyperparams'])))
            self.modelSelectD['RandForRegr'] = []
    
    def _RegrModTrainTest(self):
        '''
        '''
       
        #Loop over the defined models
        for m in self.regressorModels:
            #Retrieve the model name and the model itself
            name,model = m
            
            if self.modelSelectD[name]:
            
                Xmodel = self.X[self.modelSelectD[name]]
                
                #Split the data into training and test subsets
                X_train, X_test, y_train, y_test = model_selection.train_test_split(Xmodel, self.y, test_size=self.modelTests.trainTest.testsize)

            else:
                
                #Split the data into training and test subsets
                X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=self.modelTests.trainTest.testsize)

            #Fit the model            
            model.fit(X_train, y_train)
            
            #Predict the independent variable in the test subset
            predict = model.predict(X_test)
            #Print out the model name
            print ('Model:', name)
            #Print out RMSE
            print("    Mean squared error: %.2f" \
                % mean_squared_error(y_test, predict))
            #Print explained variance score: 1 is perfect prediction
            print('    Variance (r2) score: %.2f' \
                % r2_score(y_test, predict))
 
            if self.modelTests.trainTest.plot:
                txtstr = 'nspectra: %s\n' %(self.X.shape[0])
                txtstr += 'nbands: %s\n' %(self.X.shape[1])
                #txtstr += 'min wl: %s\n' %(self.bandL[0])
                #txtstr += 'max wl: %s\n' %(self.bandL[len(self.bandL)-1])
                #txtstr += 'bands: %s\n' %( ' ,'.join('({0})'.format(w) for w in self.aggBandL)  )
                #txtstr += 'width wl: %s' %(int(self.bandL[1]-self.bandL[0]))
                
                #txtstrHyperParams =  self.HPtuningtxt+'\nHyper Parameters:\n'+'\n'.join([key+': '+str(value) for key, value in self.tunedModD[name].items()])
                suptitle = '%s ML Regression training/test model (testsize = %s)' %(self.targetFeature, self.modelTests.trainTest.testsize)
                title = ('Model: %(mod)s; RMSE: %(rmse)2f; r2: %(r2)2f' \
                          % {'mod':name,'rmse':mean_squared_error(y_test, predict),'r2': r2_score(y_test, predict)} )
                self._PlotRegr(y_test, predict, suptitle, title, txtstr, '', color='green')
            
    def _RegrModKFold(self):
        """
        """
        
        
        #Loop over the defined models
        for m in self.regressorModels:
            #Retrieve the model name and the model itself
            name,model = m
            
            if self.modelSelectD[name]:
            
                Xmodel = self.X[self.modelSelectD[name]]
                
                predict = model_selection.cross_val_predict(model, Xmodel, self.y, cv=self.modelTests.Kfold.folds)

            else:
   
                predict = model_selection.cross_val_predict(model, self.X, self.y, cv=self.modelTests.Kfold.folds)
                

            scoring = 'r2'
            r2 = model_selection.cross_val_score(model, self.X, self.y, cv=self.modelTests.Kfold.folds, scoring=scoring)
            #The correlation coefficient
            #Print out the model name
            print ('Model:', name)
            print (model)
            #Print out correlation coefficients
            print('    Regression coefficients: \n', r2)    
            #Print out RMSE
            print("Mean squared error: %.2f" \
                  % mean_squared_error(self.y, predict))
            #Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' \
                  
                % r2_score(self.y, predict))
            
            if self.modelTests.Kfold.plot:
                '''
                if self.varianceSelect:
                    #Run the feature selection process  
                    self.regmods.VarianceSelector(self.varianceThreshold)
                    txtstr = 'Variance feature selected bands (n=%s)\n' %(self.X.shape[1])
                    
                elif self.KBestSelect:
                    txtstr = 'KBest feature selected bands (n=%s)\n' %(self.X.shape[1])
            
                elif self.RFESelect:    
                    txtstr = 'RFE feature selected bands (n=%s)\n' %(self.X.shape[1])
                    
                elif self.RFECVSelect:    
                    txtstr = 'RFECV feature selected bands (n=%s)\n' %(self.X.shape[1])
                    
                else:
                    txtstr = 'No feature selected bands (n=%s)\n' %(self.X.shape[1])
                '''  
                txtstr = 'nbands: %s\n' %(self.X.shape[1])
                #txtstr += self.aggBandL
                #txtstr += 'bands: %s\n' %( ' ,'.join('({0})'.format(w) for w in self.aggBandL)  )
                #txtstr += 'min wl: %s\n' %(self.bandL[0])
                #txtstr += 'max wl: %s\n' %(self.bandL[len(self.bandL)-1])
                #txtstr += 'width wl: %s' %(int(self.bandL[1]-self.bandL[0]))

                
                #txtstrHyperParams =  self.HPtuningtxt+'\nHyper Parameters:\n'+'\n'.join([key+': '+str(value) for key, value in self.tunedModD[name].items()])

                suptitle  = '%s ML Regression kfold model (folds = %s)' %(self.targetFeature, self.modelTests.Kfold.folds)
                title = ('Model: %(mod)s; RMSE: %(rmse)2f; r2: %(r2)2f' \
                          % {'mod':name,'rmse':mean_squared_error(self.y, predict),'r2': r2_score(self.y, predict)} )
                self._PlotRegr(self.y, predict, suptitle, title,txtstr, '', color='blue')
         
    def _FeatureImportance(self):
        '''
        '''
       
        #Loop over the defined models
        for m in self.regressorModels:
            #Retrieve the model name and the model itself
            name,model = m
            
            if self.modelSelectD[name]:
            
                Xmodel = self.X[self.modelSelectD[name]]
                
                #Split the data into training and test subsets
                X_train, X_test, y_train, y_test = model_selection.train_test_split(Xmodel, self.y, test_size=self.modelTests.trainTest.testsize)

            else:
                
                #Split the data into training and test subsets
                X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=self.modelTests.trainTest.testsize)

            #Fit the model            
            model.fit(X_train, y_train)
            
            maxFeatures = min(self.featureImportance.reportMaxFeatures, len(self.columns))
            
            # Permutation importance
            
            n_repeats = self.featureImportance.permutationRepeats
            
            permImportance = permutation_importance(model, X_test, y_test, n_repeats=n_repeats)
            
            permImportanceMean = permImportance.importances_mean
            
            permImportanceStd = permImportance.importances_std
            
            sorted_idx = permImportanceMean.argsort()
            
            permImportanceArray = permImportanceMean[sorted_idx][::-1][0:maxFeatures]
            
            errorArray = permImportanceStd[sorted_idx][::-1][0:maxFeatures]
            
            featureArray = np.asarray(self.columns)[sorted_idx][::-1][0:maxFeatures]
            
            title = "Permutation importance\n Feature: %s; Model: %s" %(self.targetFeature, name)
            
            # Convert to a pandas series
            permImportanceDF = pd.Series(permImportanceArray, index=featureArray)
                
            fig, ax = plt.subplots()
                
            permImportanceDF.plot.bar(yerr=errorArray, color=self.featureSymbolColor, ax=ax)
                
            ax.set_title(title)
                
            ax.set_ylabel("Mean accuracy decrease")
                
            fig.tight_layout()
                
            #plt.barh(featureArray[::-1], permutationArray[::-1])
            #plt.title(title)
            
            plt.show()
            
            # Feature importance
            
            if name in ['OLS','TheilSen','Huber', "Ridge", "ElasticNet", 'logistic']:
                
                if name in ['logistic']:
                
                    importances = model.coef_[0]
                
                else:
                    
                    importances = model.coef_
                    
                sorted_idx = importances.argsort()
                
                importanceArray = importances[sorted_idx][::-1][0:maxFeatures]
                
                featureArray = np.asarray(self.columns)[sorted_idx][::-1][0:maxFeatures]
                
                # Convert to a pandas series
                linearImportances = pd.Series(importanceArray, index=featureArray)
                
                fig, ax = plt.subplots()
                
                linearImportances.plot.bar(color=self.featureSymbolColor, ax=ax)
                title = "Linear feature coefficients\n Feature: %s; Model: %s" %(self.targetFeature, name)

                ax.set_title(title)
                
                ax.set_ylabel("Coefficient")
                
                fig.tight_layout()
                
                plt.show()
                
                BALLE
                
                '''
                # summarize feature importance
                for i,v in enumerate(importance):
                    print('Feature: %0d, Score: %.5f' % (i,v))
                    # plot feature importance
                    pyplot.bar([x for x in range(len(importance))], importance)
                    pyplot.show()
                '''
            
            elif name in ['RandForRegr']:
                
                importances = model.feature_importances_
                
                std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
            
                sorted_idx = importances.argsort()
                
                importanceArray = importances[sorted_idx][::-1][0:maxFeatures]
                
                errorArray = std[sorted_idx][::-1][0:maxFeatures]
                
                featureArray = np.asarray(self.columns)[sorted_idx][::-1][0:maxFeatures]
                
                # Convert to a pandas series
                forest_importances = pd.Series(importanceArray, index=featureArray)
                
                fig, ax = plt.subplots()
                
                forest_importances.plot.bar(yerr=errorArray, color=self.featureSymbolColor, ax=ax)
                title = "MDI feature importance\n Feature: %s; Model: %s" %(self.targetFeature, name)

                ax.set_title(title)
                
                ax.set_ylabel("Mean decrease in impurity")
                
                fig.tight_layout()
                
                plt.show()
                
                BALLE
                
                '''
                sorted_idx = model.feature_importances_.argsort()
            
                importanceArray = model.feature_importances_[sorted_idx][::-1][0:maxFeatures]
                '''
            else:
                
                continue
            
            featureArray = np.asarray(self.columns)[sorted_idx][::-1][0:maxFeatures]
            
            plt.barh(featureArray[::-1], importanceArray[::-1])
            
            plt.show()
            
            
            BALLE
            
            
    def _ManualFeatureSelector(self):
        '''
        '''
        
        # Reset self.columns
        self.columns = self.manualFeatureSelection.spectra
        
        # Create the dataframe for the sepctra, OK if emply
        
        spectraDF = self.spectraDF[ self.columns  ]
        
        # Create any derivative covariates requested
        
        for b in range(len(self.manualFeatureSelection.derivatives.firstwavelength)):
            
            bandL = [self.manualFeatureSelection.derivatives.firstwavelength[b],
                     self.manualFeatureSelection.derivatives.lastwavelength[b]]
            
            derviationBandDF = self.spectraDF[ bandL  ]
                        
            bandFrame, bandColumn = self._SpectraDerivativeFromDf(derviationBandDF,bandL)

   
            frames = [spectraDF,bandFrame]
                
            spectraDF = pd.concat(frames, axis=1)
                
            self.columns.extend(bandColumn)
            
        # reset self.spectraDF
        self.spectraDF = spectraDF
         
    def _VarianceSelector(self):
        '''
        '''
        
        threshold = self.globalFeatureSelection.variancethreshold.threshold
        
        #istr = 'Selected features:\nVarianceThreshold (%s)'% threshold
        
        #self.selectstrL.append(istr)
        
        # define the list of covariates to use
        self.columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]
        
        # extract the covariate columns as X
        X = self.spectraDF[self.columnsX]
        
        #Initiate the MinMaxScale
        scaler = MinMaxScaler()
        
        scaler.fit(X)
        
        #Scale the data as defined by the scaler
        Xscaled = scaler.transform(X)
        
        #Initiate  VarianceThreshold
        select = VarianceThreshold(threshold=threshold)
        
        #Fit the independent variables
        select.fit(Xscaled)  
        
        #Get the selected features from get_support as a boolean list with True or False  
        selectedFeatures = select.get_support()
        
        #Create a list to hold discarded columns
        discardL = []
        
        #Create a list to hold retained columns
        self.retainL = []
        
        if self.verbose:
        
            print ('    Selecting features using  VarianceThreshold, threhold =',threshold)
        
            print ('    Scaling function:',scaler.fit(X))
        
        for sf in range(len(selectedFeatures)):

            if selectedFeatures[sf]:
                self.retainL.append([self.columnsX[sf],select.variances_[sf]])

            else:
                discardL.append([self.columnsX[sf],select.variances_[sf]])
                
        if self.verbose:
            
            #print the selected features and their variance
            print ('    Discarded features [name, (variance):')
                
            printL = ['%s (%.3f)'%(i[0],i[1]) for i in discardL]
                        
            for row in printL:
                print ('    ',row)
                
            print ('\n    Retained features [name, (variance)]:')
            
            printretainL = ['%s (%.3f)'%(i[0], i[1]) for i in self.retainL]
            
            for row in printretainL:
                print ('    ',row)
                        
        self.retainL = [d[0] for d in self.retainL]
        
        self.spectraDF = self.spectraDF[ self.retainL ]
        
    def _UnivariateSelector(self):
        '''
        '''
        nfeatures = self.X.shape[1]
        
        if self.modelFeatureSelection.univariateSelection.SelectKBest.apply:
            
            n_features = self.modelFeatureSelection.univariateSelection.SelectKBest.n_features
   
            if n_features >= nfeatures:
            
                if self.verbose:
                    
                    infostr = '    SelectKBest: Number of features (%s) less than or equal to n_features (%s).' %(nfeatures,n_features_to_select)
                
                    print (infostr)
                    
                return 
 
            
            select = SelectKBest(score_func=f_regression, k=n_features)
           
        else:
            
            return
        
        # Select and fit the independent variables, return the selected array
        self.X = select.fit_transform(self.X, self.y)
        
        if self.verbose:
            
            print ('    Discarded %s features using SelectKBest' %(nfeatures-self.X.shape[1]))
            
    def _RFESelector(self):
        '''
        '''
                
        nfeatures = self.X.shape[1]
        
        n_features_to_select = self.modelFeatureSelection.RFE.n_features_to_select
                
        if n_features_to_select >= nfeatures:
            
            if self.verbose:
                
                infostr = '    Number of features (%s) less than or equal to n_features_to_select (%s)' %(nfeatures,n_features_to_select)
            
                print (infostr)
                
            return 
        
        step = self.modelFeatureSelection.RFE.step
        
        columns = self.X.columns
        
        if self.verbose:
            
            if self.modelFeatureSelection.RFE.CV: 
            
                print ('    RFECV feature selection, model adapted')
                
            else: 
            
                print ('    RFE feature selection, model adapted')
                
        
        for m in self.regressorModels:
            
            #Retrieve the model name and the model itself
            name,model = m
            
            
            if self.modelFeatureSelection.RFE.CV:
                                    
                select = RFECV(estimator=model, min_features_to_select=n_features_to_select, step=step)
                  
            else:
                                    
                select = RFE(estimator=model, n_features_to_select=n_features_to_select, step=step)
        
            select.fit(self.X, self.y)
            
            selectedFeatures = select.get_support()

            #Create a list to hold discarded columns
            selectL = []; discardL = []
            
            #print the selected features and their variance
            for sf in range(len(selectedFeatures)):
                if selectedFeatures[sf]:
                    selectL.append(columns[sf])
                    
                else:
                    discardL.append(columns[sf])
            if self.verbose:

                print ('            Regressor: %(m)s; Target feature: %(t)s' %{'m':name,'t':self.targetFeature})

                print ('            Selected features: %s' %(', '.join(selectL)))
                
            if self.verbose > 1:
                
                print ('            Discarded features: %s' %(', '.join(discardL)))

            self.modelSelectD[name] = selectL
 
        
    def _RemoveOutliers(self):
        """
        """
              
        self.columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]
        
        # extract the covariate columns as X
        X = self.spectraDF[self.columnsX]
          
        iniSamples = X.shape[0]
        
        if self.removeOutliers.detector.lower() in ['iforest','isolationforest']:
            
            self.outliertxt = 'outliers removed with iforest'
            
            outlierDetector = IsolationForest(contamination=self.removeOutliers.contamination)
            
        elif self.removeOutliers.detector.lower() in ['ee','eenvelope','ellipticenvelope']:
            
            self.outliertxt = 'outliers removed with eenvelope'
            
            outlierDetector = EllipticEnvelope(contamination=self.removeOutliers.contamination)
            
        elif self.removeOutliers.detector.lower() in ['lof','lofactor','localoutlierfactor']:
            
            self.outliertxt = 'outliers removed with local out fac'
        
            outlierDetector = LocalOutlierFactor(contamination=self.removeOutliers.contamination)
            
        elif self.removeOutliers.detector.lower() in ['1csvm','1c-svm','oneclasssvm']:
            
            self.outliertxt = 'outliers removed with 1cSVM'
        
            outlierDetector = OneClassSVM(nu=self.removeOutliers.contamination)
            
        else:
            
            exit('unknown outlier detector') 
            
        # The warning "X does not have valid feature names" is issued, but it is a bug and will go in next version
        yhat = outlierDetector.fit_predict(X)

        # select all rows that are not outliers
        mask = yhat != -1
                
        X['yhat'] = yhat
        
        # Remove samples with outliers from the abudance array using the X array yhat columns
        self.abundanceDf = self.abundanceDf[ X['yhat']==1 ]
        
        # Keep only non-outliers in self.X                        
        X = X[ X['yhat']==1 ]
        
        # Drop the "yhat" columns for self.X
        X = X.drop(['yhat'], axis=1)
        
        #self.spectraDF = pd.DataFrame(data=X[1:,1:], index=X[1:,0], columns=X[0,1:]) 
        
        self.spectraDF = pd.DataFrame(X) 
        
        postSamples = X.shape[0]
        
        self.nOutliers = iniSamples - postSamples
        
        self.outliertxt = '%s %s' %(self.nOutliers, self.outliertxt)
                
        if self.verbose:
            
            print ('        ',self.outliertxt)
            
    def _WardClustering(self, n_clusters):
        '''
        '''
        
        nfeatures = self.X.shape[1]
        
        if nfeatures < n_clusters:
            
            n_clusters = nfeatures
                
        ward = FeatureAgglomeration(n_clusters=n_clusters)
        
        #fit the clusters
        ward.fit(self.X, self.y) 
        
        self.clustering =  ward.labels_
        
        # Get a list of bands
        bandsL =  list(self.X)
        
        self.aggColumnL = []
        
        self.aggBandL = []
        
        for m in range(len(ward.labels_)):
            
            indices = [bandsL[i] for i, x in enumerate(ward.labels_) if x == m]
            
            if(len(indices) == 0):
                
                break
            
            self.aggColumnL.append(indices[0])
                           
            self.aggBandL.append( ', '.join(indices) )

        if self.verbose > 1:  
              
            print ('feature_names_in_',ward.n_features_in_)
        
            print ('Initial X data shape:',self.X.shape)
            
            print (self.X)
 
        # Reset the covariates (self.X)
        X = ward.transform(self.X)

        self.X = pd.DataFrame(X, columns=self.aggColumnL)
        
    def _TuneWardClustering(self):
        ''' Determines the optimal nr of cluster
        '''
        nfeatures = self.X.shape[1]
        
        nClustersL = self.featureAgglomeration.WardClustering.TuneWardClustering.clusters
        
        nClustersL = [c for c in nClustersL if c < nfeatures]
        
        kfolds = self.featureAgglomeration.WardClustering.TuneWardClustering.kfolds
        
        cv = KFold(kfolds)  # cross-validation generator for model selection
        
        ridge = BayesianRidge()
        
        cachedir = tempfile.mkdtemp()
        
        mem = Memory(location=cachedir)
        
        ward = FeatureAgglomeration(n_clusters=4, memory=mem)
        
        clf = Pipeline([('ward', ward), ('ridge', ridge)])
        
        # Select the optimal number of parcels with grid search
        clf = GridSearchCV(clf, {'ward__n_clusters': nClustersL}, n_jobs=1, cv=cv)
        
        clf.fit(self.X, self.y)  # set the best parameters
        
        #report the top three results
        self.ReportSearch(clf.cv_results_,3)

        #rerun with the best cluster agglomeration result
        tunedClusters = clf.best_params_['ward__n_clusters']

        return (tunedClusters)
       
    def RandomTuningParams(self,nFeatures):
        ''' Set hyper parameters random tuning
        '''
        self.paramDist = {}
        self.HPtuningtxt = 'Random tuning'
        # specify parameters and distributions to sample from
        for m in self.regressorModels:
            
            name,model = m
            
            print ('name'), (name), (model.get_params())
            
            if name == 'KnnRegr':
                
                self.paramDist[name] = {"n_neighbors": sp_randint(self.RandomTuning.KnnRegr.n_neigbors.min, 
                                                                  self.RandomTuning.KnnRegr.n_neigbors.max),
                              'leaf_size': sp_randint(self.RandomTuning.KnnRegr.leaf_size.min, 
                                                                  self.RandomTuning.KnnRegr.leaf_size.max),
                              'weights': self.RandomTuning.KnnRegr.weights,
                              'p': self.RandomTuning.KnnRegr.weights,
                              'algorithm': self.RandomTuning.KnnRegr.algorithm}
                
            elif name =='DecTreeRegr':
                # Convert 0 to None for max_depth
                
                max_depth = [m if m > 0 else None for m in self.RandomTuning.DecTreeRegr.max_depth]
                
                self.paramDist[name] = {"max_depth": max_depth,
                            "min_samples_split": sp_randint(self.RandomTuning.DecTreeRegr.min_samples_split.min, 
                                                            self.RandomTuning.DecTreeRegr.min_samples_split.max),
                            "min_samples_leaf": sp_randint(self.RandomTuning.DecTreeRegr.min_samples_leaf.min, 
                                                            self.RandomTuning.DecTreeRegr.min_samples_leaf.max)}
            elif name =='SVR':
                
                self.paramDist[name] = {"kernel": self.RandomTuning.SVR.kernel,
                                        "epsilon": self.RandomTuning.SVR.epsilon,
                                        "C": self.RandomTuning.SVR.epsilon}
                
            elif name =='RandForRegr':
                
                max_depth = [m if m > 0 else None for m in self.RandomTuning.RandForRegr.max_depth]
                 
                max_features_max = min(self.RandomTuning.RandForRegr.max_features.max,nFeatures)
                
                max_features_min = min(self.RandomTuning.RandForRegr.max_features.min,nFeatures)
                
                self.paramDist[name] = {"max_depth": max_depth,
                              "n_estimators": sp_randint(self.RandomTuning.RandForRegr.n_estimators.min, 
                                                                  self.RandomTuning.RandForRegr.n_estimators.max),
                              "max_features": sp_randint(1, max_features_min, 
                                                                  max_features_max),
                              "min_samples_split": sp_randint(self.RandomTuning.RandForRegr.min_samples_split.min, 
                                                                  self.RandomTuning.RandForRegr.min_samples_split.max),
                              "min_samples_leaf": sp_randint(self.RandomTuning.RandForRegr.min_samples_leaf.min, 
                                                                  self.RandomTuning.RandForRegr.min_samples_leaf.max),
                              "bootstrap": [True,False]}
                        
    def RandomTuning(self):
        '''
        '''
        
        
        for m in self.models:
            #Retrieve the model name and the model itself
            name,mod = m
            print (name, self.paramDist[name])

            search = RandomizedSearchCV(mod, param_distributions=self.paramDist[name],
                                               n_iter=self.RandomTuning.nIterSearch)
            X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=(1-self.RandomTuning.fraction))
            
            search.fit(X_train, y_train)
            
            self.ReportSearch(search.cv_results_,self.RandomTuning.n_best_report)
            
            #Retrieve the top ranked tuning
            best = np.flatnonzero(search.cv_results_['rank_test_score'] == 1)
            
            tunedModD=search.cv_results_['params'][best[0]]
            
            #Append any initial modD hyper-parameter definition 
            # The self version is for printing
            self.tunedModD[name]=search.cv_results_['params'][best[0]]
            
            for key in self.modD[name]:
                
                tunedModD[key] = self.modD[name][key] 
                
            self.regmods.modD[name] = tunedModD
    
    def _ReportRegModelParams(self):
        '''
        '''
        
        print ('Model hyper-parameters:')
        
        for model in self.regressorModels:
            
            #Retrieve the model name and the model itself
            modelname,modelhyperparams = model
            
            print ('    name', modelname, modelhyperparams.get_params()) 
            
    def ReportSearch(self, results, n_top=3):
        '''
        '''
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("") 
                
        SNULLE
                          
class Obj(object):
    ''' Convert json parameters to class objects
    '''
    
    def __init__(self, paramD):
        ''' Convert nested dict to nested class object
        
        :param dict paramD: parameters 
        '''

        for k, v in paramD.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [Obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Obj(v) if isinstance(v, dict) else v)
                
    def _SetDefautls(self):
        ''' Set class object default data if required
        '''
        
        if self.plot.figsize.x == 0:
            
            self.plot.figsize.x = 4
            
        if self.plot.figsize.y == 0:
            
            self.plot.figsize.y = 4
            
        # Check if Manual feature selection is set
        if self.manualFeatureSelection.apply:
            
            # Turn off the derivates alteratnive (done as part of the manual selection if requested)
            self.derivatives.apply = False
            
            # Turn off all other feature selection/agglomeration options
            self.globalFeatureSelection.apply = False
            
            self.modelFeatureSelection.apply = False
            
            self.featureAgglomeration.apply = False
                   
    def _SetTextPos(self, xmin, xmax, ymin, ymax):
        ''' Set position of text objects for matplotlib
        
            :param float xmin: x-axis minimum
            
            :param float xmax: x-axis maximum
            
            :param float ymin: y-axis minimum
            
            :param float ymax: y-axis maximum
            
            :returns: text x position
            :rtype: float
            
            :returns: text y position
            :rtype: float 
        '''
        
        x = self.plot.text.x*(xmax-xmin)+xmin
        
        y = self.plot.text.y*(ymax-ymin)+ymin
        
        return (x,y)
                    
class MachineLearningModel(Obj, RegressionModels):
    ''' MAchine Learning model of feature propertie from spectra
    '''
    
    def __init__(self,paramD): 
        """ Convert input parameters from nested dict to nested class object
        
            :param dict paramD: parameters 
        """
        
        # convert the input parameter dict to class objects
        Obj.__init__(self,paramD)
        
        # initiate the regression models
        RegressionModels.__init__(self)
                
        # Set class object default data if required
        self._SetDefautls()
        
        # Deep copy parameters to a new object class called params
        self.params = deepcopy(self)
        
        # Drop the plot and figure settings from paramD
        paramD.pop('plot'); paramD.pop('figure')
                
        # Deep copy the parameters to self.soillineD
        self.plotD = deepcopy(paramD)
               
        # Open and load JSON data file
        with open(self.input.jsonSpectraDataFilePath) as jsonF:
            
            self.jsonSpectraData = json.load(jsonF)
            
        with open(self.input.jsonSpectraParamsFilePath) as jsonF:
            
            self.jsonSpectraParams = json.load(jsonF)
                    
        #self.plotD['raw'] = {}
        
        #self.plotD['derivative'] = {}
                
    def _SetColorRamp(self,n):
        ''' Slice predefined colormap to discrete colors for each band
        '''
                        
        # Set colormap to use for plotting
        cmap = plt.get_cmap(self.plot.colorramp)
        
        # Segmenting colormap to the number of bands
        self.slicedCM = cmap(np.linspace(0, 1, n)) 
                 
    def _GetAbundanceData(self):
        '''
        '''
        
        # Get the list of substances included in this dataset
        
        substanceColumns = self.jsonSpectraParams['labData']
        
        #substanceColumns = self.jsonSpectraParams['targetFeatures']
        
        substanceOrderD = {}
        
        for substance in substanceColumns:
            
            substanceOrderD[substance] = substanceColumns.index(substance)
        
        n = 0
        
        # Loop over the samples
        for sample in self.jsonSpectraData['spectra']:
            
            substanceL = [None] * len(substanceColumns)
            
            for abundance in sample['abundance']:
                     
                substanceL[ substanceOrderD[abundance['substance']] ] = abundance['value']
                
            if n == 0:
            
                abundanceA = np.asarray(substanceL, dtype=float)
            
            else:
                 
                abundanceA = np.vstack( (abundanceA, np.asarray(substanceL, dtype=float) ) )
            
            n += 1
                               
        self.abundanceDf = pd.DataFrame(data=abundanceA, columns=substanceColumns)
         
    def _StartStepSpectra(self, pdSpectra, startwl, stopwl, stepwl):
        '''
        '''
        
        wlmin = pdSpectra['wl'].min()
        wlmax = pdSpectra['wl'].max()
        step = (wlmax-wlmin)/(pdSpectra.shape[0])
        
        startindex = (startwl-wlmin)/step
        
        stopindex = (stopwl-wlmin)/step
            
        stepindex = stepwl/step
            
        indexL = []; iL = []
        
        i = 0
        
        while True:
            
            if i*stepindex+startindex > stopindex+1:
                
                break
            
            indexL.append(int(i*stepindex+startindex))
            iL.append(i)
            
            i+=1
        
        df = pdSpectra.iloc[indexL]
            
        return df, indexL[0]
    
    def _SpectraDerivativeFromDf(self,dataFrame,columns):
        ''' Create spectral derivates
        '''

        # Get the derivatives
        spectraDerivativeDF = dataFrame.diff(axis=1, periods=1)
        
        # Drop the first column as it will have only NaN
        spectraDerivativeDF = spectraDerivativeDF.drop(columns[0], axis=1)
               
        # Reset columns to integers
        columns = [int(i) for i in columns]
        
        # Create the derivative columns 
        derivativeColumns = ['d%s' % int((columns[i-1]+columns[i])/2) for i in range(len(columns)) if i > 0]
        
        # Replace the columns
        spectraDerivativeDF.columns = derivativeColumns
        
        return spectraDerivativeDF, derivativeColumns
                 
    def _GetBandData(self):
        ''' Read json data into numpy array and convert to pandas dataframe
        '''
                        
        # Use the wavelength as column headers
        self.columns = self.jsonSpectraData['waveLength']
        
        # Convert the column headers to strings
        self.columns = [str(c) for c in self.columns]
                 
        n = 0
                       
        # Loop over the spectra
        for sample in self.jsonSpectraData['spectra']:
                                    
            if n == 0:
            
                spectraA = np.asarray(sample['signalMean'])
            
            else:
                 
                spectraA = np.vstack( (spectraA, np.asarray(sample['signalMean']) ) )
            
            n += 1
                               
        self.spectraDF = pd.DataFrame(data=spectraA, columns=self.columns)
        
        if self.derivatives.apply:
            
            spectraDerivativeDF,derivativeColumns = self._SpectraDerivativeFromDf(self.spectraDF, self.columns)
        
            if self.derivatives.join:
                
                frames = [self.spectraDF, spectraDerivativeDF]

                self.spectraDF = pd.concat(frames, axis=1)
                                
                self.columns.extend(derivativeColumns)
                 
            else:
                
                self.spectraDF = spectraDerivativeDF
                
                self.columns = derivativeColumns
        

        
    def _SetDstFPNs(self):
        ''' Set destination file paths and names
        '''

        FP,FN = os.path.split(self.input.jsonSpectraDataFilePath)
                
        FN = os.path.splitext(FN)[0]
        
        self.modelN = FN.split('_', 1)[1]
            
        modelRootFP = os.path.join(FP,'mlmodel')
        
        if not os.path.exists(modelRootFP):
            
            os.makedirs(modelRootFP)
        
        '''
        rawPngFN = '%s_raw.png' %(self.modelN)

        self.rawPngFPN = os.path.join(plotRootFP, rawPngFN)

        derivativePngFN = '%s_final-soillines.png' %(self.modelN)
        
        self.derivativePngFPN = os.path.join(plotRootFP, derivativePngFN)
        '''
                                          
    def _PlotSingle(self, X, Y, regr, xlabel, ylabel, title, i):
        ''' Regression and plot for single band
        
        '''
        
        xlabel='%s reflectance' %(xlabel); ylabel='%s reflectance' %(ylabel)
        
        fig, ax = plt.subplots( figsize=(self.plot.figsize.x, self.plot.figsize.y) )
        
        ax.scatter(X, Y, size=self.plot.scatter.size, color=self.slicedCM[i])
        
        ax.plot(X, regr.predict(X), color=self.slicedCM[i])
        
        ax.set_xlim(self.xylimD['xmin'], self.xylimD['xmax'])
                
        ax.set_ylim(self.xylimD['ymin'], self.xylimD['ymax'])
        
        ax.set(xlabel=xlabel, ylabel=ylabel,
                   title=title)
        
        plt.show()
        
    def _PlotMulti(self,plot, figure, pngFPN):
        ''' Regression and plot for multiple bands
        
            :param str xlabel: x-axis label
            
            :param str ylabel: y-axis label
            
            :param str title: title
            
            :param str text: text
            
            :param bool plot: interactive plot or not
            
            :param bool figure: save as file or not
            
            :param str pngFPN: path for saving file
            
            :returns: regression results
            :rtype: dict
        '''

        # Get the bands to plot
                
        plotskipstep = ceil( (len(self.spectraDF.index)-1)/self.plot.maxspectra )
                
        # Set the plot title, labels and annotation
        titleSuffix = ''
        
        xLabel, yLabel, title, text = self._PlotTitleTextn(titleSuffix,plotskipstep)

                       
        fig, ax = plt.subplots( figsize=(self.plot.figsize.x, self.plot.figsize.y)  )
        
        if self.plot.tight_layout:
            
            fig.tight_layout()

        n = int(len(self.spectraDF.index)/plotskipstep)+1
        
        # With n bands known, create the colorRamp
        self._SetColorRamp(n)
        
        # Loop over the spectra
        i = -1
        n = 0
        for index, row in self.spectraDF.iterrows():
            i += 1
            if i % plotskipstep == 0:
                
                ax.plot(self.columns, row, color=self.slicedCM[n])
                
                n += 1
                     
        if not self.xylimD:
            
            xmin,xmax = ax.get_xlim()
            
            ymin,ymax = ax.get_ylim()
            
            self.xylimD = {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}
            
        else:
            
            ax.set_xlim(self.xylimD['xmin'], self.xylimD['xmax'])
            
            ax.set_ylim(self.xylimD['ymin'], self.xylimD['ymax'])
        
        ax.set(xlabel=xLabel, ylabel=yLabel, title=title)
        
        if self.plot.legend:
                    
            ax.legend(loc=self.plot.legend)
        
        if text != None:
            
            x,y = self._SetTextPos(self.xylimD['xmin'], self.xylimD['xmax'], self.xylimD['ymin'], self.xylimD['ymax'])
            
            ax.text(x, y, text)
                                        
        if plot:
        
            plt.show()
          
        if figure:
          
            fig.savefig(pngFPN)   # save the figure to file
            
            plt.close(fig)

    def _PlotTitleTextn(self, titleSuffix,plotskipstep):
        ''' Set plot title and annotation
        
            :param str titleSuffix: amendment to title
            
            :returns: x-axis label
            :rtype: str
        
            :returns: y-axis label
            :rtype: str
            
            :returns: title
            :rtype: str
            
            :returns: text
            :rtype: str
        '''
        
        # Set title
        title = self.modelN
    
        # set the text
        text = self.plot.text.text
        
        # Add the bandwidth
        if self.plot.text.bandwidth:
                        
            bandwidth = (max(self.columns)- min(self.columns))/(len(self.columns)-1)

            text += '\nbandwidth=%s nm' %( bandwidth )
        
        # Add number of samples to text
        if self.plot.text.samples:
            
            text += '\nnspectra=%s; nbands=%s' %( self.spectraDF.shape[0],len(self.columns))
            text += '\nshowing every %s spectra' %( plotskipstep )
              
        yLabel = self.plot.rawaxislabel.x
        
        xLabel = self.plot.rawaxislabel.y
        
        return (xLabel, yLabel, title, text)
      
    def _PilotModeling(self):
        ''' Steer the sequence of processes for modeling spectra data in json format
        '''

        # Get the band data as self.spectraDF
        self._GetBandData()
        
        # Get and add the abundance data
        self._GetAbundanceData()
        
        # Set the regressor models to apply
        self._RegModelSelectSet()
        
        self._ReportRegModelParams()
        
        self.omitL = []
        
        # Any manual feature selection is applied to the original dataframe - i.e. affect all models the same 
        if self.manualFeatureSelection.apply:
            
            self._ManualFeatureSelector()
        
        # The feature selection is applied to the original dataframe - i.e. affect all models the same 
        if self.globalFeatureSelection.apply:
            
            self._VarianceSelector()
            
        # RemoveOutliers is applied to the full dataset and affects all models
        if self.removeOutliers.apply:
                
            self._RemoveOutliers()

        # Loop over the targetfeatures to model 
        for self.targetFeature in self.targetFeatures:
                        
            self._ExtractDataFrame()
            
            self._SetTargetFeatureSymbol()
            
            # Covariate (X) Agglomeration
            if self.featureAgglomeration.apply:
                
                if self.featureAgglomeration.WardClustering.apply:
                    
                    if self.featureAgglomeration.WardClustering.TuneWardClustering.apply:
                        
                        n_clusters = self._TuneWardClustering()
                        
                        if self.verbose:
                            
                            print ('        Tuned Ward clusters:', n_clusters)
    
                    else:
                        
                        n_clusters = self.featureAgglomeration.WardClustering.n_clusters
                         
                    self._WardClustering(n_clusters)
                
            if  self.modelFeatureSelection.apply:
                
                if self.modelFeatureSelection.RFE.apply:
                     
                    self._RFESelector()

                elif self.modelFeatureSelection.univariateSelection.apply:
                     
                    self._UnivariateSelector()
                    
            if self.featureImportance.apply:
                
                self._FeatureImportance()

            if self.modelTests.trainTest.apply:
            
                self._RegrModTrainTest()
        
            if self.modelTests.Kfold.apply:
            
                self._RegrModKFold()
                
def SetupProcesses(docpath, arrangeddatafolder, projFN, jsonpath='json-soillines', createjsonparams=False):
    '''Setup and loop processes
    
    :paramn docpath: path to text file 
    :type: lstr
            
    :param projFN: project filename
    :rtype: str
    
    :param jsonpath: path to directory
    :type: str
            
    '''
    
    if not os.path.exists(docpath):
        
        exitstr = "The docpath does not exists: %s" %(docpath)
        
        exit(exitstr)
          
    dstRootFP = os.path.join(os.path.dirname(__file__),docpath,arrangeddatafolder)
        
    if not os.path.exists(dstRootFP):
        
        exitstr = "The destination path does not exists: %s" %(dstRootFP)
        
        exit(exitstr)
                
    modeljsonFP = os.path.join(dstRootFP,jsonpath)
    
    if not os.path.exists(modeljsonFP):
        
        os.makedirs(modeljsonFP)
    
    if createjsonparams:
        
        flag, jsonFPN = CreateParamJson(modeljsonFP)
        
        if flag:
            
            exitstr = 'ml model json parameter file already exists: %s\n' %(jsonFPN)
        
        else:
        
            exitstr = 'ml model json parameter file created: %s\n' %(jsonFPN)
        
        exitstr += ' Edit the modeling json file for your project and move+rename it to reflect the commands.\n' 
        
        exitstr += ' Add the modeling of the edited file to your project file (%s).\n' %(projFN)
        
        exitstr += ' Then set createjsonparams to False in the main section and rerun script.'
        
        exit(exitstr)
        
    projFPN = os.path.join(dstRootFP,projFN)

    infostr = 'Processing %s' %(projFPN)

    print (infostr)
    
    # Open and read the text file linking to all json files defining the project
    with open(projFPN) as f:

        jsonL = f.readlines()

    # Clean the list of json objects from comments and whithespace etc
    jsonL = [os.path.join(modeljsonFP,x.strip())  for x in jsonL if len(x) > 10 and x[0] != '#']

    #Loop over all json files and create Schemas and Tables
    for jsonObj in jsonL:
        
        print ('    jsonObj:', jsonObj)
        
        paramD = ReadModelJson(jsonObj)
        
        # Invoke the modeling
        mlModel = MachineLearningModel(paramD)
        
        # Add the raw paramD as a variable to mlModel
        mlModel.jsonparamsD = paramD

        # Set the dst file names
        mlModel._SetDstFPNs()

        # run the modeling
        mlModel._PilotModeling()
                                  
if __name__ == '__main__':
    ''' If script is run as stand alone
    '''
        
    docpath = '/Users/thomasgumbricht/docs-local/OSSL/Sweden/LUCAS'
        
    arrangeddatafolder = 'arranged-data'
    
    projFN = 'ml-model_spectra.txt'
    
    jsonpath = 'json-ml-modeling'
    
    createjsonparams=False
    
    SetupProcesses(docpath, arrangeddatafolder, projFN, jsonpath, createjsonparams)