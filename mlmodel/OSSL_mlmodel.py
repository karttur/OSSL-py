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

import pprint

import tempfile

import json

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import randint as sp_randint

import pickle

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline, make_pipeline
from joblib import Memory

from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
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
from sklearn.feature_selection import SelectFromModel

from sklearn.inspection import permutation_importance

# Package application imports

from util.makeObject import Obj

from util.jsonIO import ReadAnyJson, LoadBandData

from util.defaultParams import SoilLineExtractParams, CheckMakeDocPaths, CreateArrangeParamJson, ReadProjectFile

from util.plot import SetTextPos
    
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
        #self.columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]
        self.columnsX = [item for item in self.spectraDF.columns]
        
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
        
        self.featureSymbolSize = 100
        
        if hasattr(self, 'targetFeatureSymbols'):
            
            if hasattr(self.targetFeatureSymbols, self.targetFeature):
                
                symbol = getattr(self.targetFeatureSymbols, self.targetFeature)
                
                if hasattr(symbol, 'color'):
                
                    self.featureSymbolColor = getattr(symbol, 'color')
                    
                if hasattr(symbol, 'marker'):
                
                    self.featureSymbolMarker = getattr(symbol, 'marker')
                    
                if hasattr(symbol, 'size'):
                
                    self.featureSymbolSize = getattr(symbol, 'size')
        
    def _PlotRegr(self, obs, pred, suptitle, title, txtstr,  txtstrHyperParams, regrModel, modeltest):
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
        
        if self.figure.apply:
                    
            fig.savefig(self.imageFPND[self.targetFeature][regrModel][modeltest])
            
    def _RegModelSelectSet(self):
        """ Set the regressors to evaluate
        """
        
        self.regressorModels = []

        if hasattr(self.regressionModels, 'OLS') and self.regressionModels.OLS.apply:
            
            self.regressorModels.append(('OLS', linear_model.LinearRegression(**self.jsonparamsD['regressionModels']['OLS']['hyperParams'])))
            
            self.modelSelectD['OLS'] = []
            
        if hasattr(self.regressionModels, 'TheilSen') and self.regressionModels.TheilSen.apply:
            
            self.regressorModels.append(('TheilSen', linear_model.TheilSenRegressor(**self.jsonparamsD['regressionModels']['OLS']['hyperParams'])))
            
            self.modelSelectD['TheilSen'] = []
            
        if hasattr(self.regressionModels, 'Huber') and self.regressionModels.Huber.apply:
            
            self.regressorModels.append(('Huber', linear_model.HuberRegressor(**self.jsonparamsD['regressionModels']['OLS']['hyperParams'])))
            
            self.modelSelectD['Huber'] = []
            
        if hasattr(self.regressionModels, 'KnnRegr') and self.regressionModels.KnnRegr.apply:
            self.regressorModels.append(('KnnRegr', KNeighborsRegressor( **self.jsonparamsD['regressionModels']['KnnRegr']['hyperParams'])))
            self.modelSelectD['KnnRegr'] = []
            
        if hasattr(self.regressionModels, 'DecTreeRegr') and self.regressionModels.DecTreeRegr.apply:
            self.regressorModels.append(('DecTreeRegr', DecisionTreeRegressor(**self.jsonparamsD['regressionModels']['DecTreeRegr']['hyperParams'])))
            self.modelSelectD['DecTreeRegr'] = []
            
        if hasattr(self.regressionModels, 'SVR') and self.regressionModels.SVR.apply:
            self.regressorModels.append(('SVR', SVR(**self.jsonparamsD['regressionModels']['SVR']['hyperParams'])))
            self.modelSelectD['SVR'] = []
            
        if hasattr(self.regressionModels, 'RandForRegr') and self.regressionModels.RandForRegr.apply:
            self.regressorModels.append(('RandForRegr', RandomForestRegressor( **self.jsonparamsD['regressionModels']['RandForRegr']['hyperParams'])))
            self.modelSelectD['RandForRegr'] = []
            
        if hasattr(self.regressionModels, 'MLP') and self.regressionModels.MLP.apply:
            
            '''
            # First make a pipeline with standardscaler + MLP
            mlp = make_pipeline(
                StandardScaler(),
                MLPRegressor( **self.jsonparamsD['regressionModels']['MLP']['hyperParams'])
            )
            '''
            mlp = Pipeline([('scl', StandardScaler()),
                    ('clf', MLPRegressor( **self.jsonparamsD['regressionModels']['MLP']['hyperParams']) ) ])
            
            # Then add the pipeline as MLP
            self.regressorModels.append(('MLP', mlp))
            
            self.modelSelectD['MLP'] = []
    
    def _RegrModTrainTest(self):
        '''
        '''
       
        #Loop over the defined models
        for m in self.regressorModels:
            #Retrieve the model name and the model itself
            name,model = m
            
            if self.modelSelectD[name]:
                SNULLE
                Xmodel = self.X[self.modelSelectD[name]]
                
                #Split the data into training and test subsets
                X_train, X_test, y_train, y_test = model_selection.train_test_split(Xmodel, self.y, test_size=self.modelTests.trainTest.testSize)

            else:
                
                #Split the data into training and test subsets
                X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=self.modelTests.trainTest.testSize)

            #Fit the model            
            model.fit(X_train, y_train)
            
            #Predict the independent variable in the test subset
            predict = model.predict(X_test)
            
            self.trainTestResultD[self.targetFeature][name] = {'mse':mean_squared_error(y_test, predict),
                                                               'r2': r2_score(y_test, predict),
                                                               'hyperParameterSetting': self.jsonparamsD['regressionModels'][name]['hyperParams'],
                                                               'pickle': self.traintestPickleFPND[self.targetFeature][name]
                                                               }
            
            # Save the complete model with cPickle
            pickle.dump(model, open(self.traintestPickleFPND[self.targetFeature][name],  'wb'))
                       
            if self.verbose:
                
                infoStr =  '                trainTest Model: %s\n' %(name)
                infoStr += '                    hyperParams: %s\n' %(self.jsonparamsD['regressionModels'][name]['hyperParams'])
                infoStr += '                    Mean squared error: %.2f\n' \
                % self.trainTestResultD[self.targetFeature][name]['mse']
                infoStr += '                    Variance (r2) score: %.2f\n' \
                % self.trainTestResultD[self.targetFeature][name]['r2']
            
                print (infoStr)

            if self.modelTests.trainTest.plot:
                txtstr = 'nspectra: %s\n' %(self.X.shape[0])
                txtstr += 'nbands: %s\n' %(self.X.shape[1])
                #txtstr += 'min wl: %s\n' %(self.bandL[0])
                #txtstr += 'max wl: %s\n' %(self.bandL[len(self.bandL)-1])
                #txtstr += 'bands: %s\n' %( ' ,'.join('({0})'.format(w) for w in self.aggBandL)  )
                #txtstr += 'width wl: %s' %(int(self.bandL[1]-self.bandL[0]))
                
                #txtstrHyperParams =  self.HPtuningtxt+'\nHyper Parameters:\n'+'\n'.join([key+': '+str(value) for key, value in self.tunedModD[name].items()])
                suptitle = '%s train/test model (testsize = %s)' %(self.targetFeature, self.modelTests.trainTest.testSize)
                title = ('Model: %(mod)s; RMSE: %(rmse)2f; r2: %(r2)2f' \
                          % {'mod':name,'rmse':mean_squared_error(y_test, predict),'r2': r2_score(y_test, predict)} )
                self._PlotRegr(y_test, predict, suptitle, title, txtstr, '',name, 'traintest')
                
            
    def _RegrModKFold(self):
        """
        """
        
        #Loop over the defined models
        for m in self.regressorModels:
            #Retrieve the model name and the model itself
            name,model = m
            
            
            print (self.modelSelectD[name])
        
            print (self.columns)
            
            '''
            if self.modelSelectD[name]:
            
                Xmodel = self.X[self.modelSelectD[name]]
                
                predict = model_selection.cross_val_predict(model, Xmodel, self.y, cv=self.modelTests.Kfold.folds)

            else:
            '''
            predict = model_selection.cross_val_predict(model, self.X, self.y, cv=self.modelTests.Kfold.folds)
            
            mse = mean_squared_error(self.y, predict)
            
            r2 = r2_score(self.y, predict)
                        
            self.KfoldResultD[self.targetFeature][name] = {'mse': mse,
                                                               'r2': r2,
                                                               'hyperParameterSetting': self.jsonparamsD['regressionModels'][name]['hyperParams'],
                                                               'pickle': self.KfoldPickleFPND[self.targetFeature][name]
                                                               }
            
            # Save the complete model with cPickle
            pickle.dump(model, open(self.KfoldPickleFPND[self.targetFeature][name],  'wb'))
                       
            if self.verbose:
                
                infoStr =  '                Kfold Model: %s\n' %(name)
                infoStr += '                    hyperParams: %s\n' %(self.jsonparamsD['regressionModels'][name]['hyperParams'])
                infoStr += '                    Mean squared error: %.2f\n' \
                % mse
                infoStr += '                    Variance (r2) score: %.2f\n' \
                % r2
            
                print (infoStr)

            if self.modelTests.Kfold.plot:
                txtstr = 'nspectra: %s\n' %(self.X.shape[0])
                txtstr += 'nbands: %s\n' %(self.X.shape[1])
                #txtstr += 'min wl: %s\n' %(self.bandL[0])
                #txtstr += 'max wl: %s\n' %(self.bandL[len(self.bandL)-1])
                #txtstr += 'bands: %s\n' %( ' ,'.join('({0})'.format(w) for w in self.aggBandL)  )
                #txtstr += 'width wl: %s' %(int(self.bandL[1]-self.bandL[0]))
                
                #txtstrHyperParams =  self.HPtuningtxt+'\nHyper Parameters:\n'+'\n'.join([key+': '+str(value) for key, value in self.tunedModD[name].items()])
                suptitle = '%s Kfold model (nfolds = %s)' %(self.targetFeature, self.modelTests.Kfold.folds)
                title = ('Model: %(mod)s; RMSE: %(rmse)2f; r2: %(r2)2f' \
                          % {'mod':name,'rmse':mse,'r2': r2} )
                self._PlotRegr(self.y, predict, suptitle, title, txtstr, '',name, 'Kfold')
                
                
  
                
    def _FeatureImportance(self):
        '''
        '''
       
        #Retrieve the model name and the model itself
        name,model = self.regrModel
        
        print (self.modelSelectD[name])
        
        print (self.columns)
        

        '''
        if self.modelSelectD[name]:
        
            Xmodel = self.X[self.modelSelectD[name]]
            
            #Split the data into training and test subsets
            X_train, X_test, y_train, y_test = model_selection.train_test_split(Xmodel, self.y, test_size=self.modelTests.trainTest.testSize)

        else:
        '''    
        #Split the data into training and test subsets
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=self.modelTests.trainTest.testSize)

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
        
        permImpD = {}
        
        for i in range(len(featureArray)):
            
            permImpD[featureArray[i]] = {'mean_accuracy_decrease': permImportanceArray[i],
                                         'std': errorArray[i]}
            
        self.modelFeatureImportanceD[self.targetFeature][name]['permutationsImportance'] = permImpD

                    
        # Convert to a pandas series
        permImportanceDF = pd.Series(permImportanceArray, index=featureArray)
            
        fig, ax = plt.subplots()
            
        permImportanceDF.plot.bar(yerr=errorArray, color=self.featureSymbolColor, ax=ax)
            
        ax.set_title(title)
            
        ax.set_ylabel("Mean accuracy decrease")
            
        if self.plot.tightLayout:
            
            fig.tight_layout()
                
        plt.show()
                
        if self.figure.apply:
                
            fig.savefig(self.imageFPND[self.targetFeature][name]['featureImportance']['permutationImportance']) 
        
        # Feature importance
        if name in ['OLS','TheilSen','Huber', "Ridge", "ElasticNet", 'logistic', 'SVR']:
            
            if name in ['logistic','SVR']:
            
                importances = model.coef_[0]
                                 
            else:
                
                importances = model.coef_
                                            
            absImportances = abs(importances)
            
            sorted_idx = absImportances.argsort()
            
            importanceArray = importances[sorted_idx][::-1][0:maxFeatures]
            
            featureArray = np.asarray(self.columns)[sorted_idx][::-1][0:maxFeatures]
            
            featImpD = {}
        
            for i in range(len(featureArray)):
            
                featImpD[featureArray[i]] = {'linearCoefficient': importanceArray[i]}
            
            self.modelFeatureImportanceD[self.targetFeature][name]['featureImportance'] = featImpD
            
            # Convert to a pandas series
            linearImportances = pd.Series(importanceArray, index=featureArray)
            
            fig, ax = plt.subplots()
            
            linearImportances.plot.bar(color=self.featureSymbolColor, ax=ax)
            
            title = "Linear feature coefficients\n Feature: %s; Model: %s" %(self.targetFeature, name)

            ax.set_title(title)
            
            ax.set_ylabel("Coefficient")
            
            if self.plot.tightLayout:
            
                fig.tight_layout()
                
            plt.show()
                
            if self.figure.apply:
                
                fig.savefig(self.imageFPND[self.targetFeature][name]['featureImportance']['regressionImportance'])   # save the figure to file
            
            '''
            # summarize feature importance
            for i,v in enumerate(importance):
                print('Feature: %0d, Score: %.5f' % (i,v))
                # plot feature importance
                pyplot.bar([x for x in range(len(importance))], importance)
                pyplot.show()
            '''
        elif name in ['KnnRegr','MLP']:
            ''' These models do not have any feature importance to report
            '''
            pass
        
        else:
            
            featImpD = {}
            
            importances = model.feature_importances_
            
            sorted_idx = importances.argsort()
            
            importanceArray = importances[sorted_idx][::-1][0:maxFeatures]
            
            featureArray = np.asarray(self.columns)[sorted_idx][::-1][0:maxFeatures]
            
            if name in ['RandForRegr']:
            
                std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        
                errorArray = std[sorted_idx][::-1][0:maxFeatures]
        
                for i in range(len(featureArray)):
                    
                    featImpD[featureArray[i]] = {'MDI': importanceArray[i],
                                                 'std': errorArray[i]}
                    
            else:
                
                for i in range(len(featureArray)):
                    
                    featImpD[featureArray[i]] = {'MDI': importanceArray[i]}
                
            self.modelFeatureImportanceD[self.targetFeature][name]['featureImportance'] = featImpD

            # Convert to a pandas series
            forest_importances = pd.Series(importanceArray, index=featureArray)
            
            fig, ax = plt.subplots()
            
            if name in ['RandForRegr']:
                
                forest_importances.plot.bar(yerr=errorArray, color=self.featureSymbolColor, ax=ax)
            
            else:
                
                forest_importances.plot.bar(color=self.featureSymbolColor, ax=ax)
 
            title = "MDI feature importance\n Feature: %s; Model: %s" %(self.targetFeature, name)

            ax.set_title(title)
            
            ax.set_ylabel("Mean decrease in impurity")
            
            if self.plot.tightLayout:
            
                fig.tight_layout()
            
            plt.show()
            
            if self.figure.apply:
                
                fig.savefig(self.imageFPND[self.targetFeature][name]['featureImportance']['regressionImportance'])   # save the figure to file
            
    def _ManualFeatureSelector(self):
        '''
        '''
        print (self.columns)
        # Reset self.columns
        self.columns = self.manualFeatureSelection.spectra
        
        
        print (self.columns)
        # Create the dataframe for the sepctra
        spectraDF = self.spectraDF[ self.columns  ]
        
        self.manualFeatureSelectdRawBands =  self.columns
        # Create any derivative covariates requested
        for b in range(len(self.manualFeatureSelection.derivatives.firstWaveLength)):
            
            bandL = [self.manualFeatureSelection.derivatives.firstWaveLength[b],
                     self.manualFeatureSelection.derivatives.lastWaveLength[b]]
            
            
        self.manualFeatureSelectdDerivates = bandL
            
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
        
        threshold = self.globalFeatureSelection.varianceThreshold.threshold
        
        #istr = 'Selected features:\nvarianceThreshold (%s)'% threshold
        
        #self.selectstrL.append(istr)
        
        # define the list of covariates to use
        #self.columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]
        
        self.columnsX = [item for item in self.spectraDF.columns]
        
        # extract the covariate columns as X
        X = self.spectraDF[self.columnsX]
        
        #Initiate the scaler
        
        if self.globalFeatureSelection.scaler == 'MinMaxScaler': 
        
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
        
            print ('        Selecting features using VarianceThreshold, threhold =',threshold)
        
            print ('            Scaling function MinMaxScaler:')
        
        for sf in range(len(selectedFeatures)):

            if selectedFeatures[sf]:
                self.retainL.append([self.columnsX[sf],select.variances_[sf]])

            else:
                discardL.append([self.columnsX[sf],select.variances_[sf]])
               
        self.globalFeatureSelectedD['method'] = 'varianceThreshold'
        self.globalFeatureSelectedD['threshold'] = self.globalFeatureSelection.varianceThreshold.threshold
        #self.globalFeatureSelectedD['scaler'] = self.globalFeatureSelection.scaler
        self.globalFeatureSelectedD['nCovariatesRemoved'] = len(discardL)
         
        self.varianceSelecttxt = '%s covariates removed with %s' %(len(discardL),'VarianceThreshold')
 
        if self.verbose:
            
            print ('            ',self.varianceSelecttxt)
            
            if self.verbose > 1: 

            
                print ('            ')
                #print the selected features and their variance
                print ('            Discarded features [name, (variance):')
                    
                printL = ['%s (%.3f)'%(i[0],i[1]) for i in discardL]
                            
                for row in printL:
                    print ('                ',row)
                    
                print ('            Retained features [name, (variance)]:')
                
                printretainL = ['%s (%.3f)'%(i[0], i[1]) for i in self.retainL]
                
                for row in printretainL:
                    print ('                ',row)
                        
        self.retainL = [d[0] for d in self.retainL]
        
        
        
        # Reset the covariate dataframe
        self.spectraDF = self.spectraDF[ self.retainL ]
        
    def _UnivariateSelector(self):
        '''
        '''
        nfeatures = self.X.shape[1]
        
        if self.targetFeatureSelection.univariateSelection.SelectKBest.apply:
            
            n_features = self.targetFeatureSelection.univariateSelection.SelectKBest.n_features
   
            if n_features >= nfeatures:
            
                if self.verbose:
                    
                    infostr = '    SelectKBest: Number of features (%s) less than or equal to n_features (%s).' %(nfeatures,n_features)
                
                    print (infostr)
                    
                return 
 
            
            select = SelectKBest(score_func=f_regression, k=n_features)
           
        else:
            
            return
        
        # Select and fit the independent variables, return the selected array
        X = select.fit_transform(self.X, self.y)
        
        self.columns = select.get_feature_names_out()
        # reset the covariates 
        
        self.X = pd.DataFrame(X, columns=self.columns) 
                
        
        self.targetFeatureSelectedD[self.targetFeature]['method'] ='SelectKBest'
        
        self.targetFeatureSelectedD[self.targetFeature]['nFeaturesRemoved'] = nfeatures-self.X.shape[1]
                        
        self.targetFeatureSelectionTxt = '%s features removed with %s' %( nfeatures-self.X.shape[1] ,'SelectKBest')
 
        if self.verbose:
            
            print ('            targetFeatureSelection:')

            print ('                ',self.targetFeatureSelectionTxt)
            
        if self.verbose > 1:

            print ('                Selected features: %s' %(', '.join( select.get_feature_names_out() ) ) )

    def _PermutationSelector(self):
        '''
        '''
        
        nfeatures = self.X.shape[1]
        
        n_features_to_select = self.modelFeatureSelection.RFE.n_features_to_select

        if n_features_to_select >= nfeatures:
            
            if self.verbose:
                
                infostr = '    Number of features (%s) less than or equal to n_features_to_select (%s)' %(nfeatures,n_features_to_select)
            
                print (infostr)
                
            return 
        
        #Retrieve the model name and the model itself
        name,model = self.regrModel
         
        #Split the data into training and test subsets
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=self.modelTests.trainTest.testSize)

        #Fit the model            
        model.fit(X_train, y_train)
                
        permImportance = permutation_importance(model, X_test, y_test)
        
        permImportanceMean = permImportance.importances_mean
                
        sorted_idx = permImportanceMean.argsort()
                
        self.columns = np.asarray(self.columns)[sorted_idx][::-1][0:n_features_to_select]
                       
        self.X = pd.DataFrame(self.X, columns=self.columns) 
        
        ####
        
        self.modelFeatureSelectedD[self.targetFeature][name]['method'] = 'PermutationSelector'
        
        self.modelFeatureSelectedD[self.targetFeature][name]['nFeaturesRemoved'] = nfeatures - self.X.shape[1]
                        
        self.modelFeatureSelectionTxt = '%s features removed with %s' %( nfeatures - self.X.shape[1], 'PermutationSelector')
 
        if self.verbose:
            
            print ('            modelFeatureSeelction:')

            print ('                Regressor: %(m)s; Target feature: %(t)s' %{'m':name,'t':self.targetFeature})

            print ('                ',self.modelFeatureSelectionTxt)
            
        if self.verbose > 1:

            print ('                Selected features: %s' %(', '.join(self.columns)))

        self.modelSelectD[name] = self.columns
        
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
                
                metod = 'RFECV'
            
                print ('\n            RFECV feature selection')
                
            else: 
                
                metod = 'RFE'
                
                print ('\n            RFE feature selection')
                
        #Retrieve the model name and the model itself
        name,model = self.regrModel
        
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
                        
        self.modelFeatureSelectedD[self.targetFeature][name]['method'] = metod
        
        self.modelFeatureSelectedD[self.targetFeature][name]['nFeaturesRemoved'] = len( discardL)
                        
        self.modelFeatureSelectionTxt = '%s features removed with %s' %(len(discardL),'RFE')
 
        if self.verbose:
            
            print ('            modelFeatureSeelction:')

            print ('                Regressor: %(m)s; Target feature: %(t)s' %{'m':name,'t':self.targetFeature})

            print ('                ',self.modelFeatureSelectionTxt)
            
        if self.verbose > 1:

            print ('                Selected features: %s' %(', '.join(selectL)))

        #self.modelSelectD[name] = selectL
    
    def _RemoveOutliers(self):
        """
        """
              
        #self.columnsX = [item for item in self.spectraDF.columns if item not in self.omitL]
        
        self.columnsX = [item for item in self.spectraDF.columns]
        
        # extract the covariate columns as X
        X = self.spectraDF[self.columnsX]
          
        iniSamples = X.shape[0]
        
        if self.removeOutliers.detector.lower() in ['iforest','isolationforest']:
                        
            outlierDetector = IsolationForest(contamination=self.removeOutliers.contamination)
            
        elif self.removeOutliers.detector.lower() in ['ee','eenvelope','ellipticenvelope']:
                        
            outlierDetector = EllipticEnvelope(contamination=self.removeOutliers.contamination)
            
        elif self.removeOutliers.detector.lower() in ['lof','lofactor','localoutlierfactor']:
        
            outlierDetector = LocalOutlierFactor(contamination=self.removeOutliers.contamination)
            
        elif self.removeOutliers.detector.lower() in ['1csvm','1c-svm','oneclasssvm']:
                    
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
                
        self.spectraDF = pd.DataFrame(X) 
        
        postSamples = X.shape[0]
        
        self.nOutliers = iniSamples - postSamples
        
        self.outliersRemovedD['method'] = self.removeOutliers.detector
        self.outliersRemovedD['nOutliersRemoved'] = self.nOutliers
                
        self.outliertxt = '%s outliers removed with %s' %(self.nOutliers,self.removeOutliers.detector)
 
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
                    
        self.agglomeratedFeaturesD['method'] = 'WardClustering'
            
        self.agglomeratedFeaturesD['n_clusters'] = n_clusters
                                    
        self.agglomeratedFeaturesD['tuneWardClusteringApplied'] = self.featureAgglomeration.wardClustering.tuneWardClustering.apply
                    
        self.agglomeratetxt = '%s input features clustered to %s covariates using  %s' %(len(self.columns),len(self.aggColumnL),self.agglomeratedFeaturesD['method'])
 
        if self.verbose:
            
            print ('\n                ',self.agglomeratetxt)
            
           
            if self.verbose > 1:  
                
                print ('                Clusters:')
              
                for x in range(len(self.aggColumnL)):
            
                    print ('                    ',self.aggBandL[x])
          
        # Reset the covariates (self.X)
        X = ward.transform(self.X)
        
        # Reset the main dataframe
        self.spectraDF = pd.DataFrame(X, columns=self.aggColumnL)
        
        # Reset the main column list
        self.columns = self.aggColumnL
        
        # reset the covariates 
        self.X = pd.DataFrame(X, columns=self.aggColumnL)
        
    def _TuneWardClustering(self):
        ''' Determines the optimal nr of cluster
        '''
        nfeatures = self.X.shape[1]
        
        nClustersL = self.featureAgglomeration.wardClustering.tuneWardClustering.clusters
        
        nClustersL = [c for c in nClustersL if c < nfeatures]
        
        kfolds = self.featureAgglomeration.wardClustering.tuneWardClustering.kfolds
        
        cv = KFold(kfolds)  # cross-validation generator for model selection
        
        ridge = BayesianRidge()
        
        cachedir = tempfile.mkdtemp()
        
        mem = Memory(location=cachedir)
        
        ward = FeatureAgglomeration(n_clusters=4, memory=mem)
        
        clf = Pipeline([('ward', ward), ('ridge', ridge)])
        
        # Select the optimal number of parcels with grid search
        clf = GridSearchCV(clf, {'ward__n_clusters': nClustersL}, n_jobs=1, cv=cv)
        
        clf.fit(self.X, self.y)  # set the best parameters
        
        if self.verbose:
            
            print ('            Report for tuning Ward Clustering')
            
        #report the top three results
        self._ReportSearch(clf.cv_results_,3)

        #rerun with the best cluster agglomeration result
        tunedClusters = clf.best_params_['ward__n_clusters']
        
        if self.verbose:
                            
            print ('                Tuned Ward clusters:', tunedClusters)

        return (tunedClusters)
       
    def _RandomtuningParams(self,nFeatures):
        ''' Set hyper parameters for random tuning
        '''
        self.paramDist = {}
        
        self.HPtuningtxt = 'Random tuning'
        
        # specify parameters and distributions to sample from
        name,model = self.regrModel
                
        if name == 'KnnRegr':
            
            self.paramDist[name] = {"n_neighbors": sp_randint(self.hyperParams.RandomTuning.KnnRegr.n_neigbors.min, 
                                                              self.hyperParams.RandomTuning.KnnRegr.n_neigbors.max),
                          'leaf_size': sp_randint(self.hyperParams.RandomTuning.KnnRegr.leaf_size.min, 
                                                              self.hyperParams.RandomTuning.KnnRegr.leaf_size.max),
                          'weights': self.hyperParams.RandomTuning.KnnRegr.weights,
                          'p': self.hyperParams.RandomTuning.KnnRegr.weights,
                          'algorithm': self.hyperParams.RandomTuning.KnnRegr.algorithm}
            
        elif name =='DecTreeRegr':
            # Convert 0 to None for max_depth
            
            max_depth = [m if m > 0 else None for m in self.hyperParams.RandomTuning.DecTreeRegr.max_depth]
            
            self.paramDist[name] = {"max_depth": max_depth,
                        "min_samples_split": sp_randint(self.hyperParams.RandomTuning.DecTreeRegr.min_samples_split.min, 
                                                        self.hyperParams.RandomTuning.DecTreeRegr.min_samples_split.max),
                        "min_samples_leaf": sp_randint(self.hyperParams.RandomTuning.DecTreeRegr.min_samples_leaf.min, 
                                                        self.hyperParams.RandomTuning.DecTreeRegr.min_samples_leaf.max)}
        elif name =='SVR':
            
            self.paramDist[name] = {"kernel": self.hyperParams.RandomTuning.SVR.kernel,
                                    "epsilon": self.hyperParams.RandomTuning.SVR.epsilon,
                                    "C": self.hyperParams.RandomTuning.SVR.epsilon}
            
        elif name =='RandForRegr':
            
            max_depth = [m if m > 0 else None for m in self.hyperParams.RandomTuning.RandForRegr.tuningParams.max_depth]
             
            max_features_max = min(self.hyperParams.RandomTuning.RandForRegr.tuningParams.max_features.max,nFeatures)
            
            max_features_min = min(self.hyperParams.RandomTuning.RandForRegr.tuningParams.max_features.min,nFeatures)
                        
            self.paramDist[name] = {"max_depth": max_depth,
                          "n_estimators": sp_randint(self.hyperParams.RandomTuning.RandForRegr.tuningParams.n_estimators.min, 
                                                              self.hyperParams.RandomTuning.RandForRegr.tuningParams.n_estimators.max),
                          "max_features": sp_randint(max_features_min, 
                                                              max_features_max),
                          "min_samples_split": sp_randint(self.hyperParams.RandomTuning.RandForRegr.tuningParams.min_samples_split.min, 
                                                              self.hyperParams.RandomTuning.RandForRegr.tuningParams.min_samples_split.max),
                          "min_samples_leaf": sp_randint(self.hyperParams.RandomTuning.RandForRegr.tuningParams.min_samples_leaf.min, 
                                                              self.hyperParams.RandomTuning.RandForRegr.tuningParams.min_samples_leaf.max),
                          "bootstrap": self.hyperParams.RandomTuning.RandForRegr.bootstrap}
            
        elif name =='MLP':
                                                  
            self.paramDist[name] = {
                        "hidden_layer_sizes": self.hyperParams.RandomTuning.MLP.hidden_layer_sizes,
                        "solver": self.hyperParams.RandomTuning.MLP.solver,
                        "alpha": sp_randint(self.hyperParams.RandomTuning.MPL.tuningParams.alpha.min, 
                                    self.hyperParams.RandomTuning.MPL.tuningParams.alpha.max),
                        "max_iter": sp_randint(self.hyperParams.RandomTuning.MPL.tuningParams.max_iter.min, 
                                    self.hyperParams.RandomTuning.MPL.tuningParams.max_iter.max)}
      
    def _ExhaustivetuningParams(self,nFeatures):
        '''
        '''
        
        self.HPtuningtxt = 'Exhasutive tuning'
        
        # specify parameters and distributions to sample from
        self.paramGrid = {}
            
        name,model = self.regrModel
        
        if name == 'KnnRegr':
            
            self.paramGrid[name] = [{"n_neighbors": self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.n_neigbors, 
                               'weights': self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.weights,
                               'algorithm': self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.algorithm,
                               'leaf_size': self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.leaf_size,
                               'p': self.hyperParams.ExhaustiveTuning.KnnRegr.tuningParams.p}
                               ]
        elif name =='DecTreeRegr':
            max_depth = [m if m > 0 else None for m in self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.max_depth]

            self.paramGrid[name] = [{
                                "splitter": self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.splitter,
                                "max_depth": self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.max_depth,
                                "min_samples_split": self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.min_samples_split,
                                "min_samples_leaf": self.hyperParams.ExhaustiveTuning.DecTreeRegr.tuningParams.min_samples_leaf}]
        
        elif name =='SVR':            
            self.paramGrid[name] = [{"kernel": self.hyperParams.ExhaustiveTuning.SVR.tuningParams.kernel,
                                "epsilon": self.hyperParams.ExhaustiveTuning.SVR.tuningParams.epsilon,
                                "C": self.hyperParams.ExhaustiveTuning.SVR.tuningParams.C
                              }]
      
        elif name =='RandForRegr':    
            max_depth = [m if m > 0 else None for m in self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.max_depth]

            self.paramGrid[name] = [{
                            "max_depth": max_depth,
                          "n_estimators": self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.n_estimators,
                          "min_samples_split": self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.min_samples_split,
                          "min_samples_leaf": self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.min_samples_leaf,
                          "bootstrap": self.hyperParams.ExhaustiveTuning.RandForRegr.tuningParams.bootstrap}]
    
        elif name =='MLP':    
            self.paramGrid[name] = [{
                        "hidden_layer_sizes": self.hyperParams.ExhaustiveTuning.MLP.tuningParams.hidden_layer_sizes,
                        "solver": self.hyperParams.ExhaustiveTuning.MLP.tuningParams.solver,
                        "alpha": self.hyperParams.ExhaustiveTuning.MLP.tuningParams.alpha,
                        "max_iter": self.hyperParams.ExhaustiveTuning.MLP.tuningParams.max_iter}]
                       
    def _RandomTuning(self):
        '''
        '''
         
        #Retrieve the model name and the model itself
        name,mod = self.regrModel
                
        nFeatures = self.X.shape[1]
        
        # Get the tuning parameters  
        self._RandomtuningParams(nFeatures)
        
        if self.verbose:
            
            print ('\n                HyperParameter random tuning:')
        
            print ('                    ',name, self.paramDist[name])

        search = RandomizedSearchCV(mod, param_distributions=self.paramDist[name],
                                           n_iter=self.params.hyperParameterTuning.nIterSearch)
        
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=(1-self.params.hyperParameterTuning.fraction))
        
        search.fit(X_train, y_train)
        
        resultD = self._ReportSearch(search.cv_results_,self.params.hyperParameterTuning.n_best_report)
        
        self.tunedHyperParamsD[self.targetFeature][name] = resultD
        
        # Set the hyperParameters to the best result 
        for key in resultD[1]['hyperParameters']:
            
            self.jsonparamsD['regressionModels'][name]['hyperParams'][key] = resultD[1]['hyperParameters'][key]
            
    def _ExhaustiveTuning(self):
        '''
        '''
        
        #Retrieve the model name and the model itself
        name,mod = self.regrModel
                
        nFeatures = self.X.shape[1]
        
        # Get the tuning parameters  
        self._ExhaustivetuningParams(nFeatures)
        
        if self.verbose:
            
            print ('\n                HyperParameter exhaustive tuning:')
        
            print ('                    ',name, self.paramGrid[name])
        
        search = GridSearchCV(mod, param_grid=self.paramGrid[name])
        
        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.X, self.y, test_size=(1-self.params.hyperParameterTuning))
        
        search.fit(X_train, y_train)
        
        resultD = self._ReportSearch(search.cv_results_,self.params.hyperParameterTuning.n_best_report)
        
        self.tunedHyperParamsD[self.targetFeature][name] = resultD
               
        # Set the hyperParameters to the best result   
        for key in resultD[1]['hyperParameters']:
            
            self.jsonparamsD['regressionModels'][name]['hyperParams'][key] = resultD[1]['hyperParameters'][key]

            
    def _ReportRegModelParams(self):
        '''
        '''
        
        print ('    Model hyper-parameters:')
        
        for model in self.regressorModels:
            
            #Retrieve the model name and the model itself
            modelname,modelhyperparams = model
            
            print ('        name', modelname, modelhyperparams.get_params()) 

    def _ReportSearch(self, results, n_top=3):
        '''
        '''
        
        resultD = {}
        for i in range(1, n_top + 1):
            
            resultD[i] = {}
            
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            
            for candidate in candidates:
                
                resultD[i]['mean_test_score'] = results['mean_test_score'][candidate]
                
                resultD[i]['std'] = round(results['std_test_score'][candidate],4)
                
                resultD[i]['std'] = round(results['std_test_score'][candidate],4)
                
                resultD[i]['hyperParameters'] = results['params'][candidate]
                
                if self.verbose:
                    
                    print("                    Model with rank: {0}".format(i))
                    
                    print("                    Mean validation score: {0:.3f} (std: {1:.3f})".format(
                          results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
                    
                    print("                    Parameters: {0}".format(results['params'][candidate]))
                    
                    print("") 
                    
        return resultD
                                    
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
        
        self.paramD = paramD
                
        # Set class object default data if required
        self._SetModelDefaults()
        
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
            
            for abundance in sample['abundances']:
                     
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

        self.originalColumns = self.columns
        
    def _SetDstFPNs(self):
        ''' Set destination file paths and names
        '''

        FP,FN = os.path.split(self.input.jsonSpectraDataFilePath)
                
        FN = os.path.splitext(FN)[0]
        
        #self.name = FN.split('_', 1)[1]
        
        
            
        modelFP = os.path.join(FP,'mlmodel')
        
        if not os.path.exists(modelFP):
            
            os.makedirs(modelFP)
            
        projectFP =  os.path.join(modelFP,self.name)
        
        if not os.path.exists(projectFP):
            
            os.makedirs(projectFP)
                              
        modelresultFP = os.path.join(projectFP,'json')
        
        if not os.path.exists(modelresultFP):
            
            os.makedirs(modelresultFP)
            
        pickleFP = os.path.join(projectFP,'pickle')
        
        if not os.path.exists(pickleFP):
            
            os.makedirs(pickleFP)
            
        modelimageFP = os.path.join(projectFP,'images')
        
        if not os.path.exists(modelimageFP):
            
            os.makedirs(modelimageFP)
                 
        regrJsonFN = '%s_results.json' %(self.name)

        self.regrJsonFPN = os.path.join(modelresultFP,regrJsonFN)
        
        paramJsonFN = '%s_params.json' %(self.name)

        self.paramJsonFPN = os.path.join(modelresultFP,paramJsonFN)
        
        self.imageFPND = {}
        
        # the picke files save the regressor models for later use
        self.traintestPickleFPND = {}
        
        self.KfoldPickleFPND = {}
        
        # loop over targetfeatures
        for targetFeature in self.paramD['targetFeatures']:
            
            self.imageFPND[targetFeature] = {}
            
            self.traintestPickleFPND[targetFeature] = {}; self.KfoldPickleFPND[targetFeature] = {}
                               
            for regmodel in self.paramD['regressionModels']:
                
                trainTestPickleFN = '%s_%s_%s_traintest.xsp'    %('modelid',targetFeature, regmodel)
                
                KfoldPickleFN = '%s_%s_%s_Kfold.xsp'    %('modelid',targetFeature, regmodel)

                self.traintestPickleFPND[targetFeature][regmodel] = os.path.join(pickleFP, trainTestPickleFN)
                
                self.KfoldPickleFPND[targetFeature][regmodel] = os.path.join(pickleFP, KfoldPickleFN)
                
                self.imageFPND[targetFeature][regmodel] = {}
                
                if self.featureImportance.apply:
                
                    self.imageFPND[targetFeature][regmodel]['featureImportance'] = {}
                    
                    imgFN = '%s_%s-model_permut-imp.png'    %(targetFeature, regmodel)
                    
                    self.imageFPND[targetFeature][regmodel]['featureImportance']['permutationImportance'] = os.path.join(modelimageFP, imgFN)
                    
                    imgFN = '%s_%s-model_feat-imp.png'    %(targetFeature, regmodel)
                    
                    self.imageFPND[targetFeature][regmodel]['featureImportance']['regressionImportance'] = os.path.join(modelimageFP, imgFN)
                
                if self.modelTests.trainTest.apply:
                    
                    imgFN = '%s_%s-model_tt-result.png'    %(targetFeature, regmodel)
                    
                    self.imageFPND[targetFeature][regmodel]['traintest'] = os.path.join(modelimageFP, imgFN)
                    
                if self.modelTests.Kfold.apply:
                    
                    imgFN = '%s_%s-model_kfold-result.png'    %(targetFeature, regmodel)
                    
                    self.imageFPND[targetFeature][regmodel]['Kfold'] = os.path.join(modelimageFP, imgFN)
    
    def _DumpJson(self):
        '''
        '''
        
        resultD = {}
        
        resultD['originalInputColumns'] = len(self.originalColumns) 
        
        if self.removeOutliers.apply or self.globalFeatureSelection.apply or self.featureAgglomeration.apply:
            
            resultD['globalTweaks']= {}
               
            if self.removeOutliers.apply:
            
                resultD['globalTweaks']['removeOutliers'] = self.outliersRemovedD
                
            if self.globalFeatureSelection.apply:
                                
                resultD['globalTweaks']['globalFeatureSelection'] = self.globalFeatureSelectedD
                
            if self.featureAgglomeration.apply:
            
                resultD['globalTweaks']['featureAgglomeration'] = self.agglomeratedFeaturesD
        
        if self.manualFeatureSelection.apply: 
            
            resultD['manualFeatureSelection'] = True
            
        if self.targetFeatureSelection.apply: 
            
            resultD['targetFeatureSelection'] = self.targetFeatureSelectedD
            
        if self.modelFeatureSelection.apply: 
            
            resultD['modelFeatureSelection'] = self.modelFeatureSelectedD
            
        if self.featureImportance:
            
            resultD['featureImportance'] = self.modelFeatureImportanceD
                    
        if self.hyperParameterTuning.apply:
            
            resultD['hyperParameterTuning'] = {}
            
            if self.hyperParameterTuning.randomTuning.apply:
                
                # Set the results from the hyperParameter Tuning    
                resultD['hyperParameterTuning']['randomTuning'] = self.tunedHyperParamsD
                        
            if self.hyperParameterTuning.exhaustiveTuning.apply: 
                
                # Set the results from the hyperParameter Tuning    
                resultD['hyperParameterTuning']['exhaustiveTuning'] = self.tunedHyperParamsD
          
        # Add the finally selected bands
        
        resultD['appliedModelingFeatures'] = self.finalFeatureLD
         
        # Add the final model results  
        
        if self.modelTests.apply:
            
            resultD['modelResults'] = {}
        
            if self.modelTests.trainTest.apply:
            
                resultD['modelResults']['traintest'] = self.trainTestResultD
                
            if self.modelTests.Kfold.apply:
            
                resultD['modelResults']['Kfold'] = self.KfoldResultD
                
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(resultD)
        
        jsonF = open(self.regrJsonFPN, "w")
  
        json.dump(resultD, jsonF, indent = 2)
        
        jsonF = open(self.paramJsonFPN, "w")
  
        json.dump(self.paramD, jsonF, indent = 2)
                                
    def _PlotSingle(self, X, Y, regr, xlabel, ylabel, title, i):
        ''' Regression and plot for single band
        
        '''
        
        xlabel='%s reflectance' %(xlabel); ylabel='%s reflectance' %(ylabel)
        
        fig, ax = plt.subplots( figsize=(self.plot.figSize.x, self.plot.figSize.y) )
        
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

                       
        fig, ax = plt.subplots( figsize=(self.plot.figSize.x, self.plot.figSize.y)  )
        
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
        title = self.name
    
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
             
        if self.hyperParameterTuning.apply:
                        
            if self.hyperParameterTuning.exhaustiveTuning.apply:
                
                hyperParameterTuning = 'ExhaustiveTuning'
                
                self.tuningParamD = ReadModelJson(self.input.hyperParameterExhaustiveTuning)
                
            elif self.hyperParameterTuning.randomTuning.apply:
                
                hyperParameterTuning = 'RandomTuning'
                
                self.tuningParamD = ReadModelJson(self.input.hyperParameterRandomTuning)
                
            else:
                
                errorStr = 'Hyper parameter tuning requested, but no method assigned'
                
                exit(errorStr)
                
            self.hyperParams = Obj(self.tuningParamD )
                
        # Set the dictionaries to hold the model results
        self.trainTestResultD = {}; self.KfoldResultD  = {}; self.tunedHyperParamsD = {}
        self.globalFeatureSelectedD = {}; self.outliersRemovedD = {}; 
        self.agglomeratedFeaturesD = {}; self.targetFeatureSelectedD = {}
        self.modelFeatureSelectedD = {}; self.modelFeatureImportanceD = {}
        self.finalFeatureLD = {}
        
        # Create the subDicts for all model + target related presults
        for targetFeature in self.targetFeatures:
                
            self.tunedHyperParamsD[targetFeature] = {}; self.trainTestResultD[targetFeature] = {}
            self.KfoldResultD[targetFeature] = {}; self.modelFeatureSelectedD[targetFeature] = {}
            self.targetFeatureSelectedD[targetFeature] = {}; self.modelFeatureImportanceD[targetFeature] = {}
            self.finalFeatureLD[targetFeature] = {}
            
            for regModel in self.jsonparamsD['regressionModels']:
                
                if self.paramD['regressionModels'][regModel]['apply']:

                
                    self.trainTestResultD[targetFeature][regModel] = {}
                    self.KfoldResultD[targetFeature][regModel] = {}
                    self.modelFeatureSelectedD[targetFeature][regModel] = {}
                    self.modelFeatureImportanceD[targetFeature][regModel] = {}
                    self.finalFeatureLD[targetFeature][regModel] = {}

                    if self.paramD['hyperParameterTuning']['apply'] and self.tuningParamD[hyperParameterTuning][regModel]['apply']:
                        
                        self.tunedHyperParamsD[targetFeature][regModel] = {}
                                        
        #self.omitL = []
        
        # RemoveOutliers is applied to the full dataset and affects all models
        if self.removeOutliers.apply:
                
            self._RemoveOutliers()
            
        # Any manual feature selection is applied to the original dataframe - i.e. affect all models the same 
        if self.manualFeatureSelection.apply:
            
            self._ManualFeatureSelector()
        
        # The feature selection is applied to the original dataframe - i.e. affect all models the same 
        if self.globalFeatureSelection.apply:
            
            self._VarianceSelector()
            
        # Loop over the target features to model 
        for self.targetFeature in self.targetFeatures:
            
            if self.verbose:
                
                infoStr = '            Target feature: %s' %(self.targetFeature)
                
                print (infoStr)
                        
            self._ExtractDataFrame()
            
            self._SetTargetFeatureSymbol()
            
            if self.targetFeatureSelection.apply:

                if self.targetFeatureSelection.univariateSelection.apply:
                         
                    self._UnivariateSelector()

            # Covariate (X) Agglomeration
            if self.featureAgglomeration.apply:
                
                if self.featureAgglomeration.wardClustering.apply:
                    
                    if self.featureAgglomeration.wardClustering.tuneWardClustering.apply:
                        
                        n_clusters = self._TuneWardClustering()
                        
                    else:
                        
                        n_clusters = self.featureAgglomeration.wardClustering.n_clusters
                                         
                    self._WardClustering(n_clusters)       
            # End of target feature related selection and clustering
            
            #Loop over the defined models
            for self.regrModel in self.regressorModels:
                
                if  self.modelFeatureSelection.apply:
                    
                    if self.modelFeatureSelection.permutationSelector.apply:
                        
                        self._PermutationSelector()
                    
                    elif self.modelFeatureSelection.RFE.apply:
                        
                        if self.regrModel[0] in ['KnnRegr','MLP']:
                            
                            self._PermutationSelector()
                        
                        else:
                         
                            self._RFESelector()  
                        
                if self.featureImportance.apply:
                    
                    self._FeatureImportance()
                    
                if self.hyperParameterTuning.apply:
                    
                    if self.hyperParameterTuning.exhaustiveTuning.apply:
                        
                        self._ExhaustiveTuning()
                    
                    elif self.hyperParameterTuning.randomTuning.apply:
                        
                        self._RandomTuning()
                        
                    # Reset the regressor with the optimized hyperparameter tuning
                    
                    # Set the regressor models to apply
                    self._RegModelSelectSet()
        
                if self.verbose > 1:
                
                    # Report the regressor model settings (hyper parameters)
                    self._ReportRegModelParams()
                  
                self.finalFeatureLD[self.targetFeature][self.regrModel[0]] = self.columns.tolist()
                
                if self.modelTests.apply:
                          
                    if self.modelTests.trainTest.apply:
                    
                        self._RegrModTrainTest()
                
                    if self.modelTests.Kfold.apply:
                    
                        self._RegrModKFold()
        
        self._DumpJson()
                
def SetupProcesses(docpath, createjsonparams, arrangeddatafolder, projFN, jsonpath):
    '''Setup and loop processes
    
    :paramn docpath: path to text file 
    :type: lstr
            
    :param projFN: project filename
    :rtype: str
    
    :param jsonpath: path to directory
    :type: str
            
    '''
    
    dstRootFP, jsonFP = CheckMakeDocPaths(docpath, arrangeddatafolder, jsonpath)
    
    if createjsonparams:
        
        CreateArrangeParamJson(jsonFP, projFN, 'mlmodel')
        
    jsonProcessObjectL = ReadProjectFile(dstRootFP, projFN, jsonFP)
           
    #Loop over all json files 
    for jsonObj in jsonProcessObjectL:
                
        print ('    jsonObj:', jsonObj)
        
        paramD = ReadModelJson(jsonObj)
        
        # Invoke the modeling
        mlModel = MachineLearningModel(paramD)
        
        mlModel.paramD = paramD
        
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
    #docpath = '/Users/thomasgumbricht/docs-local/OSSL/Europe/LUCAS'
    
    createjsonparams=False
        
    arrangeddatafolder = 'arranged-data'
    
    projFN = 'ml-model_spectra.txt'
    
    jsonpath = 'json-ml-modeling'
        
    SetupProcesses(docpath, createjsonparams , arrangeddatafolder, projFN, jsonpath)