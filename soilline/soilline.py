'''
Created on 23 Sep 2021

Updated 29 Sep 2022

@author: thomasgumbricht

Notes
-----
The module soilline.py:

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
    spectral endMembers for darkSoil and lightSoil. The names of the destination files cannot be set by the 
    user, they are defaulted as follows:
    
    soillines result files:
    
        "rootFP"#visnirjson#visnir_OSSL_"region"_"date"_"first waveLength"-"last waveLength"_"band width"_result-soillines.json
            
    endmember result files:
    
        "rootFP"#visnirjson#visnir_OSSL_"region"_"date"_"first waveLength"-"last waveLength"_"band width"_result-endMembers.json
        
    If requested the script also produced png images showing the raw and/or final soillines:
    
        "rootFP"#visnirjson#visnir_OSSL_"region"_"date"_"first waveLength"-"last waveLength"_"band width"_raw-soillines.png
        "rootFP"#visnirjson#visnir_OSSL_"region"_"date"_"first waveLength"-"last waveLength"_"band width"_final-soillines.png
        
'''

# Standard library imports

import os

import matplotlib.pyplot as plt

import pprint

from copy import deepcopy

from math import sqrt, ceil

# Third party imports

import json

import numpy as np

import pandas as pd

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

# Package application imports

def SoilLineExtractParams():
    ''' Default parameters for soilline extraction from soil spectral library data
    
        :returns: parameter dictionary
        :rtype: dict
    '''
    
    paramD = {}
    
    paramD['verbose'] = 1
    
    paramD['userid'] = "youruserid - any for now"
    
    paramD['importVersion'] = "OSSL-202308"
    
    paramD['campaign'] = {'campaignShortId':'OSSL'}
    
    paramD['campaign']['campaignType'] = 'laboratory'
    
    paramD['campaign']['theme'] = 'soil'
    
    paramD['campaign']['product'] = 'diffuse reflectance'
    
    paramD['campaign']['units'] = 'fraction'
        
    paramD['campaign']['geoRegion'] = "Sweden"
    
    paramD['input'] = {}
    
    paramD['input']['jsonSpectraDataFilePath'] = 'path/to(jsonfile/with/spectraldata.json'
    
    #paramD['input']['xband'] = 620
    
    paramD['input']['yband'] = 1020
        
    paramD['model'] = {}

    paramD['model']['regressor'] = 'OLS'
    
    paramD['model']['soilLineExtend'] = 0.0
    
    paramD['model']['soilLineOriginalRange'] = True
    
    paramD['model']['withinNrStd'] = 1.0
       
    paramD['model']['onlyOutliersAbove'] = False
    
    paramD['plot'] = {}
    
    paramD['plot']['rawSoilLines'] = True
    
    paramD['plot']['finalSoilLines'] = True
            
    paramD['plot']['colorRamp'] = "jet"
    
    paramD['plot']['maxbands'] = 6
    
    paramD['plot']['fiSsize'] = {'x':0,'y':0}
    
    paramD['plot']['legend'] = False
    
    paramD['plot']['tightLayout'] = False
    
    paramD['plot']['scatter'] = {'size':50}
    
    paramD['plot']['endMembers'] = {}
    
    paramD['plot']['endMembers']['lightSoil'] = {'size':200,'color':'lightgrey'}
    
    paramD['plot']['endMembers']['darkSoil'] = {'size':200,'color':'black'}
        
    paramD['plot']['text'] = {'x':0.1,'y':0.9}
    
    paramD['plot']['text']['bandWidth'] = True
    
    paramD['plot']['text']['samples'] = True
    
    paramD['plot']['text']['text'] = ''
    
    paramD['figure'] = {} 
    
    paramD['figure']['rawSoilLines'] = True
    
    paramD['figure']['finalSoilLines'] = True
                
    return (paramD)    
    
def CreateParamJson(docpath):
    """ Create the default json parameters file structure, only to create template if lacking
    
        :param str dstrootFP: directory path 
        
        :param str jsonpath: subfolder under directory path 
    """
    
    # Get the default params
    paramD = SoilLineExtractParams()
    
    # Set the json FPN
    jsonFPN = os.path.join(docpath, 'template_extract_soillines.json')
    
    if os.path.exists(jsonFPN):
        
        return (True, jsonFPN)
    
    # Dump the paramD as a json object   
    jsonF = open(jsonFPN, "w")
  
    json.dump(paramD, jsonF, indent = 2)
  
    jsonF.close()
    
    return (False, jsonFPN)
    
def ReadSoilLineExtractJson(jsonFPN):
    """ Read the parameters for extracting soillines
    
    :param jsonFPN: path to json file
    :type jsonFPN: str
    
    :return paramD: parameters
    :rtype: dict
   """
    
    with open(jsonFPN) as jsonF:
    
        paramD = json.load(jsonF)
        
    return (paramD)
    
class MLRegressors:
    """ Define Machine Learning regressors for soilline extraction

    """
    def __init__(self, regressor):
        ''' Set the selected Machine Learning regressor
        
        :param str regressor: selected regressor
        '''
        
        self.modD = {}
        
        self.modD["OLS"] = {}
        
        self.modD["TheilSen"] = {}
        
        self.modD["Huber"] = {'max_iter':1000, 'alpha':0.0, 'epsilon':1.15}
        
        if regressor == 'OLS':
            self.regressor = linear_model.LinearRegression(**self.modD['OLS'])
            
        elif regressor == 'TheilSen':
            self.regressor = linear_model.TheilSenRegressor(**self.modD['TheilSen'])
            
        elif regressor == 'Huber':
            self.regressor = linear_model.HuberRegressor(**self.modD['Huber'])
            
        else: 
            exitstr = 'EXITING - unrecognized regressor: %s' %(regressor) 
            exit (exitstr)
                    
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
        
        if self.plot.figSize.x == 0:
            
            self.plot.figSize.x = 8
            
        if self.plot.figSize.y == 0:
            
            self.plot.figSize.y = 6
            
        if self.model.soilLineOriginalRange:
            
            self.model.soilLineExtend = 0
            
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
                    
class SoilLine(Obj,MLRegressors):
    ''' Retrieve soilline from soil spectral library data
    '''
    
    def __init__(self,paramD): 
        """ Convert input parameters from nested dict to nested class object
        
            :param dict paramD: parameters 
        """
        
        # convert the input parameter dict to class objects
        Obj.__init__(self,paramD)
                
        # Set class object default data if required
        self._SetDefautls()
        
        # Deep copy parameters to a new object class called params
        self.params = deepcopy(self)
        
        # Drop the plot and figure settings from paramD
        #essentialParamD = {k:v for k,v in paramD.items() if k not in ['plot','figure']}
        paramD.pop('plot'); paramD.pop('figure')
        
        # Deep copy the parameters to self.emD
        self.emD = deepcopy(paramD)
        
        # Deep copy the parameters to self.soillineD
        self.soillineD = deepcopy(paramD)
        
        # Initiate the regressors 
        MLRegressors.__init__(self, self.model.regressor)
        
        if self.verbose:
            
            infoStr = '        Reading spectral data file: %s' %(self.input.jsonSpectraDataFilePath)
            
            print (infoStr)
        
        # Open and load JSON data file
        with open(self.input.jsonSpectraDataFilePath) as jsonF:
            
            self.jsonSpectraData = json.load(jsonF)
                    
        self.emD['content'] = 'soil spectral endMembers'
        
        self.emD['endMembers'] = {}
        
        self.soillineD['raw'] = {}
        
        self.soillineD['final'] = {}
                
        self.xylimD = {}
        
    def _SetColorRamp(self):
        ''' Slice predefined colormap to discrete colors for each band
        '''
        
        # Get the number of bands
        n = len(self.columns)
                
        # Set colormap to use for plotting
        cmap = plt.get_cmap(self.plot.colorRamp)
        
        # Segmenting colormap to the number of bands
        self.slicedCM = cmap(np.linspace(0, 1, n)) 
           
    def _GetBandData(self):
        ''' Read json data into numpy array and convert to pandas dataframe
        '''
        
        substanceEmD = {}
        
        # Loop over data to retrieve all substance abundances
        for spectra in self.jsonSpectraData['spectra']:
                                    
            for item in spectra['abundance']:
                
                for k,v in item.items():
                    
                    if k == 'substance':
                        
                        substance = v
                         
                        if substance not in substanceEmD:
                        
                            substanceEmD[substance] = {'name':substance, 'reference': None, 'abundance':0, 'sampleMean': None}
                    
                    else:
              
                        if v > substanceEmD[substance]['abundance']: 
                            
                            substanceEmD[substance]['reference'] = spectra['id']
                            
                            substanceEmD[substance]['abundance'] = v
                            
                            substanceEmD[substance]['sampleMean'] = spectra['sampleMean']
            
    
        self.substanceEmL = [] 
                                     
        for k in  substanceEmD:
                            
            self.substanceEmL.append(substanceEmD[k])
            
 
                                  
        # Use the waveLength as column headers
        self.columns = self.jsonSpectraData['waveLength']
        
        # Check that the refence band is in the columns available
        
        if not self.input.yband in self.columns:
            
            array = np.asarray(self.columns)
            idx = (np.abs(array - self.input.yband)).argmin()
    
            exitstr = 'The reference y-band %s is not included in the dataset.\n The closest value included is %s\n.' %(self.input.yband,self.columns[idx])
            exitstr = ' Please update the json parameter file and rerun.'
            
            exit(exitstr)
          
        n = 0
                       
        # Loop over the spectra
        for sample in self.jsonSpectraData['spectra']:
                                    
            if n == 0:
            
                spectraA = np.asarray(sample['sampleMean'])
            
            else:
                 
                spectraA = np.vstack( (spectraA, np.asarray(sample['sampleMean']) ) )
            
            n += 1
              
        if spectraA.shape[0] < 30:
            
            exitstr = 'too few values (%s) for extracting soillines - perhaps your extract setting was to tough' %(spectraA.shape[1])
            
            exit (exitstr)
                            
        self.spectraDf = pd.DataFrame(data=spectraA, columns=self.columns)
                     
        if self.model.soilLineOriginalRange:
            
            self.darkref = self.spectraDf[self.input.yband].min()
            
            self.lightref = self.spectraDf[self.input.yband].max()
        
        # With n bands known, create the colorRamp
        self._SetColorRamp()
          
    def _SetDstFPNs(self):
        ''' Set destination file paths and names
        '''

        FP,FN = os.path.split(self.input.jsonSpectraDataFilePath)
                
        FN = os.path.splitext(FN)[0]
        
        self.modelN = FN.split('_', 1)[1]
            
        soillineRootFP = os.path.join(FP,'soillines')
        
        if not os.path.exists(soillineRootFP):
            
            os.makedirs(soillineRootFP)
            
        soillinejsonFP = os.path.join(soillineRootFP,'json')
        
        if not os.path.exists(soillinejsonFP):
            
            os.makedirs(soillinejsonFP)
            
            
        soillineresultFP = os.path.join(soillineRootFP,'models')
        
        if not os.path.exists(soillineresultFP):
            
            os.makedirs(soillineresultFP)
            
        soillineimageFP = os.path.join(soillineRootFP,'images')
        
        if not os.path.exists(soillineimageFP):
            
            os.makedirs(soillineimageFP)
                 
        regrJsonFN = '%s_%s_soillines.json' %(self.modelN,self.model.regressor)

        self.regrJsonFPN = os.path.join(soillineresultFP,regrJsonFN)
        
        endmemberJsonFN = '%s_%s_endMembers.json' %(self.modelN, self.model.regressor)

        self.endmemberJsonFPN = os.path.join(soillineresultFP,endmemberJsonFN)

        endmember4OrthoJsonFN = '%s_%s_ortho-endMembers.json' %(self.modelN, self.model.regressor)

        self.endmember4OrthoJsonFPN = os.path.join(soillineresultFP,endmember4OrthoJsonFN)

        rawModelPngFN = '%s_%s_raw-soillines.png' %(self.modelN,self.model.regressor)

        self.rawModelPngFPN = os.path.join(soillineimageFP, rawModelPngFN)

        finalModelPngFN = '%s_%s_final-soillines.png' %(self.modelN,self.model.regressor)
        
        self.finalModelPngFPN = os.path.join(soillineimageFP, finalModelPngFN)

        self.singleModelPngFPND = {}

                                           
    def _RegressPlotSingle(self, X, Y, regr, xlabel, ylabel, title, i):
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
        
    def _RegressPlotMulti(self, xlabel, ylabel, title, text, plot, figure, pngFPN):
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
        colPlotL = []
        
        plotskipstep = ceil( (len(self.columns)-1)/self.plot.maxbands )
                      
        for i, col in enumerate(self.columns):
            
            if i % plotskipstep == 0:
                
                colPlotL.append(col)   
 
        resultD = {}
                       
        if self.model.soilLineOriginalRange:
            
            darkref = self.darkref
            
            lightref = self.lightref
        
        else:
            # Extend the soil line, if self.soilLineExtend == 0, also delta= 0 and no extension   
            delta = (self.spectraDf[self.input.yband].max()-self.spectraDf[self.input.yband].min())*self.model.soilLineExtend/2
            
            # Set the extended dark reference
            darkref = self.spectraDf[self.input.yband].min()-delta
            
            # Set the extended light reference
            lightref = self.spectraDf[self.input.yband].max()+delta
        
        fig, ax = plt.subplots( figsize=(self.plot.figSize.x, self.plot.figSize.y)  )
        
        if self.plot.tightLayout:
            
            fig.tight_layout()
                    
        Y = self.spectraDf[[self.input.yband]].values
                
        Y = np.ravel(Y)
        
        # Loop over the bands
        for i,xcol in enumerate(self.columns):
                        
            if xcol == self.input.yband:
                
                continue
            
            X = self.spectraDf[[xcol]].values
                                  
            self.regressor.fit(X, Y)
            
            predict = self.regressor.predict(X)
            
            if self.model.regressor == 'OLS':
                '''            
                m =self.regressor.coef_[0][0]
                
                c = self.regressor.intercept_[0]
                '''
                
                m=self.regressor.coef_[0]
                
                c = self.regressor.intercept_
            else:
                
                m=self.regressor.coef_[0]
                
                c = self.regressor.intercept_
                
            r2 = r2_score(Y, predict)
                                    
            rmse = np.sqrt(mean_squared_error(predict, Y))
            
            regression = 'refband[%s] = %.3f*band[%s] + %.3f' %(self.input.yband, m, xcol, c)
                        
            resultD[xcol] = {'m':round(m,3), }
            
            resultD[xcol]['c'] = round(c,3)
            
            resultD[xcol]['r2'] = round(r2,3)
            
            resultD[xcol]['rmse'] = round(rmse,3)
            
            resultD[xcol]['regression'] = regression
            
            # Calculate the light edge from the regression and the reference light
            light=(lightref-self.regressor.intercept_)/self.regressor.coef_[0] 
            
            # Calculate the dark edge from the regression and the reference dark
            dark=(darkref-self.regressor.intercept_)/self.regressor.coef_[0] 
                    
            # Save results in dict
            if self.model.regressor == 'OLS': 
                
                #self.emD['endMembers'][xcol] = {'darkSoil':dark[0], 'lightSoil':light[0], 'rmse':rmse}
                self.emD['endMembers'][xcol] = {'darkSoil':dark, 'lightSoil':light, 'rmse':rmse}
                
            else:
                
                self.emD['endMembers'][xcol] = {'darkSoil':dark, 'lightSoil':light, 'rmse':rmse}
            
            self.emD['model']['nsamples'] = self.spectraDf.shape[0]
              
            if (plot or figure) and self.plot.endMembers.darkSoil.size and xcol in colPlotL:
                
                if self.plot.endMembers.darkSoil.color in ['auto','ramp']:
                    
                    color = self.slicedCM[i]
                    
                else:
                    
                    color = self.plot.endMembers.darkSoil.color
                
                ax.scatter( dark, darkref,color=color, s=self.plot.endMembers.darkSoil.size, edgecolors=self.slicedCM[i])
                
                if self.plot.endMembers.lightSoil.color in ['auto','ramp']:
                    
                    color = self.slicedCM[i]
                    
                else:
                    
                    color = self.plot.endMembers.lightSoil.color
                           
                ax.scatter(light, lightref, color=color,s=self.plot.endMembers.lightSoil.size, edgecolors=self.slicedCM[i])
                
        if plot or figure:
            
            for i,xcol in enumerate(self.columns):
                        
                if xcol == self.input.yband or xcol not in colPlotL:
                    
                    continue
                
                X = self.spectraDf[[xcol]].values
                
                self.regressor.fit(X, Y)
            
                ax.scatter(X, Y, color=self.slicedCM[i])
            
                ax.plot(X, self.regressor.predict(X), color=self.slicedCM[i], label=xcol)
                
            if not self.xylimD:
                
                xmin,xmax = ax.get_xlim()
                
                ymin,ymax = ax.get_ylim()
                
                self.xylimD = {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}
                
            else:
                
                ax.set_xlim(self.xylimD['xmin'], self.xylimD['xmax'])
                
                ax.set_ylim(self.xylimD['ymin'], self.xylimD['ymax'])
            
            ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
            
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
            
        return resultD
              
    def _RegressSoilLine(self, xcol, i, title):
        '''  Linear regression with single band covariate
            
            :param str xcol: dataframe column
            
            :param int i: dataframe column index
            
            :param str title: title
            
            :returns: regression slope, m
            :rtype: float
            
            :returns: regression intercept, c
            :rtype: float
        '''
        # https://stackoverflow.com/questions/29934083/linear-regression-on-pandas-dataframe-using-sklearn-indexerror-tuple-index-ou
       
        X = self.spectraDf[[xcol]].values
        
        Y = self.spectraDf[[self.input.yband]].values
        
        Y = np.ravel(Y)
                
        self.regressor.fit(X, Y)
        
        if self.model.regressor == 'OLS':
                
            #m =self.regressor.coef_[0][0]
                
            #c = self.regressor.intercept_[0]
            
            m=self.regressor.coef_[0]
                
            c = self.regressor.intercept_
            
        else:
                
            m=self.regressor.coef_[0]
                
            c = self.regressor.intercept_
                                                               
        return (m, c)
    
    def _SoilDeviation(self, xcol, i):
        ''' Calculate individual spectra deviation from regressed linear soilline
        
            :param str xcol: dataframe column
            
            :param int i: dataframe column index
            
            :returns: column id for deviation estimates in dataframe, colid
            :rtype: str
        '''
        
        slope, icept =  self._RegressSoilLine(xcol, i, 'before')
        
        X = self.spectraDf[[xcol]].values
        
        Y = self.spectraDf[[self.input.yband]].values
        
        denom = sqrt(slope**2+1)
        
        colid = 'dev%s' %(xcol)
        
        self.spectraDf[colid] = ((slope*X)-Y+icept)/denom
            
        return colid
                
    def _RemoveOutliers(self, colid, devStd):
        """ Remove individual samples with large deviations from estimated soilline
        
            :param str colid: dataframe column with deviation estimates
            
            :param float devStd: standard deviation of the "colid" column
        """
        
        ''' https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
            https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            TODO The direction vis-a-vis the regression line should be explicit I think
        '''
              
        if self.model.onlyOutliersAbove:
            
            # Remove all scatter points larger than 1 negative standard deviation        
            self.spectraDf = self.spectraDf[self.spectraDf[colid] > -devStd*self.model.withinNrStd]
            
        else:
            
            # Remove outliers on both sides
            self.spectraDf = self.spectraDf[abs(self.spectraDf[colid]) < devStd*self.model.withinNrStd]
     
    def translate4OrthoEm(self):
        """ Translate endmbmers to ortho model input format
        """
        
        # Create empty nedmember for Ortho modeling endmember 
        # self.orthoEmLD = []
        
        # assemble the spectral endMembers for draksoil and lightSoil
        
        darkSoilL = []; lightSoilL = []
        
        soilStdL = []; 
        
        for k, v in self.emD["endMembers"].items():
            
            darkSoilL.append(v['darkSoil'])
            
            lightSoilL.append(v['lightSoil'])
            
            soilStdL.append(v['rmse'])
            
        jsonDumpD = {}
        
        jsonDumpD['system'] = self.modelN
        
        jsonDumpD['userid'] = self.params.userid
        
        jsonDumpD['importVersion'] = self.params.importVersion
        
        campaignD = {'campaingid': self.modelN, 
                     'campaignShortId': self.params.campaign.campaignShortId,
                     'campaignType':self.params.campaign.campaignType,
                     'theme': self.params.campaign.theme,
                     'product':self.params.campaign.product,
                     'geoRegion':self.params.campaign.geoRegion,
 
                     }
        
        jsonDumpD['campaign'] = campaignD
        
        jsonDumpD['waveLength'] = self.jsonSpectraData['waveLength']
            
        darkSoilD = {"name":"darkSoil", "reference": "dark end soilline regression", "abundance":-999, "signalMean": darkSoilL, "signalStd": soilStdL}
          
        lightSoilD = {"name":"lightSoil", "reference": "light end soilline regression", "abundance":-999, "signalMean": lightSoilL, "signalStd": soilStdL}

        # Add lightSoilD to the endmemberlist
        self.substanceEmL.insert(0,lightSoilD)
        
        # Add darkSoilD to the endmemberlist
        self.substanceEmL.insert(0,darkSoilD)
        
        jsonDumpD["spectra"] = self.substanceEmL
                
        #jsonDumpD.extend(self.substanceEmL)
        
        return jsonDumpD
                                            
    def _JsonDumpResults(self):
        ''' Export, or dump, the results as json files
        '''
                
        jsonF = open(self.regrJsonFPN, "w")
  
        json.dump(self.soillineD, jsonF, indent = 2)
  
        jsonF.close()
                
        jsonF = open(self.endmemberJsonFPN, "w")
  
        json.dump(self.emD, jsonF, indent = 2)
  
        jsonF.close()
        
        jsonDumpD = self.translate4OrthoEm()
        
        jsonF = open(self.endmember4OrthoJsonFPN, "w")
  
        json.dump(jsonDumpD, jsonF, indent = 2)
        
      
        '''
        jsonF = open(self.paramsJsonFPN, "w")
  
        json.dump(self.paramD, jsonF, indent = 2)
  
        jsonF.close()
        '''
        
    def _PlotTitleTextn(self, titleSuffix):
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
        title = '%s Soil line regressions %s' %(self.model.regressor, titleSuffix)
    
        # set the text
        text = self.plot.text.text
        
        # Add the bandWidth
        if self.plot.text.bandWidth:
                        
            bandWidth = (max(self.columns)- min(self.columns))/(len(self.columns)-1)

            text += '\nbandWidth=%s nm' %( bandWidth )
        
        # Add number of samples to text
        if self.plot.text.samples:
            
            text += '\nnSamples=%s; nBands=%s' %( self.spectraDf.shape[0],len(self.columns))
              
        yLabel = 'Reflectance ref.band (%s)' %(self.input.yband)
        
        xLabel = 'Reflectance other bands'
        
        return (xLabel, yLabel, title, text)
      
    def _PilotExtractSoilline(self):
        ''' Steer the sequence of processes for extracting soillines spectral soil data in json format
        '''

        # Get the band data
        self._GetBandData()
        
        # Set the plot title, labels and annotation
        titleSuffix = '(all data points)'
        
        xLabel, yLabel, title, text = self._PlotTitleTextn(titleSuffix)

        # Regression and plot for raw data
        self.soillineD['raw']['nSamples'] = self.spectraDf.shape[0]
        
        self.soillineD['raw']['soilLines'] = self._RegressPlotMulti( xLabel, yLabel, title, text,
                                                  self.plot.rawSoilLines, self.figure.rawSoilLines, self.rawModelPngFPN )

        # Get the standard deviation for all soil points using raw data
        # To be retained a spectra must pass all the input data
        devStdD = {}
        
        for i,xcol in enumerate(self.columns):
        
            if xcol in [self.input.yband]:
                
                continue
            
            colid = self._SoilDeviation(xcol, i)
            
            devStdD[xcol] = self.spectraDf[colid].std()
                    
        # Remove outliers for all input bands
        # To be retained a spectra must pass all the input data  
        for xcol in devStdD:
                            
            colid = 'dev%s' %(xcol)
            
            self._RemoveOutliers(colid, devStdD[xcol])
                        
        # Set the plot title, labels and annotation
        titleSuffix = '(final soillines)'
        
        xLabel, yLabel, title, text = self._PlotTitleTextn(titleSuffix)
        
        self.soillineD['final']['nSamples'] = self.spectraDf.shape[0]
        
        self.soillineD["final"]['soilLines'] = self._RegressPlotMulti( xLabel, yLabel, title, text,
                                         self.plot.finalSoilLines, self.figure.finalSoilLines, self.finalModelPngFPN )
        
        if self.verbose:
            
            pp = pprint.PrettyPrinter(indent=2)

            pp.pprint(self.soillineD)
            
            pp.pprint(self.emD)
            
        self._JsonDumpResults()
  
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
        
    soillineFP = os.path.join(dstRootFP, 'soillines')
        
    soillinejsonFP = os.path.join(dstRootFP,jsonpath)
    

    if not os.path.exists(soillinejsonFP):
        
        os.makedirs(soillinejsonFP)
    
    if createjsonparams:
        
        flag, jsonFPN = CreateParamJson(soillinejsonFP)
        
        if flag:
            
            exitstr = 'soilline json parameter file already exists: %s\n' %(jsonFPN)
        
        else:
        
            exitstr = 'soilline json parameter file created: %s\n' %(jsonFPN)
        
        exitstr += ' Edit the soilline json file for your project and move+rename it to reflect the commands.\n' 
        
        exitstr += ' Add the path of the edited file to your project file (%s).\n' %(projFN)
        
        exitstr += ' Then set createjsonparams to False in the main section and rerun script.'
        
        exit(exitstr)
        
    projFPN = os.path.join(dstRootFP,projFN)

    infostr = 'Processing %s' %(projFPN)

    print (infostr)
    


    # Open and read the text file linking to all json files defining the project
    with open(projFPN) as f:

        jsonL = f.readlines()

    # Clean the list of json objects from comments and whithespace etc
    jsonL = [os.path.join(soillinejsonFP,x.strip())  for x in jsonL if len(x) > 10 and x[0] != '#']

    #Loop over all json files and create Schemas and Tables
    for jsonObj in jsonL:
        
        print ('    jsonObj:', jsonObj)
        
        paramD = ReadSoilLineExtractJson(jsonObj)
        
        # Invoke the soil line
        sl = SoilLine(paramD)
        
        # Set the dst file names
        sl._SetDstFPNs()
        
        # run the soilline extractor
        sl._PilotExtractSoilline()
                                  
if __name__ == '__main__':
    ''' If script is run as stand alone
    '''
    
    '''   
    docpath = '/Users/thomasgumbricht/OSSL/se/OSSL_se-oster+vaster-goetland_20220907/soillines'
    
    projFN = 'extract_soillines.txt'
    
    jsonpath = ''
    
    SetupProcesses(docpath, projFN, jsonpath)
    '''
    
    docpath = '/Users/thomasgumbricht/docs-local/OSSL/Sweden/LUCAS'
        
    arrangeddatafolder = 'arranged-data'
    
    projFN = 'extract_soillines.txt'
    
    jsonpath = 'json-soillines'
    
    createjsonparams=False
    
    SetupProcesses(docpath, arrangeddatafolder, projFN, jsonpath, createjsonparams)

