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
    
    paramD['input'] = {}
    
    paramD['campaign'] = {'id':'OSSL'}
    
    paramD['campaign']['kind'] = 'laboratory'
    
    paramD['campaign']['instrument'] = 'unknown VIS-NIR lab spectrometer'
    
    paramD['campaign']['dataprep'] = 'Drying and sieving'
    
    paramD['campaign']['fielddate'] = '2010-10-10'
    
    paramD['campaign']['spectradate'] = '2010-10-12'
    
    paramD['campaign']['region'] = {'name':'South central Sweden',
                                    'east':16,
                                    'west':23,
                                    'south':54,
                                    'north':58}
    
    paramD['soilsample'] = {'kind':'NA',
                          'landuse':'NA',
                          'dominatparticle': 'NA',
                          'horizon':'NA',
                          'mindepth':0,
                          'maxdepth':20}
    
    paramD['input']['jsonFPN'] = '/Users/thomasgumbricht/OSSL/se/OSSL_se-oster+vaster-goetland_20220907/OSSL_se-oster+vaster-goetland_20220907_600-1040_40.json'
    
    paramD['input']['units'] = 'percent'
    
    #paramD['input']['xband'] = 620
    
    paramD['input']['yband'] = 1020
        
    paramD['model'] = {}

    paramD['model']['regressor'] = 'OLS'
    
    paramD['model']['soillineExtend'] = 0.2
    
    paramD['model']['soillineOriginalRange'] = True
    
    paramD['model']['withinNrStd'] = 1.0
    
    paramD['model']['stdAccept'] = 1.0
    
    paramD['model']['onlyOutliersAbove'] = False
    
    paramD['plot'] = {}
    
    paramD['plot']['rawsoillines'] = True
    
    paramD['plot']['finalsoillines'] = True
        
    paramD['plot']['intermediatelsollines'] = False
    
    paramD['plot']['colorramp'] = "jet"
    
    paramD['plot']['maxbands'] = 6
    
    paramD['plot']['figsize'] = {'x':0,'y':0}
    
    paramD['plot']['legend'] = False
    
    paramD['plot']['tight_layout'] = False
    
    paramD['plot']['scatter'] = {'size':50}
    
    paramD['plot']['endmembers'] = {}
    
    paramD['plot']['endmembers']['lightsoil'] = {'size':200,'color':'lightgrey'}
    
    paramD['plot']['endmembers']['darksoil'] = {'size':200,'color':'black'}
        
    paramD['plot']['text'] = {'x':0.1,'y':0.9}
    
    paramD['plot']['text']['bandwidth'] = True
    
    paramD['plot']['text']['samples'] = True
    
    paramD['plot']['text']['text'] = ''
    
    paramD['figure'] = {} 
    
    paramD['figure']['rawsoillines'] = True
    
    paramD['figure']['finalsoillines'] = True
    
    paramD['figure']['intermediatelsollines'] = False
    
    paramD['xylim'] = {}
    
    paramD['xylim']['xmin'] = 15
    
    paramD['xylim']['xmax'] = 80
    
    paramD['xylim']['ymin'] = 45
    
    paramD['xylim']['ymax'] = 80
        
    return (paramD)

def CreateParamJson(docpath):
    """ Create the default json parameters file structure, only to create template if lacking
    
        :param str docpath: directory path   
    """
    
    # Get the default params
    paramD = SoilLineExtractParams()
    
    # Set the json FPN
    jsonFPN = os.path.join(docpath, 'extract_soillines.json')
    
    # Dump the paramD as a json object
    
    jsonF = open(jsonFPN, "w")
  
    json.dump(paramD, jsonF, indent = 2)
  
    jsonF.close()
    
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
        
        if self.plot.figsize.x == 0:
            
            self.plot.figsize.x = 8
            
        if self.plot.figsize.y == 0:
            
            self.plot.figsize.y = 6
            
        if self.model.soillineOriginalRange:
            
            self.model.soillineExtend = 0
            
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
        
        # Open and load JSON data file
        with open(self.input.jsonFPN) as jsonF:
            
            self.jsonData = json.load(jsonF)
                    
        self.emD['content'] = 'soil spectral endmembers'
        
        self.emD['endmembers'] = {}
        
        self.soillineD['raw'] = {}
        
        self.soillineD['final'] = {}
                
        self.xylimD = {}
        
    def _SetColorRamp(self):
        ''' Slice predefined colormap to discrete colors for each band
        '''
        
        # Get the number of bands
        n = len(self.columns)
                
        # Set colormap to use for plotting
        cmap = plt.get_cmap(self.plot.colorramp)
        
        # Segmenting colormap to the number of bands
        self.slicedCM = cmap(np.linspace(0, 1, n)) 
           
    def _GetBandData(self):
        ''' Read json data into numpy array and convert to pandas dataframe
        '''
        
        n = 0
        
        
        substanceEmD = {}
        
        # Loop over data to retrieve all substance abundances
        for spectra in self.jsonData['labspectra']:
                        
            for item in spectra['abundance']:
                
                for k,v in item.items():
                    
                    if k == 'substance':
                        
                        substance = v
                         
                        if substance not in substanceEmD:
                        
                            substanceEmD[substance] = {'name':substance, 'reference': None, 'abundance':0, 'nsamples':1, 'mean': None}
                    
                    else:
                        
                        substanceEmD[substance]['nsamples'] += 1
                        
                        if v > substanceEmD[substance]['abundance']: 
                            
                            substanceEmD[substance]['reference'] = spectra['name']
                            
                            substanceEmD[substance]['abundance'] = v
                            
                            substanceEmD[substance]['mean'] = spectra['mean']
             
        self.substanceEmL = [] 
                                     
        for k in  substanceEmD:
                            
            self.substanceEmL.append(substanceEmD[k])
                                  
        # Use the bands as column headers
        self.columns = self.jsonData['bands']
                         
        # Loop over the spectra
        for sample in self.jsonData['labspectra']:
                                    
            if n == 0:
            
                spectraA = np.asarray(sample['mean'])
            
            else:
                 
                spectraA = np.vstack( (spectraA, np.asarray(sample['mean']) ) )
            
            n += 1
              
        if spectraA.shape[0] < 30:
            
            exitstr = 'too few values (%s) for extracting soillines - perhaps your extract setting was to tough' %(spectraA.shape[1])
            
            exit (exitstr)
                            
        self.spectraDf = pd.DataFrame(data=spectraA, columns=self.columns)
                
        if self.model.soillineOriginalRange:
            
            self.darkref = self.spectraDf[self.input.yband].min()
            
            self.lightref = self.spectraDf[self.input.yband].max()
        
        # With n bands known, create the colorRamp
        self._SetColorRamp()
          
    def _SetDstFPNs(self):
        ''' Set destination file paths and names
        '''
                
        FP, srcFN = os.path.split(self.input.jsonFPN)
        
        FP = os.path.join(FP,'soillines')
        
        if not os.path.exists(FP):
            
            os.makedirs(FP)
            
        #paramsJsonFN = '%s_%s_params.json' %(os.path.splitext(srcFN)[0],self.model.regressor)
        
        #self.paramsJsonFPN = os.path.join(FP,paramsJsonFN)
        
        regrJsonFN = '%s_%s_result-soillines.json' %(os.path.splitext(srcFN)[0],self.model.regressor)
        
        self.regrJsonFPN = os.path.join(FP,regrJsonFN)
                
        endmemberJsonFN = '%s_%s_result-endmembers.json' %(os.path.splitext(srcFN)[0], self.model.regressor)
        
        self.endmemberJsonFN = os.path.join(FP,endmemberJsonFN)
        
        endmember4OrthoJsonFN = '%s_%s_ortho-endmembers.json' %(os.path.splitext(srcFN)[0], self.model.regressor)
        
        self.endmember4OrthoJsonFPN = os.path.join(FP,endmember4OrthoJsonFN)
        
        rawModelPngFN = '%s_%s_raw-soillines.png' %(os.path.splitext(srcFN)[0],self.model.regressor)
        
        self.rawModelPngFPN = os.path.join(FP, rawModelPngFN)
        
        finalModelPngFN = '%s_%s_final-soillines.png' %(os.path.splitext(srcFN)[0],self.model.regressor)
                
        self.finalModelPngFPN = os.path.join(FP, finalModelPngFN)
        
        self.singleModelPngFPND = {}
        
        for c in self.columns:
            
            singleModelPngFN = '%s_%s_%s_soilline.png' %(os.path.splitext(srcFN)[0], self.model.regressor, c)
            
            singleModelPngFN = '%s_%s_%s_soilline.png' %(os.path.splitext(srcFN)[0], self.model.regressor, c)
            
            self.singleModelPngFPND[c] = os.path.join(FP, singleModelPngFN)
                                           
    def _RegressPlotSingle(self, X, Y, regr, xlabel, ylabel, title, i):
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
                       
        if self.model.soillineOriginalRange:
            
            darkref = self.darkref
            
            lightref = self.lightref
        
        else:
            # Extend the soil line, if self.soillineExtend == 0, also delta= 0 and no extension   
            delta = (self.spectraDf[self.yband].max()-self.spectraDf[self.yband].min())*self.soillineExtend/2
            
            # Set the extended dark reference
            darkref = self.spectraDf[self.yband].min()-delta
            
            # Set the extended light reference
            lightref = self.spectraDf[self.yband].max()+delta
        
        fig, ax = plt.subplots( figsize=(self.plot.figsize.x, self.plot.figsize.y)  )
        
        if self.plot.tight_layout:
            
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
                
                m =self.regressor.coef_[0][0]
                
                c = self.regressor.intercept_[0]
            
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
                
                self.emD['endmembers'][xcol] = {'darksoil':dark[0], 'lightsoil':light[0], 'rmse':rmse}
                
            else:
                
                self.emD['endmembers'][xcol] = {'darksoil':dark, 'lightsoil':light, 'rmse':rmse}
            
            self.emD['model']['nsamples'] = self.spectraDf.shape[0]
              
            if (plot or figure) and self.plot.endmembers.darksoil.size and xcol in colPlotL:
                
                if self.plot.endmembers.darksoil.color in ['auto','ramp']:
                    
                    color = self.slicedCM[i]
                    
                else:
                    
                    color = self.plot.endmembers.darksoil.color
                
                ax.scatter( dark, darkref,color=color, s=self.plot.endmembers.darksoil.size)
                
                if self.plot.endmembers.lightsoil.color in ['auto','ramp']:
                    
                    color = self.slicedCM[i]
                    
                else:
                    
                    color = self.plot.endmembers.lightsoil.color
                           
                ax.scatter(light, lightref, color=color,s=self.plot.endmembers.lightsoil.size)
                
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
                
            m =self.regressor.coef_[0][0]
                
            c = self.regressor.intercept_[0]
            
        else:
                
            m=self.regressor.coef_[0]
                
            c = self.regressor.intercept_
                                                
        if self.plot.intermediatelsollines:
            
            self._RegressPlotSingle(X,Y,self.regressor,xcol,title,i)
               
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
            
            # Remove utliers on both sides
            self.spectraDf = self.spectraDf[abs(self.spectraDf[colid]) < devStd*self.model.withinNrStd]
     
    def translate4OrthoEm(self):
        """ Translate endmbmers to ortho model input format
        """
        
        # Create empty nedmember for Ortho modeling endmember 
        self.orthoEmLD = []
        
        # assemble the spectral endmembers for draksoil and lightsoil
        
        darkSoilL = []; lightSoilL = []
        
        soilStdL = []; 
        
        for k, v in self.emD["endmembers"].items():
            
            darkSoilL.append(v['darksoil'])
            
            lightSoilL.append(v['lightsoil'])
            
            soilStdL.append(v['rmse'])
            
                    
        darksoilD = {"name":"darksoil", "mean": darkSoilL, "std": soilStdL, "reference": "dark end soilline regression", "abundance":-999}
          
        lightsoilD = {"name":"lightsoil", "mean": lightSoilL, "std": soilStdL, "reference": "light end soilline regression", "abundance":-999}

        jsonDumpD = [darksoilD, lightsoilD]
        
        jsonDumpD.extend(self.substanceEmL)
        
        return jsonDumpD
                                            
    def _JsonDumpResults(self):
        ''' Export, or dump, the results as json files
        '''
                
        jsonF = open(self.regrJsonFPN, "w")
  
        json.dump(self.soillineD, jsonF, indent = 2)
  
        jsonF.close()
                
        jsonF = open(self.endmemberJsonFN, "w")
  
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
        
        # Add the bandwidth
        if self.plot.text.bandwidth:
                        
            bandwidth = (max(self.columns)- min(self.columns))/(len(self.columns)-1)

            text += '\nbandwidth=%s nm' %( bandwidth )
        
        # Add number of samples to text
        if self.plot.text.samples:
            
            text += '\nnsamples=%s; nbands=%s' %( self.spectraDf.shape[0],len(self.columns))
              
        yLabel = 'Reflectance ref.band (%s)' %(self.input.yband)
        
        xLabel = 'Reflectance other bands'
        
        return (xLabel, yLabel, title, text)
      
    def _PilotExtractSoilline(self):
        ''' Steer the sequence of processes for extracting soillines spectral soil data in json format
        '''

        # Get the band data
        self._GetBandData()
        
        # Set the dst file names
        self._SetDstFPNs()

        # Set the plot title, labels and annotation
        titleSuffix = '(all data points)'
        
        xLabel, yLabel, title, text = self._PlotTitleTextn(titleSuffix)

        # Regression and plot for raw data
        self.soillineD['raw']['nsamples'] = self.spectraDf.shape[0]
        
        self.soillineD['raw']['soillines'] = self._RegressPlotMulti( xLabel, yLabel, title, text,
                                                  self.plot.rawsoillines, self.figure.rawsoillines, self.rawModelPngFPN )

        # Get the standard deviation for all soil points using raw data
        # To be retained a spectra must pass all the input data
        devStdD = {}
        
        for i,xcol in enumerate(self.columns):
        
            if xcol in [self.input.yband]:
                
                continue
            
            colid = self._SoilDeviation(xcol, i)
            
            devStdD[xcol] = max( self.model.stdAccept, self.spectraDf[colid].std() )
                    
        # Remove outliers for all input bands
        # To be retained a spectra must pass all the input data  
        for xcol in devStdD:
                            
            colid = 'dev%s' %(xcol)
            
            self._RemoveOutliers(colid, devStdD[xcol])
            
            if self.plot.intermediatelsollines:
            
                self._RegressSoilLine(xcol, i, 'after')
            
        # Set the plot title, labels and annotation
        titleSuffix = '(final soillines)'
        
        xLabel, yLabel, title, text = self._PlotTitleTextn(titleSuffix)
        
        self.soillineD['final']['nsamples'] = self.spectraDf.shape[0]
        
        self.soillineD["final"]['soillines'] = self._RegressPlotMulti( xLabel, yLabel, title, text,
                                         self.plot.finalsoillines, self.figure.finalsoillines, self.finalModelPngFPN )
        
        if self.verbose:
            
            pp = pprint.PrettyPrinter(indent=2)

            pp.pprint(self.soillineD)
            
            pp.pprint(self.emD)
            
        self._JsonDumpResults()
  
def SetupProcesses(docpath, projFN, jsonpath):
    '''Setup and loop processes
    
    :paramn docpath: path to text file 
    :type: lstr
            
    :param projFN: project filename
    :rtype: str
    
    :param jsonpath: path to directory
    :type: str
            
    '''
    
    srcFP = os.path.join(os.path.dirname(__file__),docpath)

    projFPN = os.path.join(srcFP,projFN)

    # Get the full path to the project text file
    dirPath = os.path.split(projFPN)[0]

    if not os.path.exists(projFPN):

        exitstr = 'EXITING, project file missing: %s' %(projFPN)

        exit( exitstr )

    infostr = 'Processing %s' %(projFPN)

    print (infostr)
    
    if jsonpath != "":
            
        dirPath = os.path.join(dirPath,jsonpath)

    # Open and read the text file linking to all json files defining the project
    with open(projFPN) as f:

        jsonL = f.readlines()

    # Clean the list of json objects from comments and whithespace etc
    jsonL = [os.path.join(dirPath,x.strip())  for x in jsonL if len(x) > 10 and x[0] != '#']

    #Loop over all json files and create Schemas and Tables
    for jsonObj in jsonL:
        
        print ('jsonObj:', jsonObj)
        
        paramD = ReadSoilLineExtractJson(jsonObj)
        
        # Invoke the soil line
        sl = SoilLine(paramD)
        
        # run the soilline extractor
        sl._PilotExtractSoilline()
                                  
if __name__ == '__main__':
    ''' If script is run as stand alone
    '''
       
    docpath = '/Users/thomasgumbricht/OSSL/se/OSSL_se-oster+vaster-goetland_20220907/soillines'
    
    projFN = 'extract_soillines.txt'
    
    jsonpath = ''
    
    SetupProcesses(docpath, projFN, jsonpath)
