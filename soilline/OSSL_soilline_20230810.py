'''
Created on 23 Sep 2021

Updated 29 Sep 2022

Last update 7 August 2023

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

from util.makeObject import Obj

from util.jsonIO import ReadAnyJson, LoadBandData

from util.defaultParams import SoilLineExtractParams, CheckMakeDocPaths, CreateArrangeParamJson, ReadProjectFile

from util.plot import SetTextPos

      
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
    
    return ReadAnyJson(jsonFPN)
    
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
                         
class SoilLine(Obj,MLRegressors):
    ''' Retrieve soilline from soil spectral library data
    '''
    
    def __init__(self,paramD): 
        """ Convert input parameters from nested dict to nested class object
        
            :param dict paramD: parameters 
        """
        
        self.paramD = paramD
        
        # convert the input parameter dict to class objects
        Obj.__init__(self,paramD)
                
        # Set class object default data if required
        self._SetSoilLineDefautls()
        
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
                        
                            substanceEmD[substance] = {'name':substance, 'reference': None, 'abundance':0, 'signalMean': None}
                    
                    else:
              
                        if v > substanceEmD[substance]['abundance']: 
                            
                            substanceEmD[substance]['reference'] = spectra['id']
                            
                            substanceEmD[substance]['abundance'] = v
                            
                            substanceEmD[substance]['signalMean'] = spectra['signalMean']
            
    
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
            
                spectraA = np.asarray(sample['signalMean'])
            
            else:
                 
                spectraA = np.vstack( (spectraA, np.asarray(sample['signalMean']) ) )
            
            n += 1
              
        if spectraA.shape[0] < 30:
            
            exitstr = 'too few values (%s) for extracting soillines - perhaps your extract setting was to tough' %(spectraA.shape[1])
            
            exit (exitstr)
                            
        self.spectraDF = pd.DataFrame(data=spectraA, columns=self.columns)
                     
        #if self.model.soilLineOriginalRange:
            
        # Get the input minimum and maximum for each band
        #self.darkRef = self.spectraDF[self.input.yband].min()
            
        self.darkRefDF = self.spectraDF.min(axis=0, skipna=True)

        #self.lightRef = self.spectraDF[self.input.yband].max()
            
        self.lightRefDF = self.spectraDF.max(axis=0, skipna=True)
        
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
                           
        soillineresultFP = os.path.join(soillineRootFP,'json')
        
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
        
    def _OriginalMinMaxSoilLines(self, xLabel, yLabel, text):
        ''' Soil lines drawn only from min and max reflectance in each band
        '''
          
        title = 'Soil lines from original spectra min and max\nProject: %s' %(self.params.id)
        
            
        # Convert the pandas series on darkref and lightref to a dataframe 
        self.endPointDf = pd.concat([self.darkRefDF, self.lightRefDF], axis=1)
           
        self.endPointDf.columns = ['x0', 'x1']
         
        # y0L = list for reference band y values at eeach other bands min and max values
        y0L = []; y1L = []
        
        # From the band end points get the corresponding reference (y) band value
        for band, signal in self.darkRefDF.items():

            rowDF = self.spectraDF.loc[self.spectraDF[band] == signal]
            
            y0L.append( rowDF[self.input.yband].values[0] )
            
                        
        for band, signal in self.lightRefDF.items():
                        
            rowDF = self.spectraDF.loc[self.spectraDF[band] == signal]
            
            y1L.append( rowDF[self.input.yband].values[0] )
            
        # Convert the reference y-band value end poists to a dataframe
        
        self.endPointDf.insert(2, "y0", y0L)
        
        self.endPointDf.insert(3, "y1", y1L)
                
        fig, ax = plt.subplots( figsize=(self.plot.figSize.x, self.plot.figSize.y)  )
          
        i = 0
            
        # Loop all spectra and plot soil lines 
        for index, row in self.endPointDf.iterrows():
    
            X = [row['x0'], row['x1']]
            
            Y = [row['y0'], row['y1']]
        
            ax.plot(X, Y, color = self.slicedCM[i])
            
            i += 1
        
        i = 0
            
        # Loop all spectra and plot dark and light edges     
        for index, row in self.endPointDf.iterrows():
    
            X = [row['x0'], row['x1']]
            
            Y = [row['y0'], row['y1']]
                              
            ax.scatter( X[0], Y[0],color = self.plot.endMembers.darkSoil.color, 
                        edgecolors=self.slicedCM[i], s=self.plot.endMembers.darkSoil.size/3)
            
            ax.scatter( X[1], Y[1],color = self.plot.endMembers.lightSoil.color, 
                        edgecolors=self.slicedCM[i], s=self.plot.endMembers.darkSoil.size/3)
     
            i += 1
            
            
        if not self.xylimD:
                
            xmin,xmax = ax.get_xlim()
            
            ymin,ymax = ax.get_ylim()
            
            self.xylimD = {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}
            
        else:
            
            ax.set_xlim(self.xylimD['xmin'], self.xylimD['xmax'])
            
            ax.set_ylim(self.xylimD['ymin'], self.xylimD['ymax'])
            
        ax.set(xlabel=xLabel, ylabel=yLabel, title=title)
        
        x,y = SetTextPos(self.plot.text.x, self.plot.text.y, self.xylimD['xmin'], self.xylimD['xmax'], self.xylimD['ymin'], self.xylimD['ymax'])
                
        ax.text(x, y, text)
            
        if self.plot.tightLayout:
                
            fig.tight_layout()
            
        plt.show()
        
    def _ExtractSoilLines(self, xlabel, ylabel, title, text, plot, figure, pngFPN):
        '''
        '''
        
        def EstimateSoilLine(X,Y,xcol):
            '''
            '''
            self.regressor.fit(X, Y)
            
            predict = self.regressor.predict(X)
            
            if isinstance(self.regressor.coef_[0],np.ndarray):
               
                m=self.regressor.coef_[0][0]
                    
                c = self.regressor.intercept_[0]
                
            else:
                
                m=self.regressor.coef_[0]
                    
                c = self.regressor.intercept_
   
            r2 = r2_score(Y, predict)
                                    
            rmse = np.sqrt(mean_squared_error(predict, Y))
            
            regression = 'refband[%s] = %.3f*band[%s] + %.3f' %(self.input.yband, m, xcol, c)
            
            if xcol != self.input.yband:
                        
                resultD[xcol] = {'m':round(m,3), }
                
                resultD[xcol]['c'] = round(c,3)
                
                resultD[xcol]['r2'] = round(r2,3)
                
                resultD[xcol]['rmse'] = round(rmse,3)
                
                resultD[xcol]['regression'] = regression
                
            if self.model.soilLineendMembersGlobalEdge:
                
                # Calculate the light edge from the regression and the reference light
                light=(self.lightRef-self.regressor.intercept_)/self.regressor.coef_[0] 
                
                # Calculate the dark edge from the regression and the reference dark
                dark=(self.darkRef-self.regressor.intercept_)/self.regressor.coef_[0] 
       
            else:
                
                lightref = self.endPointDf.loc[xcol]['y1']
                
                darkref = self.endPointDf.loc[xcol]['y0']
                
                # Calculate the light edge from the regression and the reference light
                light=(lightref-self.regressor.intercept_)/self.regressor.coef_[0] 
                
                # Calculate the dark edge from the regression and the reference dark
                dark=(darkref-self.regressor.intercept_)/self.regressor.coef_[0]
                
            if isinstance(light,np.ndarray):
                    
                light = light[0]
                    
            if isinstance(dark,np.ndarray):
                    
                dark = dark[0]
                    
            return dark, light, rmse               
            
        ''' _ExtractSoilLines
        '''
       
        resultD = {}
        
        colPlotL = []
        
        plotskipstep = ceil( (len(self.columns)-1)/self.plot.maxBands )
                      
        for i, col in enumerate(self.columns):
            
            if i % plotskipstep == 0:
                
                colPlotL.append(col) 
                
        fig, ax = plt.subplots( figsize=(self.plot.figSize.x, self.plot.figSize.y)  )
        
        if self.plot.tightLayout:
            
            fig.tight_layout()
       
        # Extact the reference band values
        Y = self.spectraDF[[self.input.yband]].values
         
        # Flatten the reference values to single vector       
        Y = np.ravel(Y)
                
        # Loop over the bands
        for i,xcol in enumerate(self.columns):
            
            X = self.spectraDF[[xcol]].values
            
            # Flatten the band values to single vector
            #X = np.ravel(X)
                                  
            dark,light,rmse = EstimateSoilLine(X,Y,xcol)
            
            self.emD['endMembers'][xcol] = {'darkSoil':dark, 'lightSoil':light, 'rmse':rmse}
            
            if self.plot.endMembers.darkSoil.size and xcol in colPlotL:
                
                if self.plot.endMembers.darkSoil.color in ['auto','ramp']:
                    
                    color = self.slicedCM[i]
                    
                else:
                    
                    color = self.plot.endMembers.darkSoil.color
                
                if self.model.soilLineendMembersGlobalEdge:
                
                    ax.scatter( dark, self.darkRef,color=color, s=self.plot.endMembers.darkSoil.size, edgecolors=self.slicedCM[i])
                
                else:

                    ax.scatter( dark, self.endPointDf.loc[xcol]['y0'] ,color=color, s=self.plot.endMembers.darkSoil.size, edgecolors=self.slicedCM[i])
                    
                if self.plot.endMembers.lightSoil.color in ['auto','ramp']:
                    
                    color = self.slicedCM[i]
                    
                else:
                    
                    color = self.plot.endMembers.lightSoil.color
                  
                if self.model.soilLineendMembersGlobalEdge:
                
                    ax.scatter(light, self.lightRef, color=color,s=self.plot.endMembers.lightSoil.size, edgecolors=self.slicedCM[i])
                
                else:

                    ax.scatter(light, self.endPointDf.loc[xcol]['y1'], color=color,s=self.plot.endMembers.lightSoil.size, edgecolors=self.slicedCM[i])
                    
         
                
            
        
        if plot or figure:
            
            for i,xcol in enumerate(self.columns):
                        
                if xcol == self.input.yband or xcol not in colPlotL:
                    
                    continue
                
                X = self.spectraDF[[xcol]].values
                
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
                
                x,y = SetTextPos(self.plot.text.x, self.plot.text.y, self.xylimD['xmin'], self.xylimD['xmax'], self.xylimD['ymin'], self.xylimD['ymax'])
                
                ax.text(x, y, text)
                                            
            if plot:
            
                self._EndPointSoilLines(ax)
                plt.show()
          
        if figure:
          
            fig.savefig(pngFPN)   # save the figure to file
            
            plt.close(fig)
            
        return resultD        
       
    def _RegressPlotMultiTest(self, xlabel, ylabel, title, text, plot, figure, pngFPN):
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

        def EstimateEdge(X,Y,colx):
            '''
            '''
            
            self.regressor.fit(X, Y)
            
            predict = self.regressor.predict(X)
            
            if isinstance(self.regressor.coef_[0],np.ndarray):
               
                m=self.regressor.coef_[0][0]
                    
                c = self.regressor.intercept_[0]
                
            else:
                
                m=self.regressor.coef_[0]
                    
                c = self.regressor.intercept_
   
            r2 = r2_score(Y, predict)
                                    
            rmse = np.sqrt(mean_squared_error(predict, Y))
            
            regression = 'refband[%s] = %.3f*band[%s] + %.3f' %(self.input.yband, m, xcol, c)
            
            if xcol != self.input.yband:
                        
                resultD[xcol] = {'m':round(m,3), }
                
                resultD[xcol]['c'] = round(c,3)
                
                resultD[xcol]['r2'] = round(r2,3)
                
                resultD[xcol]['rmse'] = round(rmse,3)
                
                resultD[xcol]['regression'] = regression
                
            if self.model.soilLineendMembersGlobalEdge:
                
                # Calculate the light edge from the regression and the reference light
                light=(lightref-self.regressor.intercept_)/self.regressor.coef_[0] 
                
                if isinstance(light,np.ndarray):
                    
                    light = light[0]
                
                # Calculate the dark edge from the regression and the reference dark
                dark=(darkref-self.regressor.intercept_)/self.regressor.coef_[0] 
                
                if isinstance(dark,np.ndarray):
                    
                    dark = dark[0]
                    
                # Save results in dict
      
                #print ('self.regressor.intercept_',self.regressor.intercept_)
                #print ('self.regressor.coef_[0]',self.regressor.coef_[0])
                #print (self.emD['endMembers'][xcol])
            
            else:
                
                light = self.lightRefDF[colx]
            
                dark = self.darkRefDF[colx]
                
                print (dark,light)

            
            print (col,dark,light)
            
            return dark,light, rmse
          
        ''' Start of main process '''  
        # Get the bands to plot
        colPlotL = []
        
        plotskipstep = ceil( (len(self.columns)-1)/self.plot.maxBands )
                      
        for i, col in enumerate(self.columns):
            
            if i % plotskipstep == 0:
                
                colPlotL.append(col)   
 
        resultD = {}
                       
        if self.model.soilLineOriginalRange:
            
            darkref = self.darkRef
            
            lightref = self.lightRef
            
            if self.model.soilLineendMembersGlobalEdge:
            
                darkref = self.darkRefDF
            
                lightref = self.lightRefDF
        
        else:
            # Extend the soil line, if self.soilLineExtend == 0, also delta= 0 and no extension   
            #delta = (self.spectraDF[self.input.yband].max()-self.spectraDF[self.input.yband].min())*self.model.soilLineExtend/2
            
            delta = (lightref-darkref)*self.model.soilLineExtend/2
            
            print (delta)
            
            BALLE
            # Set the extended dark reference
            darkref = self.spectraDF[self.input.yband].min()-delta
            
            # Set the extended light reference
            lightref = self.spectraDF[self.input.yband].max()+delta
        
        fig, ax = plt.subplots( figsize=(self.plot.figSize.x, self.plot.figSize.y)  )
        
        if self.plot.tightLayout:
            
            fig.tight_layout()
                    
        Y = self.spectraDF[[self.input.yband]].values
                
        Y = np.ravel(Y)
        
        # Loop over the bands
        for i,xcol in enumerate(self.columns):
            
            X = self.spectraDF[[xcol]].values
                                  
            dark,light,rmse = EstimateEdge(X,Y,xcol)
            
            self.emD['endMembers'][xcol] = {'darkSoil':dark, 'lightSoil':light, 'rmse':rmse}
              
            if (plot or figure) and self.plot.endMembers.darkSoil.size and xcol in colPlotL:
                

                    
                color = self.plot.endMembers.darkSoil.color

                ax.scatter( dark, darkref,color=color, s=self.plot.endMembers.darkSoil.size, edgecolors=self.slicedCM[i])
                

                    
                color = self.plot.endMembers.lightSoil.color
                           
                ax.scatter(light, lightref, color=color,s=self.plot.endMembers.lightSoil.size, edgecolors=self.slicedCM[i])
         
        # Redo the endmember for the y-band using the last column i n the loop
        
        if xcol == self.input.yband:
            
            xcol = self.columns[i-1]
            
        X = self.spectraDF[[self.input.yband]].values
        
        Y = self.spectraDF[[xcol]].values
        
        dark,light,rmse = EstimateEdge(X,Y,xcol)
        
        self.emD['endMembers'][self.input.yband] = {'darkSoil':dark, 'lightSoil':light, 'rmse':rmse}
        
        self.emD['model']['nsamples'] = self.spectraDF.shape[0]
        '''                         
        self.regressor.fit(X, Y)
                                    
        rmse = np.sqrt(mean_squared_error(predict, Y))

        # Calculate the light edge from the regression and the reference light
        light=(lightref-self.regressor.intercept_)/self.regressor.coef_[0] 
                    
        if isinstance(light,np.ndarray):
                
            light = light[0]
            
        # Calculate the dark edge from the regression and the reference dark
        dark=(darkref-self.regressor.intercept_)/self.regressor.coef_[0] 
              
        if isinstance(dark,np.ndarray):
                
            dark = dark[0]
        
        self.emD['endMembers'][xcol] = {'darkSoil':dark, 'lightSoil':light, 'rmse':rmse}
             
        print (self.emD['endMembers'][xcol])
        '''          
        if plot or figure:
            
            for i,xcol in enumerate(self.columns):
                        
                if xcol == self.input.yband or xcol not in colPlotL:
                    
                    continue
                
                X = self.spectraDF[[xcol]].values
                
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
                
                x,y = SetTextPos(self.plot.text.x, self.plot.text.y, self.xylimD['xmin'], self.xylimD['xmax'], self.xylimD['ymin'], self.xylimD['ymax'])
                
                ax.text(x, y, text)
                                            
            if plot:
            
                plt.show()
          
        if figure:
          
            fig.savefig(pngFPN)   # save the figure to file
            
            plt.close(fig)
            
        return resultD
              
    def _RegressPlotMulti(self, xlabel, ylabel, text, plot, figure, pngFPN):
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

        title = 'Distilled regression soil lines (%s) (%s std). \nProject: %s' %(self.model.regressor, self.model.withinNrStd, self.params.id)

        # Get the bands to plot
        colPlotL = []
        
        plotskipstep = ceil( (len(self.columns)-1)/self.plot.maxBands )
                      
        for i, col in enumerate(self.columns):
            
            if i % plotskipstep == 0:
                
                colPlotL.append(col)   
 
        resultD = {}
                       
        # Extend the soil line, if self.soillineExtend == 0, also delta= 0 and no extension   
        delta = (self.spectraDF[self.input.yband].max()-self.spectraDF[self.input.yband].min())*(self.model.edgeExtraction.distillExtendFactor-1.0)/2
            
        # Set the extended dark reference
        darkref = self.spectraDF[self.input.yband].min()-delta
            
        # Set the extended light reference
        lightref = self.spectraDF[self.input.yband].max()+delta
        
        fig, ax = plt.subplots( figsize=(self.plot.figSize.x, self.plot.figSize.y)  )
        
        if self.plot.tightLayout:
            
            fig.tight_layout()
                    
        Y = self.spectraDF[[self.input.yband]].values
                
        Y = np.ravel(Y)
        
        # Loop over the bands
        for i,xcol in enumerate(self.columns):
                        
            if xcol == self.input.yband:
                
                continue
            
            X = self.spectraDF[[xcol]].values
                                  
            self.regressor.fit(X, Y)
            
            predict = self.regressor.predict(X)
            
            if isinstance(self.regressor.coef_[0],np.ndarray):
               
                m=self.regressor.coef_[0][0]
                    
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
                
                self.emD['endMembers'][xcol] = {'darkSoil':dark, 'lightSoil':light, 'rmse':rmse}
                
            else:
                
                self.emD['endMembers'][xcol] = {'darkSoil':dark, 'lightSoil':light, 'rmse':rmse}
            
            self.emD['model']['nsamples'] = self.spectraDF.shape[0]
              
            if (plot or figure) and self.plot.endMembers.darkSoil.size and xcol in colPlotL:
                
                if self.plot.endMembers.darkSoil.color in ['auto','ramp']:
                    
                    color = self.slicedCM[i]
                    
                else:
                    
                    color = self.plot.endMembers.darkSoil.color
                
                ax.scatter( dark, darkref,color=color, s=self.plot.endMembers.darkSoil.size)
                
                if self.plot.endMembers.lightSoil.color in ['auto','ramp']:
                    
                    color = self.slicedCM[i]
                    
                else:
                    
                    color = self.plot.endMembers.lightSoil.color
                           
                ax.scatter(light, lightref, color=color,s=self.plot.endMembers.lightSoil.size)
                
        if plot or figure:
            
            for i,xcol in enumerate(self.columns):
                        
                if xcol == self.input.yband or xcol not in colPlotL:
                    
                    continue
                
                X = self.spectraDF[[xcol]].values
                
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
                
                x,y = SetTextPos(self.plot.text.x, self.plot.text.y, self.xylimD['xmin'], self.xylimD['xmax'], self.xylimD['ymin'], self.xylimD['ymax'])
                
                ax.text(x, y, text)
                                            
            if plot:
            
                plt.show()
          
        if figure:
          
            fig.savefig(pngFPN)   # save the figure to file
            
            plt.close(fig)
            
        return resultD
    
    def _RegressPlotOriginal(self, xlabel, ylabel, text, plot, figure, pngFPN):
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
        
        title = 'Regression soil lines (%s) using all data points. \nProject: %s' %(self.model.regressor, self.params.id)

        thePlotText = '%s\n showing %s bands.' %(text, self.plot.maxBands)
        
        # Get the bands to plot
        colPlotL = []
        
        plotskipstep = ceil( (len(self.columns)-1)/self.plot.maxBands )
                      
        for i, col in enumerate(self.columns):
            
            if i % plotskipstep == 0:
                
                colPlotL.append(col)   
 
        resultD = {}
        
        '''               
        #if self.model.soilLineOriginalRange:
            
        #    darkref = self.darkRef
            
        #    lightref = self.lightRef
        
        #else:
            # Extend the soil line, if self.soillineExtend == 0, also delta= 0 and no extension   
        #    delta = (self.spectraDF[self.input.yband].max()-self.spectraDF[self.input.yband].min())*self.model.soilLineExtend/2
            
            # Set the extended dark reference
        #    darkref = self.spectraDF[self.input.yband].min()-delta
            
            # Set the extended light reference
        #    lightref = self.spectraDF[self.input.yband].max()+delta
        '''
        
        fig, ax = plt.subplots( figsize=(self.plot.figSize.x, self.plot.figSize.y)  )
        
        if self.plot.tightLayout:
            
            fig.tight_layout()
                    
        Y = self.spectraDF[[self.input.yband]].values
                
        Y = np.ravel(Y)
        
        if plot or figure:
            
            for i,xcol in enumerate(self.columns):
                        
                if xcol == self.input.yband or xcol not in colPlotL:
                    
                    continue
                
                X = self.spectraDF[[xcol]].values
                
                self.regressor.fit(X, Y)
            
                ax.scatter(X, Y, color=self.slicedCM[i])
            
                ax.plot(X, self.regressor.predict(X), color=self.slicedCM[i], label=xcol)
                
        # Loop over the bands
        for i,xcol in enumerate(self.columns):
                        
            if xcol == self.input.yband:
                
                continue
            
            X = self.spectraDF[[xcol]].values
                                  
            self.regressor.fit(X, Y)
            
            predict = self.regressor.predict(X)
            
            if isinstance(self.regressor.coef_[0],np.ndarray):
               
                m=self.regressor.coef_[0][0]
                    
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
            
            # Calculate the light edge from the regression and the lightest signal for this band
                        
            lightref = self.endPointDf.loc[xcol]['y1']
                
            darkref = self.endPointDf.loc[xcol]['y0']
            
            light= (lightref-self.regressor.intercept_)/self.regressor.coef_[0] 
            
            if light >1:
                
                lightref = self.regressor.intercept_ + self.regressor.coef_[0]*1
                
                light = 1
            
            # Calculate the dark edge from the regression and the reference dark
            dark = max(0,(darkref-self.regressor.intercept_)/self.regressor.coef_[0] )
            
            if dark < 0:
                
                darkref = self.regressor.intercept_ + self.regressor.coef_[0]*0
                
                dark = 0
                
                print ('darkref', darkref)
                    
            self.emD['endMembers'][xcol] = {'darkSoil':dark, 'lightSoil':light, 'rmse':rmse}
                
            self.emD['model']['nsamples'] = self.spectraDF.shape[0]
              
            if (plot or figure) and self.plot.endMembers.darkSoil.size and xcol in colPlotL:
                 
                color = self.plot.endMembers.darkSoil.color
                
                ax.scatter( dark, darkref,color=color, edgecolors=self.slicedCM[i], s=self.plot.endMembers.darkSoil.size)
                  
                color = self.plot.endMembers.lightSoil.color
                           
                ax.scatter(light, lightref, color=color, edgecolors=self.slicedCM[i], s=self.plot.endMembers.lightSoil.size)
                
        if (plot or figure):
               
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
            
            x,y = SetTextPos(self.plot.text.x, self.plot.text.y, self.xylimD['xmin'], self.xylimD['xmax'], self.xylimD['ymin'], self.xylimD['ymax'])
                
            ax.text(x, y, thePlotText)
                                            
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
       
        X = self.spectraDF[[xcol]].values
        
        Y = self.spectraDF[[self.input.yband]].values
        
        Y = np.ravel(Y)
                
        self.regressor.fit(X, Y)
                        
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
        
        X = self.spectraDF[[xcol]].values
        
        Y = self.spectraDF[[self.input.yband]].values
        
        denom = sqrt(slope**2+1)
        
        colid = 'dev%s' %(xcol)
        
        self.spectraDF[colid] = ((slope*X)-Y+icept)/denom
            
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
            self.spectraDF = self.spectraDF[self.spectraDF[colid] > -devStd*self.model.withinNrStd]
            
        else:
            
            # Remove outliers on both sides
            self.spectraDF = self.spectraDF[abs(self.spectraDF[colid]) < devStd*self.model.withinNrStd]
     
    def translate4OrthoEm(self):
        """ Translate endmbmers to ortho model input format
        """
        
        # Create empty nedmember for Ortho modeling endmember 
        # self.orthoEmLD = []
        
        # assemble the spectral endMembers for draksoil and lightSoil
        wlL = []
        
        darkSoilL = []; lightSoilL = []
        
        soilStdL = []; 
        
        for k, v in self.emD['endMembers'].items():
            
            darkSoilL.append(v['darkSoil'])
            
            lightSoilL.append(v['lightSoil'])
            
            soilStdL.append(v['rmse'])
            
            wlL.append(k)
            
        jsonDumpD = {}
                
        # Copy from input
        copyL = ['id','name','userId','importVersion']
        
        for key in copyL:

            jsonDumpD[key] = self.paramD[key]
                
        jsonDumpD['campaign'] = self.paramD['campaign']
        
        jsonDumpD['waveLength'] = self.jsonSpectraData['waveLength']
            
        darkSoilD = {"name":"darkSoil", "reference": "dark end soilline regression", "abundance":-999, "signalMean": darkSoilL, "signalStd": soilStdL}
          
        lightSoilD = {"name":"lightSoil", "reference": "light end soilline regression", "abundance":-999, "signalMean": lightSoilL, "signalStd": soilStdL}

        # Add lightSoilD to the endmemberlist
        self.substanceEmL.insert(0,lightSoilD)
        
        # Add darkSoilD to the endmemberlist
        self.substanceEmL.insert(0,darkSoilD)
        
        jsonDumpD["spectra"] = self.substanceEmL
                        
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
        '''
        D_JsonDump = json.loads(json.dumps(self.params, default=lambda o: o.__dict__))
        
        jsonF = open(paramFPN, "w")
        
        json.dump(D, jsonF, indent = 2)
        
        jsonF.close()
        '''
        
        if self.verbose:
            
            infostr =  '        SoiLline regressions saved as: %s' %(self.regrJsonFPN)
        
            print (infostr)
            
            infostr =  '        endMembers saved as: %s' %(self.endmemberJsonFPN)
        
            print (infostr)
            
            infostr =  '        endMembers for ortho modelling saved as: %s' %(self.endmember4OrthoJsonFPN)
        
            print (infostr)
              
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
            
        # set the text
        text = self.plot.text.text
        
        # Add the bandWidth
        if self.plot.text.bandWidth:
                        
            bandWidth = (max(self.columns)- min(self.columns))/(len(self.columns)-1)

            text += '\nbandWidth=%s nm' %( bandWidth )
        
        # Add number of samples to text
        if self.plot.text.samples:
            
            text += '\nnSamples=%s; nBands=%s' %( self.spectraDF.shape[0],len(self.columns))
              
        yLabel = 'Reflectance ref.band (%s)' %(self.input.yband)
        
        xLabel = 'Reflectance other bands'
        
        return (xLabel, yLabel, text)
      
    def _PilotExtractSoilline(self):
        ''' Steer the sequence of processes for extracting soillines spectral soil data in json format
        '''

        # Get the band data
        self._GetBandData()
        
        # Set the plot title, labels and annotation
        titleSuffix = '(all data points)'
        
        xLabel, yLabel, text = self._PlotTitleTextn(titleSuffix)
        
        self._OriginalMinMaxSoilLines(xLabel, yLabel,text)

        # Regression and plot for raw data
        self.soillineD['raw']['nSamples'] = self.spectraDF.shape[0]
        
        
        #self.soillineD['raw']['soilLines'] = self._ExtractSoilLines( xLabel, yLabel, title, text,
        #                                          self.plot.rawSoilLines, self.figure.rawSoilLines, self.rawModelPngFPN )
        
        self.soillineD['raw']['soilLines'] = self._RegressPlotOriginal( xLabel, yLabel, text,
                                                  self.plot.rawSoilLines, self.figure.rawSoilLines, self.rawModelPngFPN )

        # Get the standard deviation for all soil points using raw data
        # To be retained a spectra must pass all the input data
        devStdD = {}
        
        for i,xcol in enumerate(self.columns):
        
            if xcol in [self.input.yband]:
                
                continue
            
            colid = self._SoilDeviation(xcol, i)
            
            devStdD[xcol] = self.spectraDF[colid].std()
                    
        # Remove outliers for all input bands
        # To be retained a spectra must pass all the input data  
        for xcol in devStdD:
                            
            colid = 'dev%s' %(xcol)
            
            self._RemoveOutliers(colid, devStdD[xcol])
                        
        # Set the plot title, labels and annotation
        titleSuffix = '(final soillines)'
        
        xLabel, yLabel, text = self._PlotTitleTextn(titleSuffix)
        
        self.soillineD['final']['nSamples'] = self.spectraDF.shape[0]
        
        #self.soillineD["final"]['soilLines'] = self._ExtractSoilLines( xLabel, yLabel, title, text,
        #                                 self.plot.finalSoilLines, self.figure.finalSoilLines, self.finalModelPngFPN )
        
        self.soillineD["final"]['soilLines'] = self._RegressPlotMulti( xLabel, yLabel, text,
                                         self.plot.finalSoilLines, self.figure.finalSoilLines, self.finalModelPngFPN )


        if self.verbose > 1:
            
            pp = pprint.PrettyPrinter(indent=2)

            pp.pprint(self.soillineD)
            
            pp.pprint(self.emD)
            
        self._JsonDumpResults()
     
def SetupProcesses(docpath, createjsonparams,  arrangeddatafolder, projFN, jsonpath):
    '''Setup and loop processes
    
    :param docpath: path to project root folder 
    :type: lstr
    
    :param sourcedatafolder: folder name of original OSSL data (source folder)  
    :type: lstr
    
    :param arrangeddatafolder: folder name of arranged OSSL data (destination folder) 
    :type: lstr
            
    :param projFN: project filename (in destination folder)
    :type: str
    
    :param jsonpath: folder name
    :type: str
            
    '''
        
    dstRootFP, jsonFP = CheckMakeDocPaths(docpath,arrangeddatafolder, jsonpath)
    
    if createjsonparams:
        
        CreateArrangeParamJson(jsonFP, projFN, 'soilline')
        
    jsonProcessObjectL = ReadProjectFile(dstRootFP, projFN, jsonFP)
           
    #Loop over all json files and create Schemas and Tables
    for jsonObj in jsonProcessObjectL:
                
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

    docpath = '/Users/thomasgumbricht/docs-local/OSSL/Sweden/LUCAS'
        
    arrangeddatafolder = 'arranged-data'
    
    projFN = 'extract_soillines.txt'
    
    jsonpath = 'json-soillines'
    
    createjsonparams=False
    
    SetupProcesses(docpath, createjsonparams,  arrangeddatafolder, projFN, jsonpath)

