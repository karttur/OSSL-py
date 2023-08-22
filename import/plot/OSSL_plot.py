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

import matplotlib.pyplot as plt

from copy import deepcopy

from math import ceil

# Third party imports

import json

import numpy as np

import pandas as pd

# Package application imports

def PlotParams():
    ''' Default parameters for plotting soil spectral library data
    
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
        
    paramD['input']['bandjump'] = 1
        
    paramD['plot'] = {}
    
    paramD['plot']['raw'] = True
    
    paramD['plot']['derivative'] = True
            
    paramD['plot']['colorRamp'] = "jet"
    
    paramD['plot']['maxSpectra'] = 100
    
    paramD['plot']['figSize'] = {'x':0,'y':0}
    
    paramD['plot']['legend'] = False
    
    paramD['plot']['tightLayout'] = False
    
    paramD['plot']['scatter'] = {'size':50}
            
    paramD['plot']['text'] = {'x':0.1,'y':0.9}
    
    paramD['plot']['text']['bandWidth'] = True
    
    paramD['plot']['text']['samples'] = True
    
    paramD['plot']['text']['text'] = ''
    
    paramD['figure'] = {} 
    
    paramD['figure']['raw'] = True
    
    paramD['figure']['derivative'] = True
        
    paramD['xyLim'] = {}
    
    paramD['xyLim']['xMin'] = 15
    
    paramD['xyLim']['xMax'] = 80
    
    paramD['xyLim']['yMin'] = 45
    
    paramD['xyLim']['yMax'] = 80
        
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
    
def ReadPlotJson(jsonFPN):
    """ Read the parameters for plotting
    
    :param jsonFPN: path to json file
    :type jsonFPN: str
    
    :return paramD: parameters
    :rtype: dict
   """
    
    with open(jsonFPN) as jsonF:
    
        paramD = json.load(jsonF)
        
    return (paramD)
                
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
            
                   
    def _SetTextPos(self, plot, xmin, xmax, ymin, ymax):
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
        
        x = plot.text.x*(xmax-xmin)+xmin
        
        y = plot.text.y*(ymax-ymin)+ymin
        
        return (x,y)
                    
class SpectraPlot(Obj):
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
        paramD.pop('plot')
                
        # Deep copy the parameters to self.plotD
        self.plotD = deepcopy(paramD)
               
        # Open and load JSON data file
        with open(self.input.jsonSpectraDataFilePath) as jsonF:
            
            self.jsonSpectraData = json.load(jsonF)
                    
    def _SetcolorRamp(self,n):
        ''' Slice predefined colormap to discrete colors for each band
        '''
                        
        # Set colormap to use for plotting
        cmap = plt.get_cmap(self.plot.colorRamp)
        
        # Segmenting colormap to the number of bands
        self.slicedCM = cmap(np.linspace(0, 1, n)) 
           
    def _SpectraDerivativeFromDf(self):
        ''' Create spectral derivates
        '''
        
        # Get the derivatives
        spectraDerivativeDF = self.spectraDf.diff(axis=1, periods=1)
        
        # Drop the first column as it will have only NaN
        spectraDerivativeDF = spectraDerivativeDF.drop(self.columns[0], axis=1)
               
        # Reset columns to integers
        columns = [int(i) for i in self.columns]
        
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
        
        # get the average bandwidth
        self.bandWidth = (max(self.columns)- min(self.columns))/(len(self.columns)-1)
        
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
                               
        self.spectraDf = pd.DataFrame(data=spectraA, columns=self.columns)
        
        if self.plot.derivatives.apply:
            
            self.spectraDerivativeDF,self.derivativeColumns = self._SpectraDerivativeFromDf()
               
    def _SetDstFPNs(self):
        ''' Set destination file paths and names
        '''

        FP,FN = os.path.split(self.input.jsonSpectraDataFilePath)
                
        FN = os.path.splitext(FN)[0]
        
        self.modelN = FN.split('_', 1)[1]
            
        plotRootFP = os.path.join(FP,'plot')
        
        if not os.path.exists(plotRootFP):
            
            os.makedirs(plotRootFP)

        rawPngFN = '%s_spectra.png' %(self.modelN)

        self.rawPngFPN = os.path.join(plotRootFP, rawPngFN)
        
        derivativePngFN = '%s_derivative.png' %(self.modelN)

        self.derivativePngFPN = os.path.join(plotRootFP, derivativePngFN)
        
        dualPngFN = '%s_spectra+derivative.png' %(self.modelN)

        self.dualPngFPN = os.path.join(plotRootFP, dualPngFN)
                                                  
    def _PlotMonoMulti(self, dataframe, x, plot, pngFPN):
        ''' Single subplot for multiple bands
        
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
        plotskipStep = ceil( (len(self.spectraDf.index)-1)/self.plot.maxSpectra )
                
        xLabel = plot.axisLabel.x
        
        yLabel = plot.axisLabel.y
        
        # Set the plot title, labels and annotation
        title, text = self._PlotTitleTextn(plot,plotskipStep)
               
        fig, ax = plt.subplots( figsize=(self.plot.figSize.x, self.plot.figSize.y)  )

        n = int(len(self.spectraDf.index)/plotskipStep)+1
        
        # With n bands known, create the colorRamp
        self._SetcolorRamp(n)
        
        # Loop over the spectra
        i = -1
        n = 0
        for index, row in dataframe.iterrows():
            
            i += 1
            
            if i % plotskipStep == 0:
                
                ax.plot(x, row, color=self.slicedCM[n])
                
                n += 1

        if self.plot.xlim.xmin:
                        
            ax.set_xlim(self.plot.xlim.xmin, self.plot.xlim.xmax)
            
        if plot.ylim.ymin:
                        
            ax.set_ylim(plot.ylim.ymin, plot.ylim.ymax)
          
        # Get the limits of the plot area - to fit the text 
        xyLimD = {}
         
        xyLimD['xMin'],xyLimD['xMax'] = ax.get_xlim()
        
        xyLimD['yMin'],xyLimD['yMax'] = ax.get_ylim()
        
        ax.set(xlabel=xLabel, ylabel=yLabel, title=title)
        
        if self.plot.legend:
                    
            ax.legend(loc=self.plot.legend)
        
        if text != None:
            
            x,y = self._SetTextPos(plot, xyLimD['xMin'], xyLimD['xMax'], xyLimD['yMin'], xyLimD['yMax'])
            
            ax.text(x, y, text)
            
        # Set tight layout if requested
        if self.plot.tightLayout:
            
            fig.tight_layout()
                                        
        if self.plot.screenDraw:
        
            plt.show()
          
        if self.plot.savePng:
          
            fig.savefig(pngFPN)   # save the figure to file
            
        plt.close(fig)
            
            
    def _PlotDualMulti(self,pngFPN):
        ''' Single subplot for multiple bands
        
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
        plotskipStep = ceil( (len(self.spectraDf.index)-1)/self.plot.maxSpectra )
                        
        # Set the plot title, labels and annotation   
        #title, text = self._PlotTitleTextn(plot,plotskipStep)
    
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(self.plot.figSize.x, self.plot.figSize.y), sharex=True  )
        
        n = int(len(self.spectraDf.index)/plotskipStep)+1
        
        # With n bands known, create the colorRamp
        self._SetcolorRamp(n)
        
        xraw = [int(i) for i in self.columns]
        
        xderivative = [int(i[1:len(i)]) for i in self.derivativeColumns]
                    
        # Loop over the spectra
        i = -1
        n = 0
        for index, row in self.spectraDf.iterrows():
            i += 1
            if i % plotskipStep == 0:
                
                ax[0].plot(xraw, row, color=self.slicedCM[n])
                                
                n += 1
                
        # Loop over the derivatives      
        i = -1
        n = 0
        for index, row in self.spectraDerivativeDF.iterrows():
            i += 1
            if i % plotskipStep == 0:
                
                ax[1].plot(xderivative, row, color=self.slicedCM[n])
                
                n += 1
                     
        if self.plot.xLim.xMin:
                        
            ax[0].set_xlim(self.plot.xLim.xMin, self.plot.xLim.xMax)
            ax[1].set_xlim(self.plot.xLim.xMin, self.plot.xLim.xMax)
            
        if self.plot.raw.yLim.yMin:
                        
            ax[0].set_ylim(self.plot.raw.yLim.yMin,self.plot.raw.yLim.yMax)
            
        if self.plot.derivatives.yLim.yMin:
                        
            ax[1].set_ylim(self.plot.derivatives.yLim.yMin, self.plot.derivatives.yLim.yMax)
          
        # Get the limits of the plot areas - to fit the text 
        rawxyLimD = {}; derivativexyLimD = {}
         
        rawxyLimD['xMin'],rawxyLimD['xMax'] = ax[0].get_xlim()
        
        rawxyLimD['yMin'],rawxyLimD['yMax'] = ax[0].get_ylim()
        
        derivativexyLimD['xMin'],derivativexyLimD['xMax'] = ax[1].get_xlim()
        
        derivativexyLimD['yMin'],derivativexyLimD['yMax'] = ax[1].get_ylim()
        
        ax[0].set(ylabel=self.plot.raw.axisLabel.y, title=self.plot.raw.title.title)
        
        ax[1].set(xlabel=self.plot.raw.axisLabel.x, ylabel=self.plot.derivatives.axisLabel.y, 
                  title=self.plot.derivatives.title.title)
        
        if self.plot.legend:
                    
            ax[0].legend(loc=self.plot.legend)
        
        rawtext = self._PlotTitleTextn(self.plot.raw,plotskipStep)[1]
        
        if self.plot.raw.text != None:
               
            x,y = self._SetTextPos(self.plot.raw, rawxyLimD['xMin'], rawxyLimD['xMax'], rawxyLimD['yMin'], rawxyLimD['yMax'])
            
            ax[0].text(x, y, rawtext)
            
        derivativetext = self._PlotTitleTextn(self.plot.derivatives,0)[1]
        
        if self.plot.derivatives.text != None:
               
            x,y = self._SetTextPos(self.plot.derivatives, derivativexyLimD['xMin'], derivativexyLimD['xMax'], derivativexyLimD['yMin'], derivativexyLimD['yMax'])
            
            ax[1].text(x, y, derivativetext)
            
        # Set supTitle
        if self.plot.supTitle:
            
            if self.plot.supTitle == "auto":
                
                supTitle = self.modelN
            
            else:
                
                supTitle = self.plot.supTitle
            
            fig.suptitle(supTitle)
   
        # Set tight layout if requested
        if self.plot.tightLayout:
            
            fig.tight_layout()
                                        
        if self.plot.screenDraw:
        
            plt.show()
          
        if self.plot.savePng:
          
            fig.savefig(pngFPN)   # save the figure to file
            
        plt.close(fig)


    def _PlotTitleTextn(self, plot, plotskipStep):
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
        text = plot.text.text
        
        # Add the bandwidth
        if plot.text.bandWidth:
                        
            text += '\nbandwidth=%s nm' %( self.bandWidth )
        
        # Add number of samples to text
        if plot.text.samples:
            
            text += '\nnspectra=%s; nbands=%s' %( self.spectraDf.shape[0],len(self.columns))
            
        if plot.text.skipStep:
                
            text += '\nshowing every %s spectra' %( plotskipStep )

        return (title, text)
      
    def _PilotPlot(self):
        ''' Steer the sequence of processes for plotting spectra data in json format
        '''

        # Get the band data
        self._GetBandData()
        
        # plot the data
        if self.plot.raw.apply and self.plot.derivatives.apply:
        
            self._PlotDualMulti(self.dualPngFPN )
            
        elif self.plot.derivatives.apply:

            x = [int(i[1:len(i)]) for i in self.derivativeColumns]
            
            self._PlotMonoMulti(self.spectraDerivativeDF, x, self.plot.derivatives, self.derivativePngFPN )
        
        else:    
            x = [int(i) for i in self.columns]
            self._PlotMonoMulti(self.spectraDf, x, self.plot.raw, self.rawPngFPN )
            
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
                
    plotjsonFP = os.path.join(dstRootFP,jsonpath)
    
    if not os.path.exists(plotjsonFP):
        
        os.makedirs(plotjsonFP)
    
    if createjsonparams:
        
        flag, jsonFPN = CreateParamJson(plotjsonFP)
        
        if flag:
            
            exitstr = 'plot json parameter file already exists: %s\n' %(jsonFPN)
        
        else:
        
            exitstr = 'plot json parameter file created: %s\n' %(jsonFPN)
        
        exitstr += ' Edit the plot json file for your project and move+rename it to reflect the commands.\n' 
        
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
    jsonL = [os.path.join(plotjsonFP,x.strip())  for x in jsonL if len(x) > 10 and x[0] != '#']

    #Loop over all json files and create Schemas and Tables
    for jsonObj in jsonL:
        
        print ('    jsonObj:', jsonObj)
        
        paramD = ReadPlotJson(jsonObj)
        
        # Invoke the soil line
        spectraPlt = SpectraPlot(paramD)
        
        # Set the dst file names
        spectraPlt._SetDstFPNs()
        
        # run the soilline extractor
        spectraPlt._PilotPlot()
                                  
if __name__ == '__main__':
    ''' If script is run as stand alone
    '''
        
    
    docpath = '/Users/thomasgumbricht/docs-local/OSSL/SU-Tovetorp/C14384MA-01'
    docpath = '/Users/thomasgumbricht/docs-local/OSSL/Sweden/LUCAS'
        
    arrangeddatafolder = 'arranged-data'
    
    projFN = 'plot_spectra.txt'
    
    jsonpath = 'json-plots'
    
    createjsonparams=False
    
    SetupProcesses(docpath, arrangeddatafolder, projFN, jsonpath, createjsonparams)