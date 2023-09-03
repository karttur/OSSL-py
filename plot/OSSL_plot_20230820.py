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

# Package application imports

from util.makeObject import Obj

from util.jsonIO import ReadAnyJson, LoadBandData

from util.defaultParams import PlotParams, CheckMakeDocPaths, CreateArrangeParamJson, ReadProjectFile

from util.pdFrameProcesses import SpectraDerivativeFromDf
       
def ReadPlotJson(jsonFPN):
    """ Read the parameters for plotting
    
    :param jsonFPN: path to json file
    :type jsonFPN: str
    
    :return paramD: parameters
    :rtype: dict
   """
    
    return ReadAnyJson(jsonFPN)
                     
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
        self._SetPlotDefaults()
        
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
           
                 
    def _GetBandData(self):
        ''' Read json data into numpy array and convert to pandas dataframe
        '''
                        
        self.SpectraDF = LoadBandData(self.columns, self.jsonSpectraData)
        
                
        if self.plot.derivatives.apply:
            
            self.spectraDerivativeDF,self.derivativeColumns = SpectraDerivativeFromDf(self.SpectraDF,self.columns)
                 
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
        plotskipStep = ceil( (len(self.SpectraDF.index)-1)/self.plot.maxSpectra )
                
        xLabel = plot.axisLabel.x
        
        yLabel = plot.axisLabel.y
        
        # Set the plot title, labels and annotation
        title, text = self._PlotTitleText(plot,plotskipStep)
               
        fig, ax = plt.subplots( figsize=(self.plot.figSize.x, self.plot.figSize.y)  )

        n = int(len(self.SpectraDF.index)/plotskipStep)+1
        
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

        if self.plot.xLim.xMin:
                        
            ax.set_xlim(self.plot.xLim.xMin, self.plot.xLim.xMax)
            
        if plot.yLim.yMin:
                        
            ax.set_ylim(plot.yLim.yMin, plot.yLim.yMax)
          
        # Get the limits of the plot area - to fit the text 
        xyLimD = {}
         
        xyLimD['xMin'],xyLimD['xMax'] = ax.get_xlim()
        
        xyLimD['yMin'],xyLimD['yMax'] = ax.get_ylim()
        
        ax.set(xlabel=xLabel, ylabel=yLabel, title=title)
        
        if self.plot.legend:
                    
            ax.legend(loc=self.plot.legend)
        
        if text != None:
            
            x,y = self._SetPlotTextPos(plot, xyLimD['xMin'], xyLimD['xMax'], xyLimD['yMin'], xyLimD['yMax'])
            
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
        plotskipStep = ceil( (len(self.SpectraDF.index)-1)/self.plot.maxSpectra )
                            
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(self.plot.figSize.x, self.plot.figSize.y), sharex=True  )
        
        n = int(len(self.SpectraDF.index)/plotskipStep)+1
        
        # With n bands known, create the colorRamp
        self._SetcolorRamp(n)
        
        xraw = [int(i) for i in self.columns]
        
        xderivative = [int(i[1:len(i)]) for i in self.derivativeColumns]
                    
        # Loop over the spectra
        i = -1
        n = 0
        for index, row in self.SpectraDF.iterrows():
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
        
        rawtext = self._PlotTitleText(self.plot.raw,plotskipStep)[1]
        
        if self.plot.raw.text != None:
               
            x,y = self._SetPlotTextPos(self.plot.raw, rawxyLimD['xMin'], rawxyLimD['xMax'], rawxyLimD['yMin'], rawxyLimD['yMax'])
            
            ax[0].text(x, y, rawtext)
            
        derivativetext = self._PlotTitleText(self.plot.derivatives,0)[1]
        
        if self.plot.derivatives.text != None:
               
            x,y = self._SetPlotTextPos(self.plot.derivatives, derivativexyLimD['xMin'], derivativexyLimD['xMax'], derivativexyLimD['yMin'], derivativexyLimD['yMax'])
            
            ax[1].text(x, y, derivativetext)
            
        # Set supTitle
        if self.plot.supTitle:
            
            if self.plot.supTitle == "auto":
                
                supTitle = 'Project: %s' %(self.name)
            
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

    def _PlotTitleText(self, plot, plotskipStep):
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
        
        # Add the bandwidth to text
        if plot.text.bandWidth:
                        
            text += '\nbandwidth=%s nm' %( self.bandWidth )
        
        # Add number of samples to text
        if plot.text.samples:
            
            text += '\nnspectra=%s; nbands=%s' %( self.SpectraDF.shape[0],len(self.columns))
            
        if plot.text.skipStep and plotskipStep:
                
            text += '\nshowing every %s spectra' %( plotskipStep )

        return (title, text)
      
    def _PilotPlot(self):
        ''' Steer the sequence of processes for plotting spectra data in json format
        '''

        # Use the wavelength as column headers
        columns = self.jsonSpectraData['waveLength']
        
        # get the average bandwidth
        self.bandWidth = (max(columns)- min(columns))/(len(columns)-1)
        
        # Convert the column headers to strings
        self.columns = [str(c) for c in columns]
        
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
            self._PlotMonoMulti(self.SpectraDF, x, self.plot.raw, self.rawPngFPN )
            
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
        
        CreateArrangeParamJson(jsonFP, projFN, 'plot')
        
    jsonProcessObjectL = ReadProjectFile(dstRootFP, projFN, jsonFP)
           
    #Loop over all json files and create Schemas and Tables
    for jsonObj in jsonProcessObjectL:
                
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
        
    docpath = '/Users/thomasgumbricht/docs-local/OSSL/Sweden/LUCAS'
    docpath = '/Users/thomasgumbricht/docs-local/OSSL/Europe/LUCAS'
    
    createjsonparams=False
        
    arrangeddatafolder = 'arranged-data'
    
    projFN = 'plot_spectra.txt'
    
    jsonpath = 'json-plots'

    SetupProcesses(docpath, createjsonparams, arrangeddatafolder, projFN, jsonpath)