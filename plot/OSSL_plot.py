#!/Applications/anaconda3/envs/spectraimagine_py38/bin/python3.8
'''
Created on 3 Aug 2023

@author: thomasgumbricht

Notes
-----
The module plot.py:

    requires that you have soil spectra data organised as json files in xSpectre format. 
    (see https://karttur.github.io/soil-spectro/libspectrodata/spectrodata-OSSL4ML03-plot/).
    The script OSSL_import.py arranges OSSL data to the required format.
    
    The script takes a single input parameter - the path (string) 
    to a json file with the following structure:
    
        {
          "rootpath": full/path/to/folder/where/you/saved/OSSL.zip,
          "sourcedatafolder": "data",
          "arrangeddatafolder": "arranged-data", 
          "projFN": "plot_spectra.txt", 
          "jsonfolder": "json-plots", 
          "createjsonparams": false
        }
            
        - rootpath: full path to folder with a downloaded OSSL zip file; parent folder to the exploded OSSL subfolder ("data")
        - sourcedatafolder: subfolder under "rootpath" with the exploded content of the OSSL zip file (default = "data")
        - arrangeddatafolder: subfolder under "rootpath" where the imported (rearranged) OSSL data will be stored 
        - projFN: the name of an existing txt files that sequentially lists json parameter files to run
        - jsonfolder: the relative path (vis-a-vis "rootpath") where the json parameter files (listed in "projFN") are located 
        - createjsonparams: if set to true the script will create a template json file and exit

     
    With an edited json parameter file the script reads the spectral data in xSpectre´s json format.
    The script first run the stand alone "def SetupProcesses" that reads the txt file "projFN" and 
    then sequentialy run the json parameter files listed. 
            
'''

import sys

import os

import json

import datetime

from copy import deepcopy

import csv

# Third party imports
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from math import ceil, floor

import pprint
  
def Today():
    """ Get toadys dat in string format
    
    :return todays-date 
    :rtype: str
    """
    
    return datetime.datetime.now().date().strftime("%Y%m%d")

def ReadCSV(FPN):
    ''' Standard reader for all OSSL csv data files
    
    :param FPN: path to csv file
    :type FPN: str
    
    :return headers: list of columns
    :rtype: list
    
    :return rowL: array of data
    :rtype: list of list
    '''

    rowL = []
    
    with open( FPN, 'r' ) as csvF:
                    
        reader = csv.reader(csvF)
        
        headers = next(reader, None)

        for row in reader:
            
            rowL.append(row)
            
    return headers, rowL

def DumpAnyJson(dumpD, jsonFPN):
    ''' dump, any json object
    
    :param exportD: formatted dictionary
    :type exportD: dict
    '''
            
    jsonF = open(jsonFPN, "w")
  
    json.dump(dumpD, jsonF, indent = 2)
  
    jsonF.close()

def ReadAnyJson(jsonFPN):
    """ Read json parameter file
    
    :param jsonFPN: path to json file
    :type jsonFPN: str
    
    :return paramD: parameters
    :rtype: dict
   """
    
    with open(jsonFPN) as jsonF:
    
        jsonD = json.load(jsonF)
        
    return (jsonD)

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

def PlotParams():
    ''' Default parameters for plotting soil spectral library data
    
        :returns: parameter dictionary
        :rtype: dict
    '''
    
    paramD = StandardParams()
    
    paramD['campaign'] = CampaignParams()
    
    paramD['input'] = { 'jsonSpectraDataFilePath': '/Users/thomasgumbricht/docs-local/OSSL/Sweden/LUCAS/arranged-data/visnir/OSSL-LUCAS-SE_460-1050_10/data-visnir_LUCAS_460-1050_10.json',
             'jsonSpectraParamsFilePath': '/Users/thomasgumbricht/docs-local/OSSL/Sweden/LUCAS/arranged-data/visnir/OSSL-LUCAS-SE_460-1050_10/params-visnir_LUCAS_460-1050_10.json'}

    paramD['spectraPlot'] = { 'apply': True,
                   'savePng': True,
                   'screenDraw': True,
                   'legend': False,
                   'maxSpectra': 500,
                   'colorRamp': 'jet',
                   'supTitle': 'auto',
                   'tightLayout': True,
                   'singles': {'figSize': {'x': 8, 'y': 6}},
                   'duals': {'figSize': {'x': 8, 'y': 8}},
                   'xLim': {'xMax': 1045, 'xMin': 465},
                   'raw': { 'apply': True,
                            'axisLabel': { 'x': 'Wavelength (nm)',
                                           'y': 'reflectance'},
                            'text': { 'bandWidth': True,
                                      'samples': True,
                                      'skipStep': True,
                                      'text': '',
                                      'x': 0.02,
                                      'y': 0.8},
                            'title': { 'title': 'Original spectra',
                                       'x': 0,
                                       'y': 0},
                            'yLim': {'yMax': 0.6, 'yMin': 0}},
                   'derivatives': { 'apply': True,
                                    'axisLabel': { 'x': 'Wavelength (nm)',
                                                   'y': 'derivate'},
                                    'text': { 'bandWidth': False,
                                              'samples': False,
                                              'skipStep': False,
                                              'text': 'Derivate',
                                              'x': 0.6,
                                              'y': 0.8},
                                    'title': { 'title': 'Derivates',
                                               'x': 0,
                                               'y': 0},
                                    'yLim': {'yMax': 0, 'yMin': 0}}
                                },
    paramD['featurePlot'] = { 'bins': 10,
                   'boxWhisker': True,
                   'histogram': True,
                   'savePng': True,
                   'screenDraw': True,
                   'singles': {'apply': True, 'figSize': {'x': 8, 'y': 6}},
                   'columns': { 'apply': True,
                                'figSize': {'x': 8, 'y': 8},
                                'ncolumns': 3},

                   'targetFeatureSymbols': { 'caco3_usda.a54_w.pct': { 'color': 'whitesmoke',
                                                                       'label': 'CaCo3',
                                                                       'unit': 'percent'},
                                             'cec_usda.a723_cmolc.kg': { 'color': 'seagreen',
                                                                         'label': 'Cation '
                                                                                  'Exc. '
                                                                                  'Cap.',
                                                                         'unit': 'mol*kg-1'},
                                             'cf_usda.c236_w.pct': { 'color': 'sienna',
                                                                     'label': 'Crane '
                                                                              'fraction',
                                                                     'unit': 'percent'},
                                             'clay.tot_usda.a334_w.pct': { 'color': 'tan',
                                                                           'label': 'Clay '
                                                                                    'cont.',
                                                                           'unit': 'percent'},
                                             'ec_usda.a364_ds.m': { 'color': 'dodgerblue',
                                                                    'label': 'Electric '
                                                                             'cond.',
                                                                    'unit': "ms*m'-1"},
                                             'k.ext_usda.a725_cmolc.kg': { 'color': 'lightcyan',
                                                                           'label': 'Potassion '
                                                                                    '(K)',
                                                                           'unit': 'mol*kg-1'},
                                             'n.tot_usda.a623_w.pct': { 'color': 'darkcyan',
                                                                        'label': 'Nitrogen '
                                                                                 '(N) '
                                                                                 '[tot]',
                                                                        'unit': 'percent'},
                                             'oc_usda.c729_w.pct': { 'color': 'dimgray',
                                                                     'label': 'Organic '
                                                                              'carbon '
                                                                              '(C)',
                                                                     'unit': 'percent'},
                                             'p.ext_usda.a274_mg.kg': { 'color': 'firebrick',
                                                                        'label': 'Phosphorus '
                                                                                 '(P)',
                                                                        'unit': 'mg*kg-1'},
                                             'ph.cacl2_usda.a481_index': { 'color': 'lemonchiffon',
                                                                           'label': 'pH '
                                                                                    '(CaCl)',
                                                                           'unit': 'pH'},
                                             'ph.h2o_usda.a268_index': { 'color': 'lightyellow',
                                                                         'label': 'pH '
                                                                                  '(H20)',
                                                                         'unit': 'pH'},
                                             'sand.tot_usda.c60_w.pct': { 'color': 'orange',
                                                                          'label': 'Sand '
                                                                                   'cont.',
                                                                          'unit': 'percent'},
                                             'silt.tot_usda.c62_w.pct': { 'color': 'khaki',
                                                                          'label': 'Silt '
                                                                                   'cont.',
                                                                          'unit': 'percent'}},
                   'targetFeatures': [ 'n.tot_usda.a623_w.pct',
                                       'oc_usda.c729_w.pct']}
            
    return (paramD)

def CreateArrangeParamJson(jsonFP, projFN, processstep):
    """ Create the default json parameters file structure, only to create template if lacking
    
        :param str dstrootFP: directory path 
        
        :param str jsonfolder: subfolder under directory path 
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
       
    if processstep.lower() in ['plot']:
    
        # Get the default import params
        paramD = PlotParams()
        
        # Set the json FPN
        jsonFPN = os.path.join(jsonFP, 'template_plot_ossl-spectra.json')
    
    if os.path.exists(jsonFPN):
        
        ExitMsgMsg(True)
    
    DumpAnyJson(paramD,jsonFPN)
    
    ExitMsgMsg(False)
    
def CheckMakeDocPaths(rootpath,arrangeddatafolder, jsonfolder, sourcedatafolder=False):
    """ Create the default json parameters file structure, only to create template if lacking
    
        :param str dstrootFP: directory path 
        
        :param str jsonfolder: subfolder under directory path 
    """

    if not os.path.exists(rootpath):
        
        exitstr = "The rootpath does not exists: %s" %(rootpath)
        
        exit(exitstr)
        
    if sourcedatafolder:
        
        srcFP = os.path.join(os.path.dirname(__file__),rootpath,sourcedatafolder)
            
        if not os.path.exists(srcFP):
            
            exitstr = "The source data path to the original OSSL data does not exists:\n %s" %(srcFP)
            
            exit(exitstr)
        
    dstRootFP = os.path.join(os.path.dirname(__file__),rootpath,arrangeddatafolder)
        
    if not os.path.exists(dstRootFP):
        
        os.makedirs(dstRootFP)
        
    jsonFP = os.path.join(dstRootFP,jsonfolder)
    
    if not os.path.exists(jsonFP):
        
        os.makedirs(jsonFP)
        
    return dstRootFP, jsonFP

def ReadImportParamsJson(jsonFPN):
    """ Read the parameters for importing OSSL data
    
    :param jsonFPN: path to json file
    :type jsonFPN: str
    
    :return paramD: parameters
    :rtype: dict
   """
            
    return ReadAnyJson(jsonFPN)
  
def ReadProjectFile(dstRootFP,projFN, jsonFP):
    '''
    '''
    projFPN = os.path.join(dstRootFP,projFN)

    if not os.path.exists(projFPN):

        exitstr = 'EXITING, project file missing: %s.' %(projFPN)
        
        exit( exitstr )

    infostr = 'Processing %s' %(projFPN)

    print (infostr)
    
    # Open and read the text file linking to all json files defining the project
    with open(projFPN) as f:

        jsonL = f.readlines()

    # Clean the list of json objects from comments and whithespace etc
    jsonProcessObjectL = [os.path.join(jsonFP,x.strip())  for x in jsonL if len(x) > 10 and x[0] != '#']
    
    return jsonProcessObjectL
    
def ReadPlotJson(jsonFPN):
    """ Read the parameters for plotting
    
    :param jsonFPN: path to json file
    :type jsonFPN: str
    
    :return paramD: parameters
    :rtype: dict
   """
    
    return ReadAnyJson(jsonFPN)

def LoadBandData(columns, SpectraD):
    """Read json data into numpy array and convert to pandas dataframe
    
        :returns: organised spectral data
        :rtype: pandas dataframe
    """
                                     
    n = 0
                   
    # Loop over the spectra
    for sample in SpectraD['spectra']:
                                
        if n == 0:
        
            spectraA = np.asarray(sample['signalMean'])
        
        else:
             
            spectraA = np.vstack( (spectraA, np.asarray(sample['signalMean']) ) )
        
        n += 1
                           
    return pd.DataFrame(data=spectraA, columns=columns)

def SpectraDerivativeFromDf(dataFrame,columns):
    """ Create 1st order derivates from spectral signals
    
        :returns: organised spectral derivates
        :rtype: pandas dataframe
    """

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

class Obj(object):
    ''' Convert json parameters to class objects
    '''
    
    def __init__(self, paramD):
        ''' Convert input parameters from nested dict to nested class object
        
            :param dict paramD: parameters 
        '''
        for k, v in paramD.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [Obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Obj(v) if isinstance(v, dict) else v)
       
    def _SetArrangeDefautls(self):
        ''' Set class object default data if missing
        '''
    
        if not hasattr(self, 'sitedata'):
            
            setattr(self, 'sitedata', [])
            
        sitedataMinL = ["id.layer_local_c",
                        "dataset.code_ascii_txt",
                        "longitude.point_wgs84_dd",
                        "latitude.point_wgs84_dd",
                        "location.point.error_any_m",
                        "layer.upper.depth_usda_cm",
                        "layer.lower.depth_usda_cm",
                        "id_vis","id_mir"]   
        
        for item in sitedataMinL:
            
            if not item in self.sitedata:
                
                self.sitedata.append(item)
        '''       
        self.visnirStep = int(self.input.visnirOutputBandWidth/ self.input.visnirInputBandWidth)
        
        self.mirStep = int(self.input.mirOutputBandWidth/ self.input.mirInputBandWidth)
        
        self.neonStep = int(self.input.neonOutputBandWidth/ self.input.neonInputBandWidth)
        '''

    def _SetPlotDefaults(self):
        ''' Set class object default data if required
        '''

        if self.spectraPlot.singles.figSize.x == 0:
            
            self.spectraPlot.singles.figSize.x = 8
            
        if self.spectraPlot.singles.figSize.y == 0:
            
            self.spectraPlot.singles.figSize.y = 6
            
            
        if self.spectraPlot.duals.figSize.x == 0:
            
            self.spectraPlot.duals.figSize.x = 8
            
        if self.spectraPlot.duals.figSize.y == 0:
            
            self.spectraPlot.duals.figSize.y = 8
            
        if len(self.featurePlot.targetFeatures) < self.featurePlot.columns.ncolumns:
            
            self.featurePlot.columns.ncolumns = len(self.featurePlot.targetFeatures)
                         
    def _SetPlotTextPos(self, plot, xmin, xmax, ymin, ymax):
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
    
    def _SetSoilLineDefautls(self):
        ''' Set class object default data if required
        '''
        
        if self.spectraPlot.singles.figSize.x == 0:
            
            self.spectraPlot.singles.figSize.x = 8
            
        if self.spectraPlot.singles.figSize.y == 0:
            
            self.spectraPlot.singles.figSize.y = 6

    def _SetModelDefaults(self):
        ''' Set class object default data if required
        '''
        
        if self.spectraPlot.singles.figSize.x == 0:
            
            self.spectraPlot.singles.figSize.x = 4
            
        if self.spectraPlot.singles.figSize.y == 0:
            
            self.spectraPlot.singles.figSize.y = 4
            
        # Check if Manual feature selection is set
        if self.manualFeatureSelection.apply:
            
            # Turn off the derivates alteratnive (done as part of the manual selection if requested)
            self.derivatives.apply = False
            
            # Turn off all other feature selection/agglomeration options
            self.globalFeatureSelection.apply = False
            
            self.modelFeatureSelection.apply = False
            
            self.featureAgglomeration.apply = False
                                 
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
        paramD.pop('spectraPlot')
                
        # Deep copy the parameters to self.spectraPlotD
        self.spectraPlotD = deepcopy(paramD)
               
        # Open and load JSON data file
        with open(self.input.jsonSpectraDataFilePath) as jsonF:
            
            self.jsonSpectraData = json.load(jsonF)
            
        # Open and load JSON parameter file
        with open(self.input.jsonSpectraParamsFilePath) as jsonF:
            
            self.jsonSpectraParams = json.load(jsonF)
                    
    def _SetcolorRamp(self,n):
        ''' Slice predefined colormap to discrete colors for each band
        '''
                        
        # Set colormap to use for plotting
        cmap = plt.get_cmap(self.spectraPlot.colorRamp)
        
        # Segmenting colormap to the number of bands
        self.slicedCM = cmap(np.linspace(0, 1, n)) 
                      
    def _GetBandData(self):
        ''' Read json data into numpy array and convert to pandas dataframe
        '''
                        
        self.SpectraDF = LoadBandData(self.columns, self.jsonSpectraData)
               
        if self.spectraPlot.derivatives.apply:
            
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

        rawPngFN = 'spectra.png' 

        self.rawPngFPN = os.path.join(plotRootFP, rawPngFN)
        
        derivativePngFN = 'derivative.png'

        self.derivativePngFPN = os.path.join(plotRootFP, derivativePngFN)
        
        dualPngFN = 'spectra+derivative.png'

        self.dualPngFPN = os.path.join(plotRootFP, dualPngFN)
        
        self.dualPngFPN = os.path.join(plotRootFP, dualPngFN)
                
        self.histogramPlotFPND = {}; self.boxwhiskerPlotFPND = {}
        
        for feature in self.featurePlot.targetFeatures:
            
            FN = 'histogram_%s.png' %(feature)
            
            self.histogramPlotFPND[feature] = os.path.join(plotRootFP, FN)
            
            FN = 'boxwhisker_%s.png' %(feature)
            
            self.boxwhiskerPlotFPND[feature] = os.path.join(plotRootFP, FN)
            
        self.histogramsPlotFPN = os.path.join(plotRootFP, 'histogram_all-features.png')
        
        self.boxwhiskersPlotFPN = os.path.join(plotRootFP, 'boxwhisker_all-features.png')
                   
    def _GetAbundanceData(self):
        ''' Get the abundance data
        '''
                
        # Get the list of substances included in this dataset
        substanceColumns = self.jsonSpectraParams['labData']
                
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
               
    def _histogramPlot(self):
        """ Plot target feature distributions as histograms
        """
                   
        for feature in self.featurePlot.targetFeatures:
            
            featureDf = self.abundanceDf[feature]

            tarFeat = getattr(self.featurePlot.targetFeatureSymbols, feature)
            
            ax = featureDf.plot.hist(bins=self.featurePlot.bins,color=tarFeat.color)
            
            ax.set_title(tarFeat.label)
            
            ax.set_xlabel(tarFeat.unit)
            
            if self.featurePlot.screenDraw:
            
                plt.show()
                
            if self.featurePlot.savePng:
                
                fig = ax.get_figure()
            
                fig.savefig(self.histogramPlotFPND[feature])   # save the figure to file
                       
        if len(self.featurePlot.targetFeatures) > 1 and (self.featurePlot.columns.apply):
            
            rows = ceil( len(self.featurePlot.targetFeatures) / self.featurePlot.columns.ncolumns )
            
            featureFig, featureAxs = plt.subplots( rows, self.featurePlot.columns.ncolumns, 
                                                         figsize=(self.featurePlot.columns.figSize.x, self.featurePlot.columns.figSize.y) )

            f = 0
            
            for feature in self.featurePlot.targetFeatures:
                
                row = floor(f / self.featurePlot.columns.ncolumns)
                                
                col = f-(row*self.featurePlot.columns.ncolumns)
            
                featureDf = self.abundanceDf[feature]
    
                tarFeat = getattr(self.featurePlot.targetFeatureSymbols, feature)
                
                if len(self.featurePlot.targetFeatures) <= self.featurePlot.columns.ncolumns:
                
                    featureDf.plot.hist(bins=self.featurePlot.bins,color=tarFeat.color, ax=featureAxs[col])
                    
                    featureAxs[col].set_title(tarFeat.label)
                
                    featureAxs[col].set_xlabel(tarFeat.unit)
                    
                    if col > 0:
                        
                        featureAxs[col].set(ylabel=None)
                
                else:
                    
                    featureDf.plot.hist(bins=self.featurePlot.bins,color=tarFeat.color, ax=featureAxs[row,col])

                    featureAxs[row,col].set_title(tarFeat.label)
                
                    featureAxs[row,col].set_xlabel(tarFeat.unit)
                
                    if col > 0:
                        
                        featureAxs[row,col].set(ylabel=None)
                
                f += 1
                
            if self.featurePlot.screenDraw:
  
                plt.show()
            
            if self.featurePlot.savePng:
            
                featureFig.savefig(self.histogramsPlotFPN)   # save the figure to file
            
    def _boxWhiskerPlot(self):
        """ Plot target feature distributions as Box-Whisker plots
        """
                   
        for feature in self.featurePlot.targetFeatures:
            
            tarFeat = getattr(self.featurePlot.targetFeatureSymbols, feature)
            
            ax = self.abundanceDf.boxplot(column=[feature], patch_artist = True, 
                                          boxprops = dict(facecolor = tarFeat.color))
                        
            ax.set_title(tarFeat.label)
            
            ax.set_xlabel(tarFeat.unit)
            
            if self.featurePlot.screenDraw:
            
                plt.show()
                
            if self.featurePlot.savePng:
                
                fig = ax.get_figure()
                        
                fig.savefig(self.boxwhiskerPlotFPND[feature])   # save the figure to file
            
        if len(self.featurePlot.targetFeatures) > 1 and (self.featurePlot.columns.apply):
            
            rows = ceil( len(self.featurePlot.targetFeatures) / self.featurePlot.columns.ncolumns )
            
            featureFig, featureAxs = plt.subplots( rows, self.featurePlot.columns.ncolumns, 
                                                         figsize=(self.featurePlot.columns.figSize.x, self.featurePlot.columns.figSize.y) )

            f = 0
            
            for feature in self.featurePlot.targetFeatures:
                
                row = floor(f / self.featurePlot.columns.ncolumns)
                                
                col = f-(row*self.featurePlot.columns.ncolumns)

                tarFeat = getattr(self.featurePlot.targetFeatureSymbols, feature)
                
                if len(self.featurePlot.targetFeatures) <= self.featurePlot.columns.ncolumns:
                
                    self.abundanceDf.boxplot(column=[feature], patch_artist = True, 
                                              boxprops = dict(facecolor = tarFeat.color),
                                              ax=featureAxs[col])
                                    
                    featureAxs[col].set_title(tarFeat.label)
                    
                    featureAxs[col].set_xlabel(tarFeat.unit)
                    
                    if col > 0:
                        
                        featureAxs[col].set(ylabel=None)
                        
                else:
                    
                    self.abundanceDf.boxplot(column=[feature], patch_artist = True, 
                                              boxprops = dict(facecolor = tarFeat.color),
                                              ax=featureAxs[row,col])
                                    
                    featureAxs[row,col].set_title(tarFeat.label)
                    
                    featureAxs[row,col].set_xlabel(tarFeat.unit)
                    
                    if col > 0:
                        
                        featureAxs[row,col].set(ylabel=None)
                
                f += 1
  
            if self.featurePlot.screenDraw:
            
                plt.show()
                
            if self.featurePlot.savePng:
            
                featureFig.savefig(self.boxwhiskersPlotFPN)   # save the figure to file
                                                                              
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
        plotskipStep = ceil( (len(self.SpectraDF.index)-1)/self.spectraPlot.maxSpectra )
                
        xLabel = plot.axisLabel.x
        
        yLabel = plot.axisLabel.y
        
        # Set the plot title, labels and annotation
        title, text = self._PlotTitleText(plot,plotskipStep)
               
        fig, ax = plt.subplots( figsize=(self.spectraPlot.figSize.x, self.spectraPlot.figSize.y)  )

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

        if self.spectraPlot.xLim.xMin:
                        
            ax.set_xlim(self.spectraPlot.xLim.xMin, self.spectraPlot.xLim.xMax)
            
        if plot.yLim.yMin:
                        
            ax.set_ylim(plot.yLim.yMin, plot.yLim.yMax)
          
        # Get the limits of the plot area - to fit the text 
        xyLimD = {}
         
        xyLimD['xMin'],xyLimD['xMax'] = ax.get_xlim()
        
        xyLimD['yMin'],xyLimD['yMax'] = ax.get_ylim()
        
        ax.set(xlabel=xLabel, ylabel=yLabel, title=title)
        
        if self.spectraPlot.legend:
                    
            ax.legend(loc=self.spectraPlot.legend)
        
        if text != None:
            
            x,y = self._SetPlotTextPos(plot, xyLimD['xMin'], xyLimD['xMax'], xyLimD['yMin'], xyLimD['yMax'])
            
            ax.text(x, y, text)
            
        # Set tight layout if requested
        if self.spectraPlot.tightLayout:
            
            fig.tight_layout()
                                        
        if self.spectraPlot.screenDraw:
        
            plt.show()
          
        if self.spectraPlot.savePng:
          
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
        plotskipStep = ceil( (len(self.SpectraDF.index)-1)/self.spectraPlot.maxSpectra )
                            
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(self.spectraPlot.duals.figSize.x, self.spectraPlot.duals.figSize.y), sharex=True  )
        
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
                     
        if self.spectraPlot.xLim.xMin:
                        
            ax[0].set_xlim(self.spectraPlot.xLim.xMin, self.spectraPlot.xLim.xMax)
            
            ax[1].set_xlim(self.spectraPlot.xLim.xMin, self.spectraPlot.xLim.xMax)
            
        if self.spectraPlot.raw.yLim.yMin:
                        
            ax[0].set_ylim(self.spectraPlot.raw.yLim.yMin,self.spectraPlot.raw.yLim.yMax)
            
        if self.spectraPlot.derivatives.yLim.yMin:
                        
            ax[1].set_ylim(self.spectraPlot.derivatives.yLim.yMin, self.spectraPlot.derivatives.yLim.yMax)
          
        # Get the limits of the plot areas - to fit the text 
        rawxyLimD = {}; derivativexyLimD = {}
         
        rawxyLimD['xMin'],rawxyLimD['xMax'] = ax[0].get_xlim()
        
        rawxyLimD['yMin'],rawxyLimD['yMax'] = ax[0].get_ylim()
        
        derivativexyLimD['xMin'],derivativexyLimD['xMax'] = ax[1].get_xlim()
        
        derivativexyLimD['yMin'],derivativexyLimD['yMax'] = ax[1].get_ylim()
        
        ax[0].set(ylabel=self.spectraPlot.raw.axisLabel.y, title=self.spectraPlot.raw.title.title)
        
        ax[1].set(xlabel=self.spectraPlot.raw.axisLabel.x, ylabel=self.spectraPlot.derivatives.axisLabel.y, 
                  title=self.spectraPlot.derivatives.title.title)
        
        if self.spectraPlot.legend:
                    
            ax[0].legend(loc=self.spectraPlot.legend)
        
        rawtext = self._PlotTitleText(self.spectraPlot.raw,plotskipStep)[1]
        
        if self.spectraPlot.raw.text != None:
               
            x,y = self._SetPlotTextPos(self.spectraPlot.raw, rawxyLimD['xMin'], rawxyLimD['xMax'], rawxyLimD['yMin'], rawxyLimD['yMax'])
            
            ax[0].text(x, y, rawtext)
            
        derivativetext = self._PlotTitleText(self.spectraPlot.derivatives,0)[1]
        
        if self.spectraPlot.derivatives.text != None:
               
            x,y = self._SetPlotTextPos(self.spectraPlot.derivatives, derivativexyLimD['xMin'], derivativexyLimD['xMax'], derivativexyLimD['yMin'], derivativexyLimD['yMax'])
            
            ax[1].text(x, y, derivativetext)
            
        # Set supTitle
        if self.spectraPlot.supTitle:
            
            if self.spectraPlot.supTitle == "auto":
                
                supTitle = 'Project: %s' %(self.name)
            
            else:
                
                supTitle = self.spectraPlot.supTitle
            
            fig.suptitle(supTitle)
   
        # Set tight layout if requested
        if self.spectraPlot.tightLayout:
            
            fig.tight_layout()
                                        
        if self.spectraPlot.screenDraw:
        
            plt.show()
          
        if self.spectraPlot.savePng:
          
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
        
        if self.spectraPlot.apply:
            
            # Get the band data
            self._GetBandData()
        
            # plot the spectra
            if self.spectraPlot.raw.apply and self.spectraPlot.derivatives.apply:
            
                self._PlotDualMulti(self.dualPngFPN )
                
            elif self.spectraPlot.derivatives.apply:
    
                x = [int(i[1:len(i)]) for i in self.derivativeColumns]
                
                self._PlotMonoMulti(self.spectraDerivativeDF, x, self.spectraPlot.derivatives, self.derivativePngFPN )
            
            else:    
                
                x = [int(i) for i in self.columns]
                
                self._PlotMonoMulti(self.SpectraDF, x, self.spectraPlot.raw, self.rawPngFPN )
            
        if len(self.featurePlot.targetFeatures) > 0 and (self.featurePlot.histogram or self.featurePlot.boxWhisker):
            
            self._GetAbundanceData()
            
            if self.featurePlot.histogram:
            
                self._histogramPlot()
                
            if self.featurePlot.boxWhisker:
            
                self._boxWhiskerPlot()
            
def SetupProcesses(iniParams):
    '''Setup and loop processes
    
    :param docpath: path to project root folder 
    :type: lstr
    
    :param sourcedatafolder: folder name of original OSSL data (source folder)  
    :type: lstr
    
    :param arrangeddatafolder: folder name of arranged OSSL data (destination folder) 
    :type: lstr
            
    :param projFN: project filename (in destination folder)
    :type: str
    
    :param jsonfolder: folder name
    :type: str
            
    '''

    dstRootFP, jsonFP = CheckMakeDocPaths(iniParams['rootpath'],
                                          iniParams['arrangeddatafolder'], 
                                          iniParams['jsonfolder'], 
                                          iniParams['sourcedatafolder'])
    
    if iniParams['createjsonparams']:
        
        CreateArrangeParamJson(jsonFP,iniParams['projFN'],'plot')
        
    jsonProcessObjectL = ReadProjectFile(dstRootFP, iniParams['projFN'], jsonFP)
           
    #Loop over all json files and create Schemas and Tables
    for jsonObj in jsonProcessObjectL:
        
        print ('    jsonObj:', jsonObj)

        paramD = ReadPlotJson(jsonObj)
        
        '''
        pp = pprint.PrettyPrinter(indent=2)

        pp.pprint(paramD)
        
        '''
        # Invoke the plot
        spectraPlt = SpectraPlot(paramD)
        
        # Set the dst file names
        spectraPlt._SetDstFPNs()
        
        # Run the plotting
        spectraPlt._PilotPlot()

                                         
if __name__ == '__main__':
    ''' If script is run as stand alone
    '''
    
    '''
    if len(sys.argv) != 2:
        
        sys.exit('Give the link to the json file to run the process as the only argument')
    
    #Get the json file
    jsonFPN = sys.argv[1]
    
    if not os.path.exists(jsonFPN):
        
        exitstr = 'json file not found: %s' %(jsonFPN)
    '''   
    jsonFPN = "/Users/thomasgumbricht/docs-local/OSSL/plot_ossl.json"
    
    iniParams = ReadAnyJson(jsonFPN)

    SetupProcesses(iniParams)