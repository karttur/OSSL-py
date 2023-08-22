'''
Created on 8 Sep 2022

Edited on 27 Sep 2022

Last edited 22 Feb 2023

@author: thomasgumbricht

Notes
-----
The module OSSL.py:

    requires that you have downloaded and exploded a standard zip-archive from OSSL. The exploded folder should be renamed to
    reflect its geographic and thematic content. This script then expects the exploded folder to contain the following 4 csv 
    files (in alphabetic order):
        
        - mir.data.csv 
        - soillab.data.csv
        - soilsite.data.csv
        - visnir.data.csv
     
    The script takes 3 string parameters as input:
    
        - docpath: the full path to a folder that must contain the txt file as given by the "projFN" parameter
        - projFN: the name of an existing txt files that sequentially lists json parameter files to run
        - jsonpath: the relative path (vis-a-vis "docpath") where the json parameter files (listed in "projFN") are 
    
    The parameter files must list approximately 40 parameters in a precise nested json structure with dictionaries and lists.
    You can create a template json parameter file by running "def CreateParamJson" (just uncomment under "def SetupProcesses",
    this creates a template json parameter file called "import_ossl-spectra.json" in the path given as the parameter "docpath".
    
    With an edited json parameter file pointing at the downloaded and exploded folder (parameter: rootFP), the script reads the
    csv fiels and imports the data as requested in the json parameter file. The script first run the stand alone "def SetupProcesses" 
    that reads the txt file "projFN" and then sequentialy run the json parameter files listed. 
    
    Each import (i.e. each json parameter file is run as a separate instance of the class "ImportOSSL". 
    
    Each import process result in 2 or 4 files, 2 files if either visible and near infrared (visnir) or mid infrared (mir) data are 
    imported, and 4 if both visnir and mir are imported.
    
    The names of the destination files cannot be set by the user, they are defaulted as follows,
    
    visnir result files:
    
        - parameters: "rootFP"/visnirjson/params-visnir_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width" 
        - data: "rootFP"/visnirjson/data-visnir_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"
    
    mir result files:
    
        - parameters: "rootFP"/mirjson/params-visnir_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"
        - data: "rootFP"/mirjson/data-visnir_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"
        
'''

# Standard library imports
import os

import json

import csv

from copy import deepcopy

# Third party imports
import numpy as np

def ImportParams():
    """ Default template parameters for importing OSSL csv data
    
        :returns: parameter dictionary
        
        :rtype: dict
    """
    
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
    
    paramD['input']['rootFP'] = '/path/to/folder/with/downloaded/ossl/data'
    
    paramD['input']['visnirSubFP'] = 'visnirjson'
    
    paramD['input']['mirSubFP'] = 'mirjson'
        
    paramD['input']['spectraunits'] = 'percent'
    
    paramD['input']['visnir'] = True
    
    paramD['input']['mir'] = True
    
    paramD['input']['visnirBegin'] = 460
    
    paramD['input']['visnirEnd'] = 2500
    
    paramD['input']['visnirStep'] = 2
    
    paramD['input']['visnirInputBandWidth'] = 2
    
    paramD['input']['visnirOutputBandWidth'] = 10
    
    paramD['input']['mirBegin'] = 460
    
    paramD['input']['mirEnd'] = 460
    
    paramD['input']['mirStep'] = 2
    
    paramD['input']['percentOK'] = 70
    
    paramD['input']['labData'] = ['c.tot_usda.4h2_wpct','ca.ext_usda.4b1_cmolkg','caco3_usda.4e1_wpct',
                'oc_usda.calc_wpct', 'cec.ext_usda.4b1_cmolkg','clay.tot_usda.3a1_wpct',
                'ecec_usda.4b4_cmolkg','ec.w_usda.4f1_dsm','fe.dith_usda.4g1_wpct']
            
    paramD['input']['labDataRange'] = {'oc_usda.calc_wpct':{'min':0,'max':10}}
        
    return (paramD)

def CreateParamJson(docpath):
    """ Create the default json parameters file structure, only to create template if lacking
    
        :param str docpath: directory path 
    """
    
    # Get the default params
    paramD = ImportParams()
    
    # Set the json FPN
    jsonFPN = os.path.join(docpath, 'template_import_ossl-spectra.json')
    
    # Dump the paramD as a json object
    
    jsonF = open(jsonFPN, "w")
  
    json.dump(paramD, jsonF, indent = 2)
  
    jsonF.close()

def ReadImportParamsJson(jsonFPN):
    """ Read the parameters for importing OSSL data
    
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
        ''' Convert input parameters from nested dict to nested class object
        
            :param dict paramD: parameters 
        '''
        for k, v in paramD.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [Obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Obj(v) if isinstance(v, dict) else v)
                
    def _SetDefautls(self):
        ''' Set class object default data if missing
        '''
    
        if not hasattr(self, 'sitedata'):
            
            setattr(self, 'sitedata', [])
            
        sitedataMinL = ["longitude_wgs84_dd",
                        "latitude_wgs84_dd",
                        "location.error_any_m",
                        "layer.upper.depth_usda_cm",
                        "layer.lower.depth_usda_cm",
                        "id_vis","id_mir"]   
        
        for item in sitedataMinL:
            
            if not item in self.sitedata:
                
                self.sitedata.append(item)
                
        self.input.visnirStep = int(self.input.visnirOutputBandWidth/ self.input.visnirInputBandWidth)
    
class ImportOSSL(Obj):
    ''' import soil spectra from OSSL to xspectre json format
    '''
    
    def __init__(self,paramD): 
        ''' Initiate import OSSl class
        
        :param dict param: parameters
        '''
        
        # convert the input parameter dict to class objects
        Obj.__init__(self,paramD)
            
        # Set class object default data if missing 
        self._SetDefautls()
        
        # Deep copy parameters to a new obejct class called params
        self.params = deepcopy(self)
                              
    def _SetSrcFPNs(self):
        ''' Set source file paths and names
        '''
        # All OSSL data are download as a zipped subfolder with data given standard names as of below
                    
        self.srcVISNIRFPN = os.path.join(self.params.input.rootFP,'visnir.data.csv')
        
        self.srcMIRFPN = os.path.join(self.params.input.rootFP,'mir.data.csv')
        
        self.srcSoilLabFPN = os.path.join(self.params.input.rootFP,'soillab.data.csv')
        
        self.srcSoilSiteFPN = os.path.join(self.params.input.rootFP,'soilsite.data.csv')
                
    def _SetDstFPNs(self):
        ''' Set destination file paths and names
        '''
        
        visnirFP = os.path.join(self.params.input.rootFP, self.params.input.visnirSubFP)
        
        if not os.path.exists(visnirFP):
            
            os.makedirs(visnirFP)
         
        modelN = '%s_%s-%s_%s' %(os.path.split(self.params.input.rootFP)[1], 
                    self.params.input.visnirBegin,self.params.input.visnirEnd,int(self.params.input.visnirStep*2))

        visnirParamFN = 'params-visnir_%s.json' %(modelN)
        
        self.visnirParamFPN = os.path.join(visnirFP, visnirParamFN)
        
        visnirDataFN = 'data-visnir_%s.json' %(modelN)
            
        self.visnirDataFPN = os.path.join(visnirFP, visnirDataFN)
                  
    def _ReadCSV(self, FPN):
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
        
    def _DumpVISNIRJson(self, exportD):
        ''' Export, or dump, the imported VINSNIR OSSL data as json objects
        
        :param exportD: formatted dictionary
        :type exportD: dict
        '''
                
        jsonF = open(self.visnirDataFPN, "w")
      
        json.dump(exportD, jsonF, indent = 2)
      
        jsonF.close()
        
        D = json.loads(json.dumps(self.params, default=lambda o: o.__dict__))
        
        jsonF = open(self.visnirParamFPN, "w")
        
        json.dump(D, jsonF, indent = 2)
        
        jsonF.close()
                
    def _ExtractSiteData(self, headers, rowL):
        ''' Exract the site data (ossl file: "soilsite.data.csv")
        
            :paramn headers: list of columns
            :type: list
            
            :param rowL: array of data
            :rtype: list of list
        '''

        self.siteD = {}
        
        for row in rowL:
            
            self.siteD[ row[1] ] = {}
        
            for item in self.sitedata:
                
                colNr = headers.index(item)
                            
                self.siteD[ row[1] ][item] = row[colNr]
                
            # Check if site is inside depth limits       
            if float(self.siteD[ row[1] ]["layer.upper.depth_usda_cm"]) < self.soilsample.mindepth  or float(self.siteD[ row[1] ]["layer.lower.depth_usda_cm"]) > self.soilsample.maxdepth:
                
                self.siteD[ row[1] ]["id_vis"] = "FALSE"
                
                self.siteD[ row[1] ]["id_mir"] = "FALSE"
        
    def _ExtractLabData(self, headers, rowL):
        ''' Extract the key lab data required (ossl file: "soillab.data.csv")
        
            :paramn headers: list of columns
            :type: list
            
            :param rowL: array of data
            :rtype: list of list
        '''
        
        self.labD = {}

        for row in rowL:
            
            self.labD[ row[0] ] = [] 
            
            skip = False
        
            for item in self.params.input.labData:
                
                colNr = headers.index(item)
                
                #if item in self.params.input.labDataRange:
                if hasattr(self.params.input.labDataRange,item):
                    
                    if row[colNr] != 'NA':
                        
                        itemRange = getattr(self.params.input.labDataRange,item)
                                                
                        if  float(row[colNr]) < itemRange.min or float(row[colNr]) > itemRange.max:
                            
                            skip = True  
    
            # Loop again, only accept items that are not skipped
            for item in self.params.input.labData:
                
                colNr = headers.index(item)
                
                if not skip:
                    
                    try:
                        
                        # Only if a numerical value is given
                        self.labD[ row[0] ].append( {'substance': item, 'value':float(row[colNr]) } ) 
                           
                    except:
                        
                        # Otherwise skip this lab parameter for this site
                        pass
         
    def _AverageSpectra(self, spectrA, n):
        ''' Agglomerate high resolution spectral signals to broader bands
        
            :paramn spectrA: array of spectral signsals
            :type: np array
            
            :param n: samples to agglomerate
            :rtype: int
        '''
        
        cum = np.cumsum(spectrA,0)
        
        result = cum[n-1::n]/float(n)
        
        result[1:] = result[1:] - result[:-1]
    
        remainder = spectrA.shape[0] % n
        
        if remainder != 0:
            
            pass

        return result
          
    def _ExtractVISNIRSpectraData(self, headers, rowL):
        ''' Extract VISNIR spectra from OSSL csv (ossl file: "visnor.data.csv")
        
            :paramn headers: list of columns
            :type: list
            
            :param rowL: array of data
            :rtype: list of list
        '''
        
        self.VISNIRspectraD = {}
        
        nsamples = len(rowL)
        
        nloop = 0
        
        visnirBegin = self.params.input.visnirBegin
        
        visnirEnd = self.params.input.visnirEnd
        
        # Iteratively loop over the spectra to get a minimum percent of OK data
        while True:
            
            nskip = 0
            
            for row in rowL:
                
                id_layer_uuid_c = row[2]
                                    
                if self.siteD[id_layer_uuid_c]['id_vis'] == 'TRUE':
                                    
                    # Get all the spectra as a numpy vector
                    mincol = int( 16+(visnirBegin-350)/2 )
                    
                    maxcol = int( 1092-(2500-visnirEnd)/2 )
                    
                    if 'NA' in row[mincol:maxcol]:
                        
                        nskip += 1
                        
                        continue
                
                else:
                    
                    nskip += 1
                           
            percentOK = 100*(nsamples-nskip)/nsamples
            
            if percentOK > self.params.input.percentOK:
                
                break
            
            nloop += 1
            
            if nloop % 2 == 0:
                
                # check the first column for NA, and only if it contains NA shave it off
                if 'NA' in row[mincol:mincol+1]:
                
                    visnirBegin += self.params.input.visnirStep*2
                
            else:
                
                # check the last column for NA, and only if it contains NA shave it off
                if 'NA' in row[maxcol-1:maxcol]:
                    
                    visnirEnd -= self.params.input.visnirStep*2
          
        # Reset the range of the spectra to import to contain the required fraction of true values   
        self.params.input.visnirBegin = visnirBegin
        
        self.params.input.visnirEnd = visnirEnd
                
        # Loop over the spectra                        
        for row in rowL:
                
            id_layer_uuid_c = row[2]
                                
            if self.siteD[id_layer_uuid_c]['id_vis'] == 'TRUE':
                                
                # Get all the spectra as a numpy vector
                mincol = int( 16+(self.params.input.visnirBegin-350)/2 )
                
                maxcol = int( 1092-(2500-self.params.input.visnirEnd)/2 )
                

                # discard all data that contains NA
                if 'NA' in row[mincol:maxcol]:
                
                    self.siteD[id_layer_uuid_c]['id_vis'] = 'FALSE'
                    
                    continue
                
                visnirSpectrA = np.asarray(row[mincol:maxcol]).astype(float)
                            
                spectraA = self._AverageSpectra(visnirSpectrA, self.params.input.visnirStep)
                
                self.VISNIRspectraD[id_layer_uuid_c] = spectraA
                
                
                self.numberOfwl = spectraA.shape[0]
                

            
    def _AssembleVISNIRJsonD(self):
        ''' Convert the extracted data to json objects for export
        '''
    
        modname = '%s_%s-%s_%s' %(os.path.split(self.params.input.rootFP)[1], 
                    self.params.input.visnirBegin,self.params.input.visnirEnd,int(self.params.input.visnirStep*2))
        
        exportD = {'model': modname}
        
        if self.params.input.visnirStep == 1:
            
            wl = [i for i in range(self.params.input.visnirBegin, self.params.input.visnirEnd+1, self.params.input.visnirStep*2)]
            
        else:
            
            wl = [i+self.params.input.visnirStep for i in range(self.params.input.visnirBegin, self.params.input.visnirEnd, self.params.input.visnirStep*2)]
                
        # Reduce wl if bands are cut short while averaging
        wl = wl[0:self.numberOfwl]
                
        exportD['bands'] = wl
            
        varLD = []
        
        for site in self.siteD:
                    
            if self.siteD[site]['id_vis'] == 'TRUE':
                                                             
                jsonD = {'name':site}
                
                jsonD['latitude_dd'] = self.siteD[site]['longitude_wgs84_dd']
                
                jsonD['longitude_dd'] = self.siteD[site]['latitude_wgs84_dd']
                                
                jsonD['mindepth'] = self.siteD[site]['layer.upper.depth_usda_cm']
                
                jsonD['maxdepth'] = self.siteD[site]['layer.lower.depth_usda_cm']
                
                jsonD['mean'] = self.VISNIRspectraD[site].tolist()
                
                jsonD['abundance'] = self.labD[site]
                
                varLD.append(jsonD)
                
        exportD['labspectra'] = varLD
                      
        # export, or dump, the assembled json objects      
        self._DumpVISNIRJson(exportD)
        
                                                 
    def PilotImport(self):
        ''' Steer the sequence of processes for extracting OSSL csv data to json objects
        ''' 
        
        # Set the source file names
        self._SetSrcFPNs()
        
        headers, rowL = self._ReadCSV(self.srcSoilSiteFPN)
        
        self._ExtractSiteData(headers, rowL)
        
        headers, rowL = self._ReadCSV(self.srcSoilLabFPN)
        
        self._ExtractLabData(headers, rowL)     
                
        if self.params.input.visnir:
      
            headers, rowL = self._ReadCSV(self.srcVISNIRFPN)

            self._ExtractVISNIRSpectraData(headers, rowL)
            
            # Set the sdestination file names - must be done after _ExtractVISNIRSpectraData 
            self._SetDstFPNs()
            
            self._AssembleVISNIRJsonD()
                                                      
def SetupProcesses(docpath, projFN, jsonpath):
    '''Setup and loop processes
    
    :param docpath: path to text file 
    :type: lstr
            
    :param projFN: project filename
    :type: str
    
    :param jsonpath: path to directory
    :type: str
            
    '''
    
    ''' CreateParamJson creates the default template json structure for running the python script, 
    only use it to create a backbone, then edit to fit the project you are working with 
    '''
    CreateParamJson(docpath)
    BALLE
        
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
                
        paramD = ReadImportParamsJson(jsonObj)
        
        # Invoke the import
        ossl = ImportOSSL(paramD)
        
        ossl.PilotImport()
                    
if __name__ == "__main__":
    ''' If script is run as stand alone
    '''
    
    docpath = '/Users/thomasgumbricht/OSSL/se/OSSL_se-oster+vaster-goetland_20220907/rawdata'
    
    projFN = 'extract_rawdata.txt'
    
    jsonpath = ''
     
    SetupProcesses(docpath, projFN, jsonpath)
    