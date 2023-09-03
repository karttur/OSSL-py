#!/Applications/anaconda3/envs/spectraimagine_py38/bin/python3.8
'''
Created on 8 Sep 2022

Edited on 27 Sep 2022

Edited 22 Feb 2023

Edited 7 August 2023

Last edited 30 August 2023

@author: thomasgumbricht

Notes
-----
The module OSSL_import.py:

    requires that you have downloaded and exploded a standard zip-archive from OSSL 
    (see https://karttur.github.io/soil-spectro/libspectrodata/spectrodata-OSSL-api-explorer/). 
    Create a folder for the zip file and rename it to reflect the geographic and/or thematic content. 
    When unzipping the zip file the content will be exploded into a subfolder called "data".
    The subolder "data" should contain 5 csv files  (in alphabetic order):
        
        - mir.data.csv 
        - neon.data.csv
        - soillab.data.csv
        - soilsite.data.csv
        - visnir.data.csv
     
    The script takes a single input parameter - the path (string) 
    to a json file with the following structure:
    
        {
          "rootpath": full/path/to/folder/where/you/saved/OSSL.zip,
          "sourcedatafolder": "data",
          "arrangeddatafolder": "arranged-data", 
          "jsonfolder": "json-import", 
          "projFN": "extract_rawdata.txt", 
          "createjsonparams": false
        }
    
        - rootpath: full path to folder with a downloaded OSSL zip file; parent folder to the exploded OSSL subfolder ("data")
        - sourcedatafolder: subfolder under "rootpath" with the exploded content of the OSSL zip file (default = "data")
        - arrangeddatafolder: subfolder under "rootpath" where the imported (rearranged) OSSL data will be stored 
        - projFN: the name of an existing txt files that sequentially lists json parameter files to run
        - jsonfolder: the relative path (vis-a-vis "rootpath") where the json parameter files (listed in "projFN") are located 
        - createjsonparams: if set to true the script will create a template json file and exit
    
    The parameter files must list approximately 40 parameters in a precise nested json structure with dictionaries and lists.
    You can create a template json parameter file by running "def CreateParamJson" (just uncomment under "def SetupProcesses",
    this creates a template json parameter file called "import_ossl-spectra.json" in the path given as the parameter "rootpath".
    
    With an edited json parameter file pointing at the downloaded and exploded folder (parameter: rootFP), the script reads the
    files and imports the data as requested in the json parameter file. The script first run the stand alone "def SetupProcesses" 
    that reads the txt file "projFN" and then sequentially run the json parameter files listed. 
    
    Each import process results in 2 files for each wavelength band (mir/neon/visnir) that are 
    set for import in the json command file.
    
    The names of the destination files cannot be set by the user, they are defaulted as follows,
    
        parameters: "rootFP"/"wavelength bands"/params-"wavelength-bands"_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"
                "rootFP"/arranged-data/visnir/"project_name"_"first wavelength"-"last wavelength"_"band width"/
                data-visnir_"LastPartoOfrootFP"_"first wavelength"-"last wavelength"_"band width".json
                
        data: "rootFP"/"wavelength bands"/data-"wavelength-bands"_OSSL_"region"_"date"_"first wavelength"-"last wavelength"_"band width"
                "rootFP"/arranged-data/visnir/"project_name"_"first wavelength"-"last wavelength"_"band width"/
                data-visnir_"LastPartoOfrootFP"_"first wavelength"-"last wavelength"_"band width".json
  
'''

# Standard library imports
import sys 

import os

import json

import datetime

from copy import deepcopy

import pprint

import csv

# Third party imports
import numpy as np


def Today():
    
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
    
    paramD['rootFP'] = '/path/to/folder/with/ossl/download'
    
    return paramD

def StandardXspectreParams(): 
    """ Default standard parameters for importing xSpectre spectral data
    
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

    
    
    '''
    
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
    '''
    
    paramD['soilSample'] = {'maxDepth': 20, 'minDepth': 0}
    
    paramD['visnir'] = { 'apply': True,
              'beginWaveLength': 460,
              'endWaveLength': 1050,
              'inputBandWidth': 2,
              'outputBandWidth': 10,
              'subFP': 'visnir'}
        
    paramD['neon'] = { 'apply': False,
            'beginWaveLength': 1350,
            'endWaveLength': 2550,
            'inputBandWidth': 2,
            'outputBandWidth': 10,
            'subFP': 'neon'}
        
    paramD['mir'] = { 'apply': False,
           'beginWaveLength': 2500,
           'endWaveLength': 8000,
           'inputBandWidth': 2,
           'outputBandWidth': 10,
           'subFP': 'mir'}  

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
    
    paramD['labDataRange'] =  {'caco3_usda.a54_w.pct': {'max': 10, 'min': 0}}
    
    #paramD['labDataRange'] = {}
    
    #paramD['labDataRange']['caco3_usda.a54_w.pct'] = {
    #    "min": 0,
    #    "max": 10}

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

def CreateArrangeParamJson(jsonFP, projFN, processstep):
    """ Create the default json parameters file structure, only to create template if lacking
    
        :param str jsonFP: directory path 
    
        :param str jsonFP: file name 
        
        :param str processstep: the name of the processtep for which to create a template 
    """
    
    def ExitMsgMsg(flag):
        """ Exit message from CreateArrangeParamJson
        
            :param str flag: True if template file already existed; false if template was created 
        """
        
        if flag:
            
            exitstr = 'json parameter file already exists: %s\n' %(jsonFPN)
        
        else:
        
            exitstr = 'json parameter file created: %s\n' %(jsonFPN)
        
        exitstr += ' Edit the json file for your project and rename it to reflect the commands.\n' 
        
        exitstr += ' Add the path of the edited file to your project file (%s).\n' %(projFN)
        
        exitstr += ' Then set createjsonparams to false and rerun script.'
        
        exit(exitstr)
       
    if processstep.lower() in ['import','arrange']:
    
        # Get the default import params
        paramD = ImportParams()
        
        # Set the json FPN
        jsonFPN = os.path.join(jsonFP, 'template_import_ossl-spectra.json')
        
    if processstep.lower() in ['importxspectre','arrangexspectre']:
    
        pass
        '''
        # Get the default import params
        paramD = ImportXspectreParams()
        
        # Set the json FPN
        jsonFPN = os.path.join(jsonFP, 'template_import_xspectre-spectra.json')
        
        # Set the json FPN
        jsonFPN = os.path.join(jsonFP, 'template_model_ossl-spectra.json')
        '''
    
    if os.path.exists(jsonFPN):
        
        ExitMsgMsg(True)
    
    DumpAnyJson(paramD,jsonFPN)
    
    ExitMsgMsg(False)
    
def CheckMakeDocPaths(rootpath,arrangeddatafolder, jsonfolder, sourcedatafolder=False):
    """ Create the folder and file structure
    
        :param str rootpath: root directory path 
        
        :param str arrangeddatafolder: the destination subfolder under rootpath for all processed data
        
        :param str jsonfolder: subfolder under arrangeddatafolder where the json command files are located 
        
        :param bool sourcedatafolder: if True a template json command will be creates, otherwise ignored
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
    
    :param str jsonFPN: path to json file
    
    :return paramD: parameters
    :rtype: dict
   """
            
    return ReadAnyJson(jsonFPN)
  
def ReadProjectFile(dstRootFP,projFN, jsonFP):
    """ Read the project file (txt file)
    
        :param str dstRootFP: destination root directory path 
        
        :param str projFN: project file name
        
        :param str jsonFP: subfolder uder dstRootFP where projFN is stored     
    """
           
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
        ''' Set import default data
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

    def _SetPlotDefaults(self):
        ''' Set plot default data
        '''

        if self.plot.singles.figSize.x == 0:
            
            self.plot.singles.figSize.x = 8
            
        if self.plot.singles.figSize.y == 0:
            
            self.plot.singles.figSize.y = 6
                         
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
        ''' Set soil line default data
        '''
        
        if self.plot.singles.figSize.x == 0:
            
            self.plot.singles.figSize.x = 8
            
        if self.plot.singles.figSize.y == 0:
            
            self.plot.singles.figSize.y = 6

    def _SetModelDefaults(self):
        ''' Set machine learning default data
        '''
        
        if self.plot.singles.figSize.x == 0:
            
            self.plot.singles.figSize.x = 4
            
        if self.plot.singles.figSize.y == 0:
            
            self.plot.singles.figSize.y = 4
            
        # Check if Manual feature selection is set
        if self.manualFeatureSelection.apply:
            
            # Turn off the derivates alteratnive (done as part of the manual selection if requested)
            self.derivatives.apply = False
            
            # Turn off all other feature selection/agglomeration options
            self.globalFeatureSelection.apply = False
            
            self.modelFeatureSelection.apply = False
            
            self.featureAgglomeration.apply = False
 
class ImportOSSL(Obj):
    ''' import soil spectra from OSSL to xSpectre json format
    '''
    
    def __init__(self,paramD): 
        ''' Initiate import OSSl class
        
        :param dict param: input parameters
        '''
        
        # convert the input parameter dict to class objects
        Obj.__init__(self,paramD)
            
        # Set class object default data if missing 
        self._SetArrangeDefautls()
        
        # Deep copy parameters to a new obejct class called params
        self.params = deepcopy(self)
                              
    def _SetSrcFPNs(self, sourcedatafolder):
        ''' Set source file paths and names
        '''
        
        # All OSSL data are download as a zipped subfolder with data given standard names as of below             
        self.srcVISNIRFPN = os.path.join(self.params.rootFP,sourcedatafolder,'visnir.data.csv')
        
        self.srcMIRFPN = os.path.join(self.params.rootFP,sourcedatafolder,'mir.data.csv')
        
        self.srcNEONFPN = os.path.join(self.params.rootFP,sourcedatafolder,'neon.data.csv')
        
        self.srcSoilLabFPN = os.path.join(self.params.rootFP,sourcedatafolder,'soillab.data.csv')
        
        self.srcSoilSiteFPN = os.path.join(self.params.rootFP,sourcedatafolder,'soilsite.data.csv')
        
        
    def _SetProjectNameId(self):
        ''' Set project name and id if they are not user defined
        '''
        
        if self.params.name in ['auto','']:
            
            self.params.name = self.campaign.campaignShortId
            
        if self.params.id in ['auto','']:
            
            self.params.id = '%s_%s' %(self.params.name,Today())
                         
    def _SetDstFPN(self, dstRootFP, band, subFP):
        ''' Set destination file paths and names
        
            :param str dstRootFP: destination root directory path
            
            :param str band: the spectral region of the destination data
            
            :param str subfp: the subfolder of the destination data
            
        '''

        # Get the band [visnir, mir , neon] object
        bandObject = getattr(self, band)
        
        beginWaveLength = getattr(bandObject, 'beginWaveLength')
        
        endWaveLength = getattr(bandObject, 'endWaveLength')
        
        inputBandWidth = getattr(bandObject, 'inputBandWidth')
        
        outputBandWidth = getattr(bandObject, 'outputBandWidth')
                
        # Calculate the column and wavelength step
        columnsStep = int(outputBandWidth / inputBandWidth)
        
        wlStep = int(columnsStep*inputBandWidth)
        
        projectSubFP = '%s_%s-%s_%s' %(self.params.name, 
                        beginWaveLength, endWaveLength, wlStep)
        
        FP = os.path.join(dstRootFP, subFP,projectSubFP)
            
        if not os.path.exists(FP):
                
            os.makedirs(FP)
 
        modelN = '%s_%s-%s_%s' %(os.path.split(self.params.rootFP)[1], 
                        beginWaveLength, endWaveLength, wlStep)
            
        paramFN = 'params-%s_%s.json' %(band, modelN)
        
        paramFPN = os.path.join(FP, paramFN)
            
        dataFN = 'data-%s_%s.json' %(band, modelN)
                
        dataFPN = os.path.join(FP, dataFN)
                
        return (modelN, paramFPN, dataFPN, columnsStep, wlStep)
                    
    def _DumpSpectraJson(self, exportD, dataFPN, paramFPN, band):
        ''' Export, or dump, the imported VINSNIR OSSL data as json objects
        
        :param exportD dict: formatted dictionary
        
        :param dataFPN str: full path to destination json datafile
        
        :param paramFPN str: full path to destination json parameter file
        
        :param str band: the spectral region of the destination data    
        '''
                
        jsonF = open(dataFPN, "w")
      
        json.dump(exportD, jsonF, indent = 2)
      
        jsonF.close()
        
        D = json.loads(json.dumps(self.params, default=lambda o: o.__dict__))
        
        if self.verbose > 1:
            
            pp = pprint.PrettyPrinter(indent=1)

            pp.pprint(D)
            
        jsonF = open(paramFPN, "w")
        
        json.dump(D, jsonF, indent = 2)
        
        jsonF.close()
        
        if self.verbose:
        
            infostr =  '        %s extraction parameters saved as: %s' %(band, paramFPN)
            
            print (infostr)
            
            infostr =  '        %s extracted data saved as: %s' %(band, dataFPN)
            
            print (infostr)
             
    def _ExtractSiteData(self, headers, rowL):
        ''' Exract the site data (ossl file: "soilsite.data.csv")
        
            :paramn headers: list of columns
            :type: list
            
            :param rowL: array of data
            :rtype: list of list
        '''

        metadataItemL = ['id.layer_local_c', 'dataset.code_ascii_txt', 
                         'id.layer_uuid_txt', 'longitude.point_wgs84_dd', 
                         'latitude.point_wgs84_dd', 'layer.sequence_usda_uint16', 
                         'layer.upper.depth_usda_cm', 'layer.lower.depth_usda_cm', 
                         'observation.date.begin_iso.8601_yyyy.mm.dd', 'observation.date.end_iso.8601_yyyy.mm.dd', 
                         'surveyor.title_utf8_txt', 'id.project_ascii_txt', 
                         'id.location_olc_txt', 'layer.texture_usda_txt', 
                         'pedon.taxa_usda_txt', 'horizon.designation_usda_txt', 
                         'longitude.county_wgs84_dd', 'latitude.county_wgs84_dd', 
                         'location.point.error_any_m', 'location.country_iso.3166_txt', 
                         'observation.ogc.schema.title_ogc_txt', 'observation.ogc.schema_idn_url', 
                         'surveyor.contact_ietf_email', 'surveyor.address_utf8_txt', 
                         'dataset.title_utf8_txt', 'dataset.owner_utf8_txt', 
                         'dataset.address_idn_url', 'dataset.doi_idf_url', 
                         'dataset.license.title_ascii_txt', 'dataset.license.address_idn_url', 
                         'dataset.contact.name_utf8_txt', 'dataset.contact_ietf_email', 
                         'id.dataset.site_ascii_txt', 'id_mir', 'id_vis', 'id_neon']

        metadataColumnL = []
        
        for item in metadataItemL:
            
            metadataColumnL.append(metadataItemL.index(item))
            
        self.SitemetatadaItemD = dict(zip(metadataItemL,metadataColumnL))
        
        self.siteD = {}
        
        self.minLat = 90; self.maxLat = -90; self.minLon = 180; self.maxLon = -180
        
        for row in rowL:
            
            #self.siteD[ row[1] ] = {}
            self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ] = {}
        
            for item in self.sitedata:
                
                colNr = headers.index(item)
                            
                self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ][item] = row[colNr]
                
            # Check if site is inside depth limits       
            if float(self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ]["layer.upper.depth_usda_cm"]) < self.soilSample.minDepth  or float(self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ]["layer.lower.depth_usda_cm"]) > self.soilSample.maxDepth:
                
                self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ]["id_vis"] = "FALSE"
                
                self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ]["id_mir"] = "FALSE"
                
                self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ]["id_neon"] = "FALSE"
                
            else:
                
                if float(row[self.SitemetatadaItemD['latitude.point_wgs84_dd']] ) < self.minLat:
                    
                    self.minLat =  float(row[self.SitemetatadaItemD['latitude.point_wgs84_dd']] )
                    
                elif float(row[self.SitemetatadaItemD['latitude.point_wgs84_dd']] ) > self.maxLat:
                    
                    self.maxLat =  float(row[self.SitemetatadaItemD['latitude.point_wgs84_dd']] )

                if float(row[self.SitemetatadaItemD['longitude.point_wgs84_dd']] ) < self.minLon:
                    
                    self.minLon =  float(row[self.SitemetatadaItemD['longitude.point_wgs84_dd']] )
                    
                elif float(row[self.SitemetatadaItemD['longitude.point_wgs84_dd']] ) > self.maxLon:
                    
                    self.maxLon =  float(row[self.SitemetatadaItemD['longitude.point_wgs84_dd']] )
                    
    def _ExtractLabData(self, headers, rowL):
        ''' Extract the key lab data required (ossl file: "soillab.data.csv")
        
            :paramn headers: list of columns
            :type: list
            
            :param rowL: array of data
            :rtype: list of list
        '''
        
        metadataItemL = ['id.layer_local_c','dataset.code_ascii_txt','id.layer_uuid_txt']

        metadataColumnL = []
        
        for item in metadataItemL:
            
            metadataColumnL.append(metadataItemL.index(item))
            
        self.LabmetatadaItemD = dict(zip(metadataItemL,metadataColumnL))
        
        self.labD = {}

        for row in rowL:
            
            self.labD[ row[self.LabmetatadaItemD['id.layer_uuid_txt'] ] ] = [] 
            
            skip = False
        
            for item in self.params.labData:
                
                colNr = headers.index(item)
                
                #if item in self.params.labDataRange:
                if hasattr(self.params, 'labDataRange'):
                    
                    if hasattr(self.params.labDataRange,item):
                        
                        if row[colNr] != 'NA':
                            
                            itemRange = getattr(self.params.labDataRange,item)
                                                    
                            if float(row[colNr]) < itemRange.min or float(row[colNr]) > itemRange.max:
                                
                                skip = True  
    
            # Loop again, only accept items that are not skipped
            for item in self.params.labData:
                
                colNr = headers.index(item)
                
                if not skip:
                    
                    try:
                        
                        # Only if a numerical value is given
                        self.labD[ row[self.LabmetatadaItemD['id.layer_uuid_txt']] ].append( {'substance': item, 'value':float(row[colNr]) } ) 
                           
                    except:
                        
                        # Otherwise skip this lab parameter for this site
                        pass
                    
    def _ExtractNEONLabData(self, headers, rowL):
        ''' Extract the key lab data required (ossl file: "soillab.data.csv") for NEON (lacks uuid)
        
            :paramn headers: list of columns
            :type: list
            
            :param rowL: array of data
            :rtype: list of list
        '''
        
        metadataItemL = ['id.layer_local_c','dataset.code_ascii_txt','id.layer_uuid_txt']

        metadataColumnL = []
        
        for item in metadataItemL:
            
            metadataColumnL.append(metadataItemL.index(item))
            
        self.NeonLabmetatadaItemD = dict(zip(metadataItemL,metadataColumnL))
        
        self.labD = {}

        for row in rowL:
            
            self.labD[ row[self.LabmetatadaItemD['id.layer_local_c'] ] ] = [] 
            
            skip = False
        
            for item in self.params.labData:
                
                colNr = headers.index(item)
                
                #if item in self.params.labDataRange:
                if hasattr(self.params.labDataRange,item):
                    
                    if row[colNr] != 'NA':
                        
                        itemRange = getattr(self.params.labDataRange,item)
                                                
                        if float(row[colNr]) < itemRange.min or float(row[colNr]) > itemRange.max:
                            
                            skip = True  
    
            # Loop again, only accept items that are not skipped
            for item in self.params.labData:
                
                colNr = headers.index(item)
                
                if not skip:
                    
                    try:
                        
                        # Only if a numerical value is given
                        self.labD[ row[self.LabmetatadaItemD['id.layer_local_c']] ].append( {'substance': item, 'value':float(row[colNr]) } ) 
                           
                    except:
                        
                        # Otherwise skip this lab parameter for this site
                        pass
         
    def _AverageSpectra(self, spectrA, inputBeginWaveLength, inputEndWaveLength, inputBandWidth, outputBeginWaveLength, outputEndWaveLength, outputBandWidth):
        ''' Average high resolution spectral signals to broader bands
        
            :paramn spectrA: array of spectral signsals
            :type: np array
            
            :param inputBeginWaveLength: first wavelength (nm) of spectrA
            :rtype: float
            
            :param inputEndWaveLength: last wavelength (nm) of spectrA
            :rtype: float
            
            :param inputBandWidth: the spectral resolution (nm) of spectrA
            :rtype: float
            
            :param outputBeginWaveLength: first wavelength (nm) of the output spectra
            :rtype: float
            
            :param outputEndWaveLength: last wavelength (nm) of the output spectra
            :rtype: float
            
            :param outputBandWidth: the spectral resolution (nm) of the output spectra
            :rtype: float
        '''
        halfwlstep = outputBandWidth/2
                
        inputWls = np.arange(inputBeginWaveLength, inputEndWaveLength+1, inputBandWidth)
        
        outputWls = np.arange(outputBeginWaveLength, outputEndWaveLength+1, outputBandWidth)

        meanSpectra = np.interp(outputWls, inputWls, spectrA, 
                                   left=outputBeginWaveLength-halfwlstep, 
                                   right=outputEndWaveLength+halfwlstep)
                                   
        return outputWls, meanSpectra
          
    def _ExtractVISNIRSpectraData(self, headers, rowL):
        ''' Extract VISNIR spectra from OSSL csv (ossl file: "visnir.data.csv")
        
            :paramn headers: list of columns
            :type: list
            
            :param rowL: array of data
            :rtype: list of list
        '''
        
        # The list of metadata items must be a complete list of the headers initial metadata
        metadataItemL = ['id.layer_local_c','dataset.code_ascii_txt',
                         'id.layer_uuid_txt','id.scan_local_c',
                         'scan.visnir.date.begin_iso.8601_yyyy.mm.dd',
                         'scan.visnir.date.end_iso.8601_yyyy.mm.dd',
                         'scan.visnir.model.name_utf8_txt',
                         'scan.visnir.model.code_any_txt',
                         'scan.visnir.method.optics_any_txt',
                         'scan.visnir.method.preparation_any_txt',
                         'scan.visnir.license.title_ascii_txt',
                         'scan.visnir.license.address_idn_url',
                         'scan.visnir.doi_idf_url','scan.visnir.contact.name_utf8_txt',
                         'scan.visnir.contact.email_ietf_txt']

        
        metadataColumnL = []
        
        for item in metadataItemL:
            
            metadataColumnL.append(metadataItemL.index(item))
            
        self.VISNIRmetatadaItemD = dict(zip(metadataItemL,metadataColumnL))
                
        self.VISNIRspectraD = {};  self.VISNIRmetaD = {}

        mincoltest = int( len(metadataItemL)+(self.params.visnir.beginWaveLength-350)/2 )
        
        mincol = int( len(metadataItemL) )
                    
        maxcoltest = int( len(headers)-(2500-self.params.visnir.endWaveLength)/2 )
        
        maxcol = int( len(headers) )
            
        for row in rowL:
                                     
            if self.siteD[ row[self.VISNIRmetatadaItemD['id.layer_uuid_txt']] ]['id_vis'] == 'TRUE':
                                
                if 'NA' in row[mincoltest:maxcoltest]:
                    
                    self.siteD[ row[self.VISNIRmetatadaItemD['id.layer_uuid_txt']] ]['id_vis'] = 'FALSE'
                    
                    print ('WARING The requested wavelength range contains NoData')
                    
                    continue
                
                row = [0 if i == 'NA' else i for i in row]
                
                visnirSpectrA = np.asarray(row[mincol:maxcol]).astype(float)
                            
                self.outputWls, spectraA = self._AverageSpectra(visnirSpectrA, 350, 2500, 2, self.params.visnir.beginWaveLength, self.params.visnir.endWaveLength, self.params.visnir.outputBandWidth)
                         
                spectraA = np.round(spectraA, 4) 
                                     
                self.VISNIRmetaD[ row[self.VISNIRmetatadaItemD['id.layer_uuid_txt']] ] = {'scandatebegin': row[self.VISNIRmetatadaItemD['scan.visnir.date.begin_iso.8601_yyyy.mm.dd']] ,
                                    'scandateend': row[self.VISNIRmetatadaItemD['scan.visnir.date.end_iso.8601_yyyy.mm.dd']] ,
                                    'sampleprep': row[self.VISNIRmetatadaItemD['scan.visnir.method.preparation_any_txt']],
                                    'instrument': row[self.VISNIRmetatadaItemD['scan.visnir.model.name_utf8_txt']]}
                                
                self.VISNIRspectraD[ row[self.VISNIRmetatadaItemD['id.layer_uuid_txt']] ] = spectraA
                
                self.VISNIRnumberOfwl = spectraA.shape[0]
                
    def _ExtractNEONpectraData(self, headers, rowL):
        ''' Extract NEON (NeoSpectra) NIR spectra from OSSL csv (ossl file: "neon.data.csv")
        
            :paramn headers: list of columns
            :type: list
            
            :param rowL: array of data
            :rtype: list of list
        '''
        
        # The list of metadata items must be a complete list of the headers initial metadata
        metadataItemL = ['id.layer_local_c','id.scan_local_c',
                         'scan.lab_utf8_txt','scan.nir.date.begin_iso.8601_yyyy.mm.dd',
                         'scan.nir.date.end_iso.8601_yyyy.mm.dd','scan.nir.model.name_utf8_txt',
                         'scan.nir.model.serialnumber_utf8_int','scan.nir.accessory.used_utf8_txt'
                         ,'scan.nir.method.preparation_any_txt','scan.nir.license.title_ascii_txt',
                         'scan.nir.license.address_idn_url','scan.nir.doi_idf_url',
                         'scan.nir.contact.name_utf8_txt','scan.nir.contact.email_ietf_txt']
                         
        metadataColumnL = []
        
        for item in metadataItemL:
            
            metadataColumnL.append(metadataItemL.index(item))
            
        self.NEONmetatadaItemD = dict(zip(metadataItemL,metadataColumnL))
                
        self.NEONspectraD = {}

        mincol = int( len(metadataItemL)+(self.params.visnirBegin-1350)/2 )
                    
        maxcol = int( len(headers)-1-(2550-self.params.visnirEnd)/2 )
            
        for row in rowL:
                                     
            if self.siteD[ row[self.NEONmetatadaItemD['id.layer_local_c']] ]['id_neon'] == 'TRUE':
                                
                if 'NA' in row[mincol:maxcol]:
                    
                    self.siteD[ row[self.NEONmetatadaItemD['id.layer_local_c']] ]['id_neon'] = 'FALSE'
                    
                    continue
                
                visnirSpectrA = np.asarray(row[mincol:maxcol]).astype(float)
                            
                spectraA = self._AverageSpectra(visnirSpectrA, self.params.visnirStep)
                                
                self.NEONspectraD[ row[self.NEONmetatadaItemD['id.layer_local_c']] ] = spectraA
                
                self.NEONnumberOfwl = spectraA.shape[0]
      
    def _SetProjectJson(self,modname):
        '''
        '''       
        projectid = '%s_%s_%s' %(self.params.campaign.geoRegion, modname, Today())
        
        projectname = '%s_%s' %(self.params.campaign.geoRegion, modname)
        
        projectD = {'id': projectid, 'name':projectname, 'userId': self.params.userId,
                    'importVersion': self.params.importVersion}
                        
        return projectD
   
    def _SetCampaignD(self, modname):
        ''' Set parameters defining the campaign
        '''
        
        campaignD = {'campaignId': modname, 
                     'campaignShortId': self.params.campaign.campaignShortId,
                     'campaignType':self.params.campaign.campaignType,
                     'theme': self.params.campaign.theme,
                     'product':self.params.campaign.product,
                     'geoRegion':self.params.campaign.geoRegion,
                     'minLat':self.minLat,
                     'maxLat':self.maxLat,
                     'minLon':self.minLon,
                     'maxLon':self.maxLon,
                     }
        
        return campaignD
    
    def _ReportSiteMeta(self,site):
        """
        """
                                                             
        metaD = {'siteLocalId': self.siteD[site]['id.layer_local_c']} 
        
        metaD['dataset'] = self.siteD[site]['dataset.code_ascii_txt'] 
        
        ''' Latitude and Longitude id changed in online OSSL'''
        #jsonD['latitude_dd'] = self.siteD[site]['latitude_wgs84_dd']
        metaD['latitude_dd'] = self.siteD[site]['latitude.point_wgs84_dd']
        
        #jsonD['longitude_dd'] = self.siteD[site]['latitude_wgs84_dd']
        metaD['longitude_dd'] = self.siteD[site]['longitude.point_wgs84_dd']
                        
        metaD['minDepth'] = self.siteD[site]['layer.upper.depth_usda_cm']
        
        metaD['maxDepth'] = self.siteD[site]['layer.lower.depth_usda_cm']
        
        return (metaD)
                 
    def _AssembleVISNIRJsonD(self):
        """ Convert the extracted data to json objects for export
        """
            
        projectD = self._SetProjectJson(self.visnirModelN)
                
        projectD['campaign'] = self._SetCampaignD(self.visnirModelN)
             
                 
        projectD['waveLength'] = self.outputWls.tolist()
      
        varLD = []
        
        for site in self.siteD:
                    
            if self.siteD[site]['id_vis'] == 'TRUE':
                
                metaD = self._ReportSiteMeta(site)
                
                jsonD = {'id':site, 'meta' : metaD}
                
                # Add the VISNIR scan specific metadata for this layer
                for key in self.VISNIRmetaD[ site]:
                                  
                    metaD[key] = self.VISNIRmetaD[site][key] 

                # Add the VISNIR spectral signal                                
                jsonD['signalMean'] = self.VISNIRspectraD[site].tolist()
                
                jsonD['abundances'] = self.labD[site]
                
                varLD.append(jsonD)
                
        projectD['spectra'] = varLD
                      
        # export, or dump, the assembled json objects      
        self._DumpSpectraJson(projectD, self.visnirDataFPN, self.visnirParamFPN , "VISNIR")

        
    def _AssembleNEONJsonD(self, arrangeddatafolder):
        ''' Convert the extracted data to json objects for export
        '''
    
        modname = '%s_%s-%s_%s' %(os.path.split(self.params.rootFP)[1], 
                    self.params.neonBegin,self.params.neonEnd,int(self.params.neonStep*2))

        exportD = self._SetReportJson(modname)
        
        exportD['campaign'] = self._SetCampaignD(modname)
               
        if self.params.neonStep == 1:
            
            wl = [i for i in range(self.params.neonBegin, self.params.neonEnd+1, self.params.neonStep*2)]
            
        else:
            
            wl = [i+self.params.neonStep for i in range(self.params.neonBegin, self.params.neonEnd, self.params.neonStep*2)]
                
        # Reduce wl if bands are cut short while averaging
        wl = wl[0:self.neonnumberOfwl]
                
        exportD['wavelength'] = wl
      
        varLD = []
        
        for site in self.siteD:
                    
            if self.siteD[site]['id_vis'] == 'TRUE':
                                                             
                jsonD = {'uuid':site}
                
                ''' Latitude and Longitude id changed in online OSSL'''
                #jsonD['latitude_dd'] = self.siteD[site]['latitude_wgs84_dd']
                jsonD['latitude_dd'] = self.siteD[site]['latitude.point_wgs84_dd']
                
                #jsonD['longitude_dd'] = self.siteD[site]['latitude_wgs84_dd']
                jsonD['longitude_dd'] = self.siteD[site]['longitude.point_wgs84_dd']
                                
                jsonD['mindepth'] = self.siteD[site]['layer.upper.depth_usda_cm']
                
                jsonD['maxdepth'] = self.siteD[site]['layer.lower.depth_usda_cm']
                
                jsonD['samplemean'] = self.neonspectraD[site].tolist()
                
                jsonD['abundance'] = self.labD[site]
                
                varLD.append(jsonD)
                
        exportD['labspectra'] = varLD
                      
        # export, or dump, the assembled json objects      
        self._DumpNEONJson(exportD)
                                                  
    def _PilotImport(self,sourcedatafolder, dstRootFP):
        ''' Steer the sequence of processes for extracting OSSL csv data to json objects
        ''' 
        
        # Set the source file names
        self._SetSrcFPNs(sourcedatafolder)
        
        self._SetProjectNameId()
        
        # REad the site data
        headers, rowL = ReadCSV(self.srcSoilSiteFPN)
        
        # Extract the site data
        self._ExtractSiteData(headers, rowL)
        
        # Read the laboratory (wet chemistry) data
        headers, rowL = ReadCSV(self.srcSoilLabFPN)
        
        # Extract the laboratory (wet chemistry) data
        self._ExtractLabData(headers, rowL)
        
        if self.params.visnir.apply:
            
            # Set the sdestination file names - must be done after _ExtractVISNIRSpectraData 
            self.visnirModelN, self.visnirParamFPN, self.visnirDataFPN, self.visnirColumnStep, self.visnirWlStep = self._SetDstFPN(dstRootFP,'visnir',self.visnir.subFP) 
      
            headers, rowL = ReadCSV(self.srcVISNIRFPN)

            self._ExtractVISNIRSpectraData(headers, rowL)
                        
            self._AssembleVISNIRJsonD()
            
        if self.params.neon.apply:
      
            headers, rowL = ReadCSV(self.srcNEONFPN)

            self._ExtractNEONSpectraData(headers, rowL)
                        
            self._AssembleNEONJsonD()
            
        if self.params.mir.apply:
      
            headers, rowL = ReadCSV(self.srcMIRFPN)

            self._ExtractMIRSpectraData(headers, rowL)
                        
            self._AssembleNEONJsonD()
                                                      
def SetupProcesses(iniParams):
    '''Setup and loop processes
    
    :param rootpath: path to project root folder 
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
        
        CreateArrangeParamJson(jsonFP,iniParams['projFN'],'import')
        
    jsonProcessObjectL = ReadProjectFile(dstRootFP, iniParams['projFN'], jsonFP)
           
    #Loop over all json files and create Schemas and Tables
    for jsonObj in jsonProcessObjectL:

        print ('    jsonObj:', jsonObj)

        paramD = ReadImportParamsJson(jsonObj)
        

        '''    
        pp = pprint.PrettyPrinter(indent=2)

        pp.pprint(paramD)
        '''
        
        # Invoke the import
        ossl = ImportOSSL(paramD)
        
        ossl._PilotImport(iniParams['sourcedatafolder'], dstRootFP)
                    
if __name__ == "__main__":
    ''' If script is run as stand alone
    '''
        
       
    if len(sys.argv) != 2:
        
        sys.exit('Give the link to the json file to run the process as the only argument')
    
    #Get the json file
    jsonFPN = sys.argv[1]
    
    if not os.path.exists(jsonFPN):
        
        exitstr = 'json file not found: %s' %(jsonFPN)
           
    #jsonFPN = "/Users/thomasgumbricht/docs-local/OSSL/import_ossl.json"
    
    iniParams = ReadAnyJson(jsonFPN)

    SetupProcesses(iniParams)
    