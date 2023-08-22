'''
Created on 8 Sep 2022

Edited on 27 Sep 2022

Edited 22 Feb 2023

last edited 2 August 2023

@author: thomasgumbricht

Notes
-----
The module OSSL.py:

    requires that you have downloaded and exploded a standard zip-archive from OSSL 
    (see https://karttur.github.io/soil-spectro/libspectrodata/spectrodata-OSSL-api-explorer/). 
    The exploded folder should be renamed to reflect its geographic and/or thematic content. 
    This script then expects the exploded folder to contain the following 5 csv 
    files (in alphabetic order):
        
        - mir.data.csv 
        - neon.data.csv
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

import datetime

# Third party imports
import numpy as np

def Today():
    
    return datetime.datetime.now().date().strftime("%Y%m%d")

def ImportParams():
    """ Default template parameters for importing OSSL csv data
    
        :returns: parameter dictionary
        
        :rtype: dict
    """
    
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
    
    paramD['soilsample'] = {'mindepth':0,'maxdepth':20}
    
    paramD['input'] = {}
    
    paramD['input']['rootFP'] = '/path/to/folder/with/ossl/download'
    
    paramD['input']['visnirSubFP'] = 'visnir'
    
    paramD['input']['mirSubFP'] = 'mir'
    
    paramD['input']['neonSubFP'] = 'neon'
        

    
    paramD['input']['visnir'] = True
    
    paramD['input']['mir'] = False
    
    paramD['input']['neon'] = False
    
    
    paramD['input']['visnirBegin'] = 460
    
    paramD['input']['visnirEnd'] = 2500
    
    paramD['input']['visnirInputBandWidth'] = 2
    
    paramD['input']['visnirOutputBandWidth'] = 10
    
    #paramD['input']['visnirStep'] = 2
    
    
    
    paramD['input']['mirBegin'] = 2500
    
    paramD['input']['mirEnd'] = 8000
    
    paramD['input']['mirInputBandWidth'] = 2
    
    paramD['input']['mirOutputBandWidth'] = 10
    
    #paramD['input']['mirStep'] = 2
    
    
    paramD['input']['neonBegin'] = 1350
    
    paramD['input']['neonEnd'] = 2550
    
    paramD['input']['neonInputBandWidth'] = 2
    
    paramD['input']['neonOutputBandWidth'] = 10
    
    #paramD['input']['neonStep'] = 2
    
    
    paramD['input']['percentOK'] = 70
    
    
    ''' USDA oriented input data
    paramD['input']['labData'] = ['c.tot_usda.a622_w.pct','ca.ext_usda.a1059_mg.kg',
                'caco3_usda.a54_w.pct','oc_usda.c729_w.pct', 'cec_usda.a723_cmolc.kg',
                'clay.tot_usda.a334_w.pct','fe.ext_usda.a1064_mg.kg']
    
    paramD['input']['labDataRange'] = {'oc_usda.calc_wpct':{'min':0,'max':10}}
    
    '''
    ''' LUCAS oriented input data'''
    paramD['input']['labData'] = ['caco3_usda.a54_w.pct',
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
    
    paramD['input']['labDataRange'] = {}
    
    paramD['input']['labDataRange']['caco3_usda.a54_w.pct'] = {
        "min": 0,
        "max": 10}

  
    return (paramD)

def CreateParamJson(jsonFP):
    """ Create the default json parameters file structure, only to create template if lacking
    
        :param str dstrootFP: directory path 
        
        :param str jsonpath: subfolder under directory path 
    """
    
    # Get the default params
    paramD = ImportParams()
    
    # Set the json FPN
    jsonFPN = os.path.join(jsonFP, 'template_import_ossl-spectra.json')
    
    if os.path.exists(jsonFPN):
        
        return (True, jsonFPN)
    
    # Dump the paramD as a json object   
    jsonF = open(jsonFPN, "w")
  
    json.dump(paramD, jsonF, indent = 2)
  
    jsonF.close()
    
    return (False, jsonFPN)

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
                              
    def _SetSrcFPNs(self, sourcedatafolder):
        ''' Set source file paths and names
        '''
        # All OSSL data are download as a zipped subfolder with data given standard names as of below
                    
        self.srcVISNIRFPN = os.path.join(self.params.input.rootFP,sourcedatafolder,'visnir.data.csv')
        
        self.srcMIRFPN = os.path.join(self.params.input.rootFP,sourcedatafolder,'mir.data.csv')
        
        self.srcSoilLabFPN = os.path.join(self.params.input.rootFP,sourcedatafolder,'soillab.data.csv')
        
        self.srcSoilSiteFPN = os.path.join(self.params.input.rootFP,sourcedatafolder,'soilsite.data.csv')
                
    def _SetDstFPNs(self, dstRootFP):
        ''' Set destination file paths and names
        '''
        
        # VISNIR files
        if self.params.input.visnir:
        
            visnirFP = os.path.join(dstRootFP, self.params.input.visnirSubFP)
            
            if not os.path.exists(visnirFP):
                
                os.makedirs(visnirFP)
             
            modelN = '%s_%s-%s_%s' %(os.path.split(self.params.input.rootFP)[1], 
                        self.params.input.visnirBegin,self.params.input.visnirEnd,int(self.params.input.visnirStep*2))
    
            visnirParamFN = 'params-visnir_%s.json' %(modelN)
            
            self.visnirParamFPN = os.path.join(visnirFP, visnirParamFN)
            
            visnirDataFN = 'data-visnir_%s.json' %(modelN)
                
            self.visnirDataFPN = os.path.join(visnirFP, visnirDataFN)
        
        # MIR files
        if self.params.input.mir:
            mirFP = os.path.join(dstRootFP, self.params.input.mirSubFP)
            
            if not os.path.exists(mirFP):
                
                os.makedirs(mirFP)
             
            modelN = '%s_%s-%s_%s' %(os.path.split(self.params.input.rootFP)[1], 
                        self.params.input.mirBegin,self.params.input.mirEnd,int(self.params.input.mirStep*2))
    
            mirParamFN = 'params-mir_%s.json' %(modelN)
            
            self.mirParamFPN = os.path.join(mirFP, mirParamFN)
            
            mirDataFN = 'data-mir_%s.json' %(modelN)
                
            self.mirDataFPN = os.path.join(mirFP,  mirDataFN)
        
        # NEON files
        if self.params.input.mir:
            neonFP = os.path.join(dstRootFP, self.params.input.neonSubFP)
            
            if not os.path.exists(neonFP):
                
                os.makedirs(neonFP)
             
            modelN = '%s_%s-%s_%s' %(os.path.split(self.params.input.rootFP)[1], 
                        self.params.input.neonBegin,self.params.input.neonEnd,int(self.params.input.neonStep*2))
    
            neonParamFN = 'params-neon_%s.json' %(modelN)
            
            self.neonParamFPN = os.path.join(neonFP,  neonParamFN)
            
            neonDataFN = 'data-neon_%s.json' %(modelN)
                
            self.neonDataFPN = os.path.join(neonFP,  neonDataFN)
                  
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
        
        infostr =  '        VISNIR extraction parameters saved as: %s' %(self.visnirParamFPN)
        
        print (infostr)
        
        infostr =  '        VISNIR extracted data saved as: %s' %(self.visnirDataFPN)
        
        print (infostr)
        
    def _DumpNEONJsonOld(self, exportD):
        ''' Export, or dump, the imported NEON OSSL data as json objects
        
        :param exportD: formatted dictionary
        
        :type exportD: dict
        '''
                
        jsonF = open(self.neonDataFPN, "w")
      
        json.dump(exportD, jsonF, indent = 2)
      
        jsonF.close()
        
        D = json.loads(json.dumps(self.params, default=lambda o: o.__dict__))
        
        jsonF = open(self.neonParamFPN, "w")
        
        json.dump(D, jsonF, indent = 2)
        
        jsonF.close()
        
        infostr =  '        NEON extraction parameters saved as: %s' %(self.neonParamFPN)
        
        print (infostr)
        
        infostr =  '        NEON extracted data saved as: %s' %(self.neonDataFPN)
        
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
        
        self.minlat = 90; self.maxlat = -90; self.minlon = 180; self.maxlon = -180
        
        for row in rowL:
            
            #self.siteD[ row[1] ] = {}
            self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ] = {}
        
            for item in self.sitedata:
                
                colNr = headers.index(item)
                            
                self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ][item] = row[colNr]
                
            # Check if site is inside depth limits       
            if float(self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ]["layer.upper.depth_usda_cm"]) < self.soilsample.mindepth  or float(self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ]["layer.lower.depth_usda_cm"]) > self.soilsample.maxdepth:
                
                self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ]["id_vis"] = "FALSE"
                
                self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ]["id_mir"] = "FALSE"
                
                self.siteD[ row[self.SitemetatadaItemD['id.layer_uuid_txt']] ]["id_neon"] = "FALSE"
                
            else:
                
                if float(row[self.SitemetatadaItemD['latitude.point_wgs84_dd']] ) < self.minlat:
                    
                    self.minlat =  float(row[self.SitemetatadaItemD['latitude.point_wgs84_dd']] )
                    
                elif float(row[self.SitemetatadaItemD['latitude.point_wgs84_dd']] ) > self.maxlat:
                    
                    self.maxlat =  float(row[self.SitemetatadaItemD['latitude.point_wgs84_dd']] )

                if float(row[self.SitemetatadaItemD['longitude.point_wgs84_dd']] ) < self.minlon:
                    
                    self.minlon =  float(row[self.SitemetatadaItemD['longitude.point_wgs84_dd']] )
                    
                elif float(row[self.SitemetatadaItemD['longitude.point_wgs84_dd']] ) > self.maxlon:
                    
                    self.maxlon =  float(row[self.SitemetatadaItemD['longitude.point_wgs84_dd']] )
                    
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
        
            for item in self.params.input.labData:
                
                colNr = headers.index(item)
                
                #if item in self.params.input.labDataRange:
                if hasattr(self.params.input.labDataRange,item):
                    
                    if row[colNr] != 'NA':
                        
                        itemRange = getattr(self.params.input.labDataRange,item)
                                                
                        if float(row[colNr]) < itemRange.min or float(row[colNr]) > itemRange.max:
                            
                            skip = True  
    
            # Loop again, only accept items that are not skipped
            for item in self.params.input.labData:
                
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
        
            for item in self.params.input.labData:
                
                colNr = headers.index(item)
                
                #if item in self.params.input.labDataRange:
                if hasattr(self.params.input.labDataRange,item):
                    
                    if row[colNr] != 'NA':
                        
                        itemRange = getattr(self.params.input.labDataRange,item)
                                                
                        if float(row[colNr]) < itemRange.min or float(row[colNr]) > itemRange.max:
                            
                            skip = True  
    
            # Loop again, only accept items that are not skipped
            for item in self.params.input.labData:
                
                colNr = headers.index(item)
                
                if not skip:
                    
                    try:
                        
                        # Only if a numerical value is given
                        self.labD[ row[self.LabmetatadaItemD['id.layer_local_c']] ].append( {'substance': item, 'value':float(row[colNr]) } ) 
                           
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

        mincol = int( len(metadataItemL)+(self.params.input.visnirBegin-350)/2 )
                    
        maxcol = int( len(headers)-1-(2500-self.params.input.visnirEnd)/2 )
            
        for row in rowL:
                                     
            if self.siteD[ row[self.VISNIRmetatadaItemD['id.layer_uuid_txt']] ]['id_vis'] == 'TRUE':
                                
                if 'NA' in row[mincol:maxcol]:
                    
                    self.siteD[ row[self.VISNIRmetatadaItemD['id.layer_uuid_txt']] ]['id_vis'] = 'FALSE'
                    
                    continue
                
                visnirSpectrA = np.asarray(row[mincol:maxcol]).astype(float)
                            
                spectraA = self._AverageSpectra(visnirSpectrA, self.params.input.visnirStep)
                
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

        mincol = int( len(metadataItemL)+(self.params.input.visnirBegin-1350)/2 )
                    
        maxcol = int( len(headers)-1-(2550-self.params.input.visnirEnd)/2 )
            
        for row in rowL:
                                     
            if self.siteD[ row[self.NEONmetatadaItemD['id.layer_local_c']] ]['id_neon'] == 'TRUE':
                                
                if 'NA' in row[mincol:maxcol]:
                    
                    self.siteD[ row[self.NEONmetatadaItemD['id.layer_local_c']] ]['id_neon'] = 'FALSE'
                    
                    continue
                
                visnirSpectrA = np.asarray(row[mincol:maxcol]).astype(float)
                            
                spectraA = self._AverageSpectra(visnirSpectrA, self.params.input.visnirStep)
                                
                self.NEONspectraD[ row[self.NEONmetatadaItemD['id.layer_local_c']] ] = spectraA
                
                self.NEONnumberOfwl = spectraA.shape[0]
      
    def _AssembleVISNIRJsonD(self):
        ''' Convert the extracted data to json objects for export
        '''
    
        modname = '%s_%s-%s_%s' %(os.path.split(self.params.input.rootFP)[1], 
                    self.params.input.visnirBegin,self.params.input.visnirEnd,int(self.params.input.visnirStep*2))
        
            
        exportid = '%s_%s_%s' %(self.params.campaign.georegion, modname, Today())
        
        exportD = {'id': exportid}
        
        exportD['userid'] = self.params.userid
        
        exportD['importVersion'] = self.params.importversion
        
        campaignD = {'campaingId': modname, 
                     'campaignShortId': self.params.campaign.campaignshortid,
                     'campaignType':self.params.campaign.campaigntype,
                     'theme': self.params.campaign.theme,
                     'product':self.params.campaign.product,
                     'geoRegion':self.params.campaign.georegion,
                     'minLat':self.minlat,
                     'maxLat':self.maxlat,
                     'minLon':self.minlon,
                     'maxLon':self.maxlon,
                     }
        
        exportD['campaign'] = campaignD
               
        if self.params.input.visnirStep == 1:
            
            wl = [i for i in range(self.params.input.visnirBegin, self.params.input.visnirEnd+1, self.params.input.visnirStep*2)]
            
        else:
            
            wl = [i+self.params.input.visnirStep for i in range(self.params.input.visnirBegin, self.params.input.visnirEnd, self.params.input.visnirStep*2)]
                
        # Reduce wl if bands are cut short while averaging
        wl = wl[0:self.VISNIRnumberOfwl]
                
        exportD['waveLength'] = wl
      
        varLD = []
        
        for site in self.siteD:
                    
            if self.siteD[site]['id_vis'] == 'TRUE':
                                                             
                
                
                metaD = {'siteLocalId': self.siteD[site]['id.layer_local_c']} 
                
                metaD['dataset'] = self.siteD[site]['dataset.code_ascii_txt'] 
                
                ''' Latitude and Longitude id changed in online OSSL'''
                #jsonD['latitude_dd'] = self.siteD[site]['latitude_wgs84_dd']
                metaD['latitude_dd'] = self.siteD[site]['latitude.point_wgs84_dd']
                
                #jsonD['longitude_dd'] = self.siteD[site]['latitude_wgs84_dd']
                metaD['longitude_dd'] = self.siteD[site]['longitude.point_wgs84_dd']
                                
                metaD['minDepth'] = self.siteD[site]['layer.upper.depth_usda_cm']
                
                metaD['maxDepth'] = self.siteD[site]['layer.lower.depth_usda_cm']
                 
                # Add the scan specific metadata for this layer
  
                for key in self.VISNIRmetaD[ site]:
                                  
                    metaD[key] = self.VISNIRmetaD[site][key]         
                
                jsonD = {'id':site, 'meta' : metaD}
                                                
                jsonD['sampleMean'] = self.VISNIRspectraD[site].tolist()
                
                jsonD['abundance'] = self.labD[site]
                
                varLD.append(jsonD)
                
        exportD['spectra'] = varLD
                      
        # export, or dump, the assembled json objects      
        self._DumpVISNIRJson(exportD)
        
    def _AssembleNEONJsonD(self, arrangeddatafolder):
        ''' Convert the extracted data to json objects for export
        '''
    
        modname = '%s_%s-%s_%s' %(os.path.split(self.params.input.rootFP)[1], 
                    self.params.input.neonBegin,self.params.input.neonEnd,int(self.params.input.neonStep*2))
        
        #exportD = {'system': modname}
        
        exportid = '%s_%s_%s' %(self.params.campaign.georegion, modname, Today())
        
        exportD = {'id': exportid}
        
        exportD['userid'] = self.params.userid
        
        exportD['importversion'] = self.params.importversion
        
        exportD['importversion'] = self.params.importversion
        
        campaignD = {'campaingid': modname, 
                     'campaignshortid': self.params.campaign.campaignshortid,
                     'campaigntype':self.params.campaign.campaigntype,
                     'theme': self.params.campaign.theme,
                     'product':self.params.campaign.product,
                     'georegion':self.params.campaign.georegion,
                     'minlat':self.minlat,
                     'maxlat':self.maxlat,
                     'minlon':self.minlon,
                     'maxlon':self.maxlon,
                     }
        
        exportD['campaign'] = campaignD
               
        if self.params.input.neonStep == 1:
            
            wl = [i for i in range(self.params.input.neonBegin, self.params.input.neonEnd+1, self.params.input.neonStep*2)]
            
        else:
            
            wl = [i+self.params.input.neonStep for i in range(self.params.input.neonBegin, self.params.input.neonEnd, self.params.input.neonStep*2)]
                
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
                                                  
    def PilotImport(self,sourcedatafolder, dstRootFP):
        ''' Steer the sequence of processes for extracting OSSL csv data to json objects
        ''' 
        
        # Set the source file names
        self._SetSrcFPNs(sourcedatafolder)
        
        headers, rowL = self._ReadCSV(self.srcSoilSiteFPN)
        
        self._ExtractSiteData(headers, rowL)
        
        headers, rowL = self._ReadCSV(self.srcSoilLabFPN)
        
        self._ExtractLabData(headers, rowL)     
                
        if self.params.input.visnir:
      
            headers, rowL = self._ReadCSV(self.srcVISNIRFPN)

            self._ExtractVISNIRSpectraData(headers, rowL)
            
            # Set the sdestination file names - must be done after _ExtractVISNIRSpectraData 
            self._SetDstFPNs(dstRootFP)
            
            self._AssembleVISNIRJsonD()
            
        if self.params.input.neon:
      
            headers, rowL = self._ReadCSV(self.srcNEONFPN)

            self._ExtractNEONSpectraData(headers, rowL)
            
            # Set the sdestination file names - must be done after _ExtractVISNIRSpectraData 
            self._SetDstFPNs(dstRootFP)
            
            self._AssembleNEONJsonD()
                                                      
def SetupProcesses(docpath, createjsonparams, sourcedatafolder, arrangeddatafolder, projFN, jsonpath):
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
        
    if not os.path.exists(docpath):
        
        exitstr = "The docpath does not exists: %s" %(docpath)
        
        exit(exitstr)
        
    srcFP = os.path.join(os.path.dirname(__file__),docpath,sourcedatafolder)
        
    if not os.path.exists(srcFP):
        
        exitstr = "The source data path to the original OSSL data does not exists:\n %s" %(srcFP)
        
        exit(exitstr)
        
    dstRootFP = os.path.join(os.path.dirname(__file__),docpath,arrangeddatafolder)
        
    if not os.path.exists(dstRootFP):
        
        os.makedirs(dstRootFP)
        
    jsonFP = os.path.join(dstRootFP,jsonpath)
    
    if not os.path.exists(jsonFP):
        
        os.makedirs(jsonFP)
    
    if createjsonparams:
        
        flag, jsonFPN = CreateParamJson(jsonFP)
        
        if flag:
            
            exitstr = 'json parameter file already exists: %s\n' %(jsonFPN)
        
        else:
        
            exitstr = 'json parameter file created: %s\n' %(jsonFPN)
        
        exitstr += ' Edit the json file for your project and rename it to reflect the commands.\n' 
        
        exitstr += ' Add the path of the edited file to your project file (%s).\n' %(projFN)
        
        exitstr += ' Then set createjsonparams to False in the main section and rerun script.'
        
        exit(exitstr)
       
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
    jsonL = [os.path.join(jsonFP,x.strip())  for x in jsonL if len(x) > 10 and x[0] != '#']

    #Loop over all json files and create Schemas and Tables
    for jsonObj in jsonL:
        
        print ('    jsonObj:', jsonObj)

        paramD = ReadImportParamsJson(jsonObj)
        
        # Invoke the import
        ossl = ImportOSSL(paramD)
        
        ossl.PilotImport(sourcedatafolder, dstRootFP)
                    
if __name__ == "__main__":
    ''' If script is run as stand alone
    '''
            
    docpath = '/Users/thomasgumbricht/docs-local/OSSL/Sweden/LUCAS'
    
    createjsonparams=False
    
    sourcedatafolder = 'data'
    
    arrangeddatafolder = 'arranged-data'
    
    projFN = 'extract_rawdata.txt'
    
    jsonpath = 'json-import'
    
    SetupProcesses(docpath, createjsonparams, sourcedatafolder, arrangeddatafolder, projFN, jsonpath)
    