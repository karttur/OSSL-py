'''
Created on 8 Sep 2022

Edited on 27 Sep 2022

Edited 22 Feb 2023

last edited 7 August 2023

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

from copy import deepcopy

import pprint

# Third party imports
import numpy as np

# Package imports

from util.makeObject import Obj

from util.utilities import Today

from util.jsonIO import ReadAnyJson

from util.defaultParams import CheckMakeDocPaths, CreateArrangeParamJson, ReadProjectFile

from util.csvReader import ReadCSV


def ReadImportParamsJson(jsonFPN):
    """ Read the parameters for importing OSSL data
    
    :param jsonFPN: path to json file
    :type jsonFPN: str
    
    :return paramD: parameters
    :rtype: dict
   """
            
    return ReadAnyJson(jsonFPN)
    
 
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
                
    def _SetDstFPN(self, dstRootFP, band, subFP):
        ''' Set destination file paths and names
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
        
        FP = os.path.join(dstRootFP, subFP)
            
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
        
        :param exportD: formatted dictionary
        :type exportD: dict
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

        mincol = int( len(metadataItemL)+(self.params.visnir.beginWaveLength-350)/2 )
                    
        maxcol = int( len(headers)-(2500-self.params.visnir.endWaveLength)/2 )
            
        for row in rowL:
                                     
            if self.siteD[ row[self.VISNIRmetatadaItemD['id.layer_uuid_txt']] ]['id_vis'] == 'TRUE':
                                
                if 'NA' in row[mincol:maxcol]:
                    
                    self.siteD[ row[self.VISNIRmetatadaItemD['id.layer_uuid_txt']] ]['id_vis'] = 'FALSE'
                    
                    continue
                
                visnirSpectrA = np.asarray(row[mincol:maxcol]).astype(float)
                            
                spectraA = self._AverageSpectra(visnirSpectrA, self.visnirColumnStep)
                          
                spectraA = np.round(spectraA, 3) 
                                     
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
        '''
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
        ''' Convert the extracted data to json objects for export
        '''
            
        projectD = self._SetProjectJson(self.visnirModelN)
                
        projectD['campaign'] = self._SetCampaignD(self.visnirModelN)
             
        if self.visnirColumnStep == 1:
            
            wl = [i for i in range(self.params.visnir.beginWaveLength, self.params.visnir.endWaveLength+1, 2)]
            
        else:
            
            wl = [i+self.visnirColumnStep for i in range(self.params.visnir.beginWaveLength, self.params.visnir.endWaveLength, self.visnirWlStep)]
                     
        # Reduce wl if bands are cut short while averaging
        #wl = wl[0:self.VISNIRnumberOfwl]
         
        projectD['waveLength'] = wl
      
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
                                                  
    def PilotImport(self,sourcedatafolder, dstRootFP):
        ''' Steer the sequence of processes for extracting OSSL csv data to json objects
        ''' 
        
        # Set the source file names
        self._SetSrcFPNs(sourcedatafolder)
        
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
            
            # Set the sdestination file names - must be done after _ExtractVISNIRSpectraData 
            #self._SetDstFPNs(dstRootFP)
            
            self._AssembleNEONJsonD()
            
        if self.params.mir.apply:
      
            headers, rowL = ReadCSV(self.srcMIRFPN)

            self._ExtractMIRSpectraData(headers, rowL)
            
            # Set the sdestination file names - must be done after _ExtractVISNIRSpectraData 
            #self._SetDstFPNs(dstRootFP)
            
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
        
    dstRootFP, jsonFP = CheckMakeDocPaths(docpath,arrangeddatafolder, jsonpath, sourcedatafolder)
    
    if createjsonparams:
        
        CreateArrangeParamJson(jsonFP,projFN,'import')
        
    jsonProcessObjectL = ReadProjectFile(dstRootFP, projFN, jsonFP)
           
    #Loop over all json files and create Schemas and Tables
    for jsonObj in jsonProcessObjectL:
        
        print ('    jsonObj:', jsonObj)

        paramD = ReadImportParamsJson(jsonObj)
        
        # Invoke the import
        ossl = ImportOSSL(paramD)
        
        ossl.PilotImport(sourcedatafolder, dstRootFP)
                    
if __name__ == "__main__":
    ''' If script is run as stand alone
    '''
            
    docpath = '/Users/thomasgumbricht/docs-local/OSSL/Sweden/LUCAS'
    #docpath = '/Users/thomasgumbricht/docs-local/OSSL/Europe/LUCAS'
    
    createjsonparams=False
    
    sourcedatafolder = 'data'
    
    arrangeddatafolder = 'arranged-data'
    
    projFN = 'extract_rawdata.txt'
    
    jsonpath = 'json-import'
    
    SetupProcesses(docpath, createjsonparams, sourcedatafolder, arrangeddatafolder, projFN, jsonpath)
    