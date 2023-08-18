'''
Created on 13 Aug 2023

@author: thomasgumbricht

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

#import pprint

# Third party imports
import numpy as np

import pandas as pd

from pprint import pprint

# Package imports

from util.makeObject import Obj

from util.utilities import Today

from util.jsonIO import ReadAnyJson

from util.defaultParams import CheckMakeDocPaths, CreateArrangeParamJson, ReadProjectFile

from util.csvReader import ReadCSV

from util.list_files import GlobGetFileList, CsvFileList

def ReadXspectreImportParamsJson(jsonFPN):
    """ Read the parameters for importing OSSL data
    
    :param jsonFPN: path to json file
    :type jsonFPN: str
    
    :return paramD: parameters
    :rtype: dict
   """
            
    return ReadAnyJson(jsonFPN)
    
 
class ImportXspectre(Obj):
    ''' import soil spectra from OSSL to xspectre json format
    '''
    
    def __init__(self,paramD): 
        ''' Initiate import OSSl class
        
        :param dict param: parameters
        '''
        
        # convert the input parameter dict to class objects
        Obj.__init__(self,paramD)
        
        self.paramD = paramD
            
        # Set class object default data if missing 
        self._SetArrangeDefautls()
        
        # Deep copy parameters to a new obejct class called params
        self.params = deepcopy(self)
                              
    def _SetSrcFPNsOLD(self, sourcedatafolder):
        ''' Set source file paths and names
        '''
        # All xSpectre data are available as json files
        
           
        self.srcVISNIRFPN = os.path.join(self.params.rootFP,sourcedatafolder,'visnir.data.csv')
        
        self.srcMIRFPN = os.path.join(self.params.rootFP,sourcedatafolder,'mir.data.csv')
        
        self.srcNEONFPN = os.path.join(self.params.rootFP,sourcedatafolder,'neon.data.csv')
        
        self.srcSoilLabFPN = os.path.join(self.params.rootFP,sourcedatafolder,'soillab.data.csv')
        
        self.srcSoilSiteFPN = os.path.join(self.params.rootFP,sourcedatafolder,'soilsite.data.csv')
        
    def _ListXspectreJsonFiles(self,sourcedatafolder):
        ''' Get the white reference and a list of the rawspectra to import
        '''

        whiteRefFPN = os.path.join(self.params.rootFP, sourcedatafolder,self.params.whiteReference)
        
        if not os.path.exists(whiteRefFPN):

            exitstr = 'EXITING, the whiteref does not exist: %s' %(self.params.whiteReference)

            exit(exitstr)

        #self.whiteRef = json.load(whiteRefFPN)
        
        self.whiteRefSpectra = ReadXspectreImportParamsJson(whiteRefFPN)



        if not hasattr(self.params, "getList"):

            self.jsonSpectraL = [self.params.listPath]

        elif self.params.getList[0:4].lower() == 'sing':

            self.jsonSpectraL = [self.params.listPath]

        elif self.params.getList[0:3].lower() == 'csv':

            self.jsonSpectraL = CsvFileList(self.params.listPath)


        elif self.params.getList[0:4].lower() == 'glob':

            self.jsonSpectraL = GlobGetFileList(self.params.listPath,
                                                    self.params.pattern)
        else:

            exit ('Parameter getList not recognised')
                
    def _SetDstFPN(self, dstRootFP, band, subFP):
        ''' Set destination file paths and names
        '''

        # Get the band [visnir, mir , neon] object
        bandObject = getattr(self, band)
        
        beginWaveLength = getattr(bandObject, 'beginWaveLength')
        
        endWaveLength = getattr(bandObject, 'endWaveLength')
                
        outputBandWidth = getattr(bandObject, 'outputBandWidth')
                
        FP = os.path.join(dstRootFP, subFP)
            
        if not os.path.exists(FP):
                
            os.makedirs(FP)
 
        modelN = '%s_%s-%s_%s' %(os.path.split(self.params.rootFP)[1], 
                        beginWaveLength, endWaveLength, outputBandWidth)
            
        paramFN = 'params-%s_%s.json' %(band, modelN)
        
        paramFPN = os.path.join(FP, paramFN)
            
        dataFN = 'data-%s_%s.json' %(band, modelN)
                
        dataFPN = os.path.join(FP, dataFN)
        
        return (modelN, paramFPN, dataFPN)
                    
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
                    
    def _ExtractXspectreLabData(self):
        ''' Extract the key lab data required (ossl file: "soillab.data.csv")
        
            :paramn headers: list of columns
            :type: list
            
            :param rowL: array of data
            :rtype: list of list
        '''
        
        self.labD = {}
        
        for key in self.paramD['labData']:
            
            labCsv = getattr(self.params.labData, key)
            
            print (labCsv.filepath)
            
            header, data = ReadCSV(labCsv.filepath)
            
            colL = list(range(0, len(header)))
            
            colL = ['col%s' %s for s in colL]
            
            n = 0
                        
            for row in data:
                
                rowD = dict(zip(colL, row))
                
                siteLocalId = self.paramD['labData'][key]['siteLocalId'] %rowD
                
                siteLocalId = siteLocalId.replace('_','-')
                
                if not siteLocalId in self.xSpectraD:
                    
                    idparts = siteLocalId.split('-')
                    
                    ssid = '%s%s' %(idparts[0],idparts[1])
                    
                    if ssid in self.spectraSiteD:
                        
                        siteLocalId = self.spectraSiteD[ssid]
                
                if not siteLocalId in self.xSpectraD:
                    

                    if 'siteLocalIdAlt' in self.paramD['labData'][key]:
                        
                        siteLocalId = self.paramD['labData'][key]['siteLocalIdAlt'] %rowD
                        
                        siteLocalId = siteLocalId.replace('_','-')
  
                        
                if not siteLocalId in self.xSpectraD:
                    
                    idparts = siteLocalId.split('-')
                    
                    ssid = '%s%s' %(idparts[0],idparts[1])
                    
                    if ssid in self.spectraSiteD:
                        
                        siteLocalId = self.spectraSiteD[ssid]
                    

                        
                if not siteLocalId in self.xSpectraD:
                    
                    
                    
                    n+=1
                    
                    print (n, siteLocalId,"mismatch")
                    
 
                    continue
                
                if not siteLocalId in self.labD:
                    
                    self.labD[siteLocalId] = []

                for analysis in self.paramD['labData'][key]['analysis']:
                   
                    try:
                        
                        float(rowD[analysis])
                        
                    except:
                        
                        break
 
                    
                    substance = self.paramD['labData'][key]['analysis'][analysis]
                    
                    #queryStr = '(%s)' %(analysis) 
                    
                    #queryStr = '%'+queryStr+'s'
                    
                    #value = rowD[analysis]
                    
                    self.labD[siteLocalId].append( {'substance': substance, 'value':float(rowD[analysis])} )
                    

         
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
          
    def _ExtractXspectreSpectraData(self):
        ''' Extract VISNIR spectra from OSSL csv (ossl file: "visnor.data.csv")
        
            :paramn headers: list of columns
            :type: list
            
            :param rowL: array of data
            :rtype: list of list
        '''
        
        # List of metadata to retrieve
        metadataItemL = ['muzzleid',
                         'location',
                         'diagnos',
                         'theme',
                         'family',
                         'species',
                         'brand',
                         'version',
                         'subsample',
                         'depth',
                         'prepcode',
                         'scan',
                         'mode',
                         'sampledate',
                         'scandate',
                         'source',
                         'sensorid',
                         'product',
                         'sampleid']
        
        metadataItemL = ['muzzleid',
                         'location',
                         'theme',
                         'depth',
                         'prepcode',
                         'sampledate',
                         'scandate',
                         'source',  
                         'sampleid']
  
        self.xSpectraD = {};  self.xSpectraMetaD = {}
        
        self.spectraSiteD = {}
        
        self.siteD = {}                
        
        halfwlstep = self.params.xspectrolum.outputBandWidth/2
        
        self.outputWls = np.arange(self.params.xspectrolum.beginWaveLength, self.params.xspectrolum.endWaveLength+1, self.params.xspectrolum.outputBandWidth)
         
        # Get the white reference data
        
        whiteRefSamplemean = np.interp(self.outputWls, np.asarray(self.whiteRefSpectra['wavelength']), np.asarray(self.whiteRefSpectra['samplemean']), 
                                   left=self.params.xspectrolum.beginWaveLength-halfwlstep, 
                                   right=self.params.xspectrolum.endWaveLength+halfwlstep)
            
        #whiteRefSamplestd = np.interp(self.outputWls, np.asarray(self.whiteRefSpectra['wavelength']), np.asarray(self.whiteRefSpectra['samplestd']), 
        #                           left=self.params.xspectrolum.beginWaveLength-halfwlstep, 
        #                           right=self.params.xspectrolum.endWaveLength+halfwlstep)
            
        whiteRefDarkmean = np.interp(self.outputWls, np.asarray(self.whiteRefSpectra['wavelength']), np.asarray(self.whiteRefSpectra['darkmean']), 
                                   left=self.params.xspectrolum.beginWaveLength-halfwlstep, 
                                   right=self.params.xspectrolum.endWaveLength+halfwlstep)
            
        #whiteRefDarkstd = np.interp(self.outputWls, np.asarray(self.whiteRefSpectra['wavelength']), np.asarray(self.whiteRefSpectra['darkStd']), 
        #                           left=self.params.xspectrolum.beginWaveLength-halfwlstep, 
        #                           right=self.params.xspectrolum.endWaveLength+halfwlstep)
        
        #whiteRefStd = np.sqrt((5*whiteRefSamplestd*whiteRefSamplestd)+(5*whiteRefDarkstd*whiteRefDarkstd)/10)
               
        for jitem in self.jsonSpectraL:
            
            #Open the json file
            
            if os.path.split(jitem)[1] == self.whiteReference or os.path.split(jitem)[1].lower().startswith('whiteref'):
                
                continue
            
            spectraD = ReadXspectreImportParamsJson(jitem )
            
            if spectraD['depth']  < self.soilSample.minDepth  or spectraD['depth'] > self.soilSample.maxDepth:
            
                continue
            
            localid = '%s_%s' %(spectraD['location'],spectraD['sampleid'])
            
            dataset = 'xspectre_%s' %(spectraD['sensorid'])
            
            try:
            
                site, minDepth, maxDepth = spectraD['sampleid'].split('-')
                
            except:
                
                print (jitem)
                
                SNULLE
                
            shortsampleid = '%s%s' %(site, minDepth)
                
            self.spectraSiteD[shortsampleid] = spectraD['sampleid']
            
            self.siteD[ spectraD['sampleid'] ] = {}
            
            self.siteD[ spectraD['sampleid'] ]['siteLocalId'] = localid
            
            self.siteD[ spectraD['sampleid'] ]['dataset'] = dataset
            
            self.siteD[ spectraD['sampleid'] ]['minDepth'] = minDepth
            
            self.siteD[ spectraD['sampleid'] ]['maxDepth'] = maxDepth
            
            for item in metadataItemL:
                     
                self.siteD[ spectraD['sampleid'] ][item] = spectraD[item]
                
            samplemean = np.interp(self.outputWls, np.asarray(spectraD['wavelength']), np.asarray(spectraD['samplemean']), 
                                   left=self.params.xspectrolum.beginWaveLength-halfwlstep, 
                                   right=self.params.xspectrolum.endWaveLength+halfwlstep)
            
            #samplestd = np.interp(self.outputWls, np.asarray(spectraD['wavelength']), np.asarray(spectraD['samplestd']), 
            #                       left=self.params.xspectrolum.beginWaveLength-halfwlstep, 
            #                       right=self.params.xspectrolum.endWaveLength+halfwlstep)
            
            darkmean = np.interp(self.outputWls, np.asarray(spectraD['wavelength']), np.asarray(spectraD['darkmean']), 
                                   left=self.params.xspectrolum.beginWaveLength-halfwlstep, 
                                   right=self.params.xspectrolum.endWaveLength+halfwlstep)
            
            #darkstd = np.interp(self.outputWls, np.asarray(spectraD['wavelength']), np.asarray(spectraD['darkStd']), 
            #                       left=self.params.xspectrolum.beginWaveLength-halfwlstep, 
            #                       right=self.params.xspectrolum.endWaveLength+halfwlstep)
            
            
            # Calculate reflectance for this sample
            
            xSpectrA = (samplemean-darkmean)/(whiteRefSamplemean-whiteRefDarkmean)
            
            #sampleSignalStd = (samplestd-darkstd)/(whiteRefSamplestd-whiteRefDarkstd)
            
            #sampleStd = np.sqrt((5*samplestd*samplestd)+(5*darkstd*darkstd)/10)
            
            #reflectanceStd = np.sqrt((5*whiteRefStd*whiteRefStd)+(5*sampleStd*sampleStd)/10)
            
            
                
            self.xSpectraMetaD[ spectraD['sampleid'] ] = {'scandatebegin': spectraD['scandate'] ,
                                    'scandateend': spectraD['scandate'] ,
                                    'sampleprep': spectraD['prepcode'],
                                    'instrument': spectraD['source']}
                                
            self.xSpectraD[ spectraD['sampleid'] ] = xSpectrA
                
            self.xSpectraNumberOfwl = xSpectrA.shape[0]
                  
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
                     'geoRegion':self.params.campaign.geoRegion
                     }
        
        '''
                     'minLat':self.minLat,
                     'maxLat':self.maxLat,
                     'minLon':self.minLon,
                     'maxLon':self.maxLon,
                     }
        '''
        
        return campaignD
    
    def _ReportXspectreSiteMeta(self,site):
                                                             
        metaD = {'siteLocalId': self.siteD[site]['siteLocalId']} 
        
        metaD['dataset'] = self.siteD[site]['dataset'] 
                                
        metaD['minDepth'] = self.siteD[site]['minDepth']
        
        metaD['maxDepth'] = self.siteD[site]['maxDepth']
        
        return (metaD)
                 
    def _AssemblexSpectrajsonD(self):
        ''' Convert the extracted data to json objects for export
        '''
            
        projectD = self._SetProjectJson(self.xspectreModelN)
                
        projectD['campaign'] = self._SetCampaignD(self.xspectreModelN)
               
      
        projectD['waveLength'] = self.outputWls.tolist()
      
        varLD = []
        
        for site in self.siteD:

            metaD = self._ReportXspectreSiteMeta(site)
            
            jsonD = {'id':site, 'meta' : metaD}
            
            
            
            if site in self.labD:
                
                # Add the xspectre scan specific metadata for this layer
                for key in self.xSpectraMetaD[ site]:
                                  
                    metaD[key] = self.xSpectraMetaD[site][key] 
    
                # Add the xspectre spectral signal                                
                jsonD['signalMean'] = self.xSpectraD[site].tolist()
            
                jsonD['abundances'] = self.labD[site]
                              
            varLD.append(jsonD)
                
        projectD['spectra'] = varLD
                              
        # export, or dump, the assembled json objects      
        self._DumpSpectraJson(projectD, self.xspectreDataFPN, self.xspectreParamFPN , "xspectre")

                                    
    def PilotImport(self,sourcedatafolder, dstRootFP):
        ''' Steer the sequence of processes for extracting OSSL csv data to json objects
        ''' 
        
        # Set the source file names
        self._ListXspectreJsonFiles(sourcedatafolder)
        
        '''
        # Read the site data
        headers, rowL = ReadCSV(self.srcSoilSiteFPN)
        
        # Extract the site data
        self._ExtractSiteData(headers, rowL)
        
        # Read the laboratory (wet chemistry) data
        headers, rowL = ReadCSV(self.srcSoilLabFPN)
        
        # Extract the laboratory (wet chemistry) data
        self._ExtractLabData(headers, rowL)
        '''
        if self.params.xspectrolum.apply:
            
            # Set the destination file names 
            self.xspectreModelN, self.xspectreParamFPN, self.xspectreDataFPN  = self._SetDstFPN(dstRootFP,'xspectrolum',self.xspectrolum.subFP) 
      
            self._ExtractXspectreSpectraData()
            
            self._ExtractXspectreLabData()
                     
            self._AssemblexSpectrajsonD()
            
            
            
        
                                                      
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
        
        CreateArrangeParamJson(jsonFP,projFN,'importxspectre')
        
    jsonProcessObjectL = ReadProjectFile(dstRootFP, projFN, jsonFP)
           
    #Loop over all json files and create Schemas and Tables
    for jsonObj in jsonProcessObjectL:
        
        print ('    jsonObj:', jsonObj)

        paramD = ReadXspectreImportParamsJson(jsonObj)
        
        # Invoke the import
        xSpectre = ImportXspectre(paramD)
        
        xSpectre.PilotImport(sourcedatafolder, dstRootFP)
                    
if __name__ == "__main__":
    ''' If script is run as stand alone
    '''
            
    docpath = '/Users/thomasgumbricht/docs-local/OSSL/SU-Tovetorp/C14384MA-01'
    
    createjsonparams = False
    
    sourcedatafolder = 'data-xspectre'
    
    arrrangedatafolder = 'arranged-data'
    
    projFN = 'import_xspectre.txt'
    
    jsonpath = 'json-import'
    
    whiteReference = 'whiteref-start_c14384ma-01_raw-spectra_tovetorp_20230627_soil.csv'
    
    SetupProcesses(docpath, createjsonparams, sourcedatafolder, arrrangedatafolder, projFN, jsonpath)
