'''
Created on 7 Aug 2023

@author: thomasgumbricht
'''

import json

import numpy as np

import pandas as pd

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

def DumpAnyJson(dumpD, jsonFPN):
    ''' dump, any json object
    
    :param exportD: formatted dictionary
    :type exportD: dict
    '''
            
    jsonF = open(jsonFPN, "w")
  
    json.dump(dumpD, jsonF, indent = 2)
  
    jsonF.close()
    
def LoadBandData(columns, SpectraD):
 
    ''' Read json data into numpy array and convert to pandas dataframe
    '''
                                     
    n = 0
                   
    # Loop over the spectra
    for sample in SpectraD['spectra']:
                                
        if n == 0:
        
            spectraA = np.asarray(sample['signalMean'])
        
        else:
             
            spectraA = np.vstack( (spectraA, np.asarray(sample['signalMean']) ) )
        
        n += 1
                           
    return pd.DataFrame(data=spectraA, columns=columns)
    
    