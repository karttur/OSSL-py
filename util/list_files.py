'''
Created on 13 Jan 2023

@author: thomasgumbricht
'''

from glob import glob

from os.path import join

from pathlib import Path

import csv

def GlobGetFileList(FP, patternL):
    '''
    '''
    
    fL = []
        
    for pattern in patternL:
                    
        fL.extend( glob(join( FP,pattern ) ) )
    
    return (fL)

def PathLibGetFileList(FP, patternL):
    '''
    '''
    
    fL = []
    
    for pattern in patternL:
            
        for path in Path(FP).rglob(pattern):
            
            fL.appendd( path.name )
            
def CsvFileList(csvFPN):
    '''
    '''
    
    fL = []
    with open(csvFPN, 'r') as csvfile:

        # the delimiter depends on how your CSV seperates values
        csvReader = csv.reader(csvfile)

        for row in csvReader:
            
            # check if row is empty
            if row[0][0] == '#' or len(row[0])<4: 
                   
                continue
            
            fL.append(row[0])    
   
    return fL