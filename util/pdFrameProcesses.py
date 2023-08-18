'''
Created on 7 Aug 2023

@author: thomasgumbricht
'''

def SpectraDerivativeFromDf(dataFrame,columns):
    ''' Create spectral derivates
    '''

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