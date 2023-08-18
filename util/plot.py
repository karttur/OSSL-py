'''
Created on 8 Aug 2023

@author: thomasgumbricht
'''


def SetTextPos(x, y, xmin, xmax, ymin, ymax):
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
    
    x = x*(xmax-xmin)+xmin
    
    y = y*(ymax-ymin)+ymin
    
    return (x,y)


    