'''
Created on 7 Aug 2023

@author: thomasgumbricht
'''

import datetime

def Today():
    
    return datetime.datetime.now().date().strftime("%Y%m%d")