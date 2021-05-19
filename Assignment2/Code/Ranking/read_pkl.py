# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:35:46 2021

@author: doist
"""

import pickle

path = 'C:/Users/doist/OneDrive/Documenten/GitHub/DataMiningTechniques/Assignment2/Code/Ranking/'
with open(path+'gwo_summary.pkl', 'rb') as handle:
    gwo_summary = pickle.load(handle)
with open(path+'pso_summary.pkl', 'rb') as handle:
    pso_summary = pickle.load(handle)
        
print(gwo_summary)
print(pso_summary)