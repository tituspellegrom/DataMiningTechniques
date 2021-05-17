# -*- coding: utf-8 -*-
"""
Created on Sat May 15 13:06:48 2021

@author: doist
"""

import gzip
import pandas as pd
import numpy as np
import copy

def import_file(file):
    data_file = gzip.open(file)
    data = np.load(data_file)
    return data

def prep_data(path, pred_file):
    # Import data
    prob_predictions = import_file(path+'val_predictions/'+str(pred_file))
    ids = pd.read_csv(path+'val_ids.csv', sep=',')
    actual = np.load(path+'y_val.npy')

    # Prepare data df
    data = pd.DataFrame(columns=['search_id', 'hotel_id', 'prob0', 'prob1', 'prob5'])
    data['search_id'] = copy.copy(ids['srch_id'])
    data['hotel_id'] = copy.copy(ids['prop_id'])
    data['prob0'] = [item[0] for item in prob_predictions]
    data['prob1'] = [item[1] for item in prob_predictions]
    data['prob5'] = [item[2] for item in prob_predictions]
    
    # Prepare true_labels df
    true_labels = pd.DataFrame(columns=['search_id', 'hotel_id', 'true_label'])
    true_labels['search_id'] = copy.copy(ids['srch_id'])
    true_labels['hotel_id'] = copy.copy(ids['prop_id'])
    true_labels['true_label'] = actual
    
    return data, true_labels