import pandas as pd
from tqdm import tqdm
import numpy as np
import dill
import os
from model_dispatcher import save_npy_gzip, load_npy_gzip

entry_data = pd.read_csv('test_ids.csv')


MODEL_NAME = 'ExtraTreesClassifier'
TEST_PREDICTIONS_FOLDER = 'test_predictions/'

proba_order = np.load(f'{TEST_PREDICTIONS_FOLDER}{MODEL_NAME}_proba_order.npy')
print(proba_order)
y_pred = load_npy_gzip(f'{TEST_PREDICTIONS_FOLDER}{MODEL_NAME}_pred.npy.gz')
y_proba = load_npy_gzip(f'{TEST_PREDICTIONS_FOLDER}{MODEL_NAME}_proba.npy.gz')

entry_data[proba_order] = y_proba
entry_data.rename(columns={0.0: 'nothing', 1.0: 'click', 5.0: 'booking'}, inplace=True)
entry_data['importance'] = pd.concat([entry_data['booking']*5, entry_data['click']], axis=1).max(axis=1)

entry_data.sort_values(['srch_id', 'importance'], ascending=[True, False], inplace=True)
print(entry_data.head())

df_out = entry_data[['srch_id', 'prop_id']]
df_out.to_csv(f'kaggle_prediction_{MODEL_NAME}.csv', index=False)


# TODO: sort within class on prediction probability
