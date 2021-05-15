import pandas as pd
from tqdm import tqdm
import numpy as np
import dill
import os
from model_dispatcher import save_npy_gzip, load_npy_gzip


MODEL_FOLDER = 'models_saved/'
TEST_PREDICTION_FOLDER = 'test_predictions/'

models_fn = [fn for fn in os.listdir(MODEL_FOLDER) if fn.endswith('.pkl')]

X_test = np.load('X_test.npy')

models_fn = ['GradientBoostingClassifier_nodownsample.pkl']
for fn in models_fn:
    print(f'Predicting Model: {fn}')

    clf = dill.load(open(MODEL_FOLDER+fn, 'rb'))
    mdl_name = fn.split('.pkl')[0]

    print(f"Probability Prediction")
    y_proba = clf.predict_proba(X_test)
    save_npy_gzip(f'{TEST_PREDICTION_FOLDER}{mdl_name}_proba.npy', y_proba)

# entry_data = pd.read_csv('test_ids.csv')
# 
# model_file = f'models_saved/{MODEL_NAME}.pkl'
# clf = dill.load(open(model_file, 'rb'))
# 
# y_pred = clf.predict(X_test)
# y_proba = clf.predict_proba(X_test)
# 
# 
# entry_data['label_pred'] = y_pred
# entry_data.sort_values(['srch_id', 'label_pred'], ascending=[True, False], inplace=True)
# 
# df_out = entry_data[['srch_id', 'prop_id']].rename(columns={'srch_id': 'SearchId', 'prop_id': 'PropertyId'})
# df_out.to_csv(f'kaggle_prediction_{MODEL_NAME}.csv', index=False)
