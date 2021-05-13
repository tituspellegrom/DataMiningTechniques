import pandas as pd
from tqdm import tqdm
import numpy as np
import dill


MODEL_NAME = 'AdaBoostClassifier'
X_test = np.load('X_test.npy')


model_file = f'models_saved/{MODEL_NAME}.pkl'
clf = dill.load(open(model_file, 'rb'))

y_pred = clf.predict(X_test)

entry_data = pd.read_csv('test_ids.csv')
entry_data['label_pred'] = y_pred

entry_data.sort_values(['srch_id', 'label_pred'], ascending=[True, False], inplace=True)

df_out = entry_data[['srch_id', 'prop_id']].rename(columns={'srch_id': 'SearchId', 'prop_id': 'PropertyId'})
df_out.to_csv(f'kaggle_prediction_{MODEL_NAME}.csv', index=False)

# TODO: sort within class on prediction probability
