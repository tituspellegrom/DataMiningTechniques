import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from model_dispatcher import load_npy_gzip
from sklearn.metrics import ndcg_score


def simple_rank(df):
    df['importance'] = df['booking_prob'] * 5 + df['click_prob']
    # df['importance'] = pd.concat([df['booking_prob']*5, df['click_prob']], axis=1).max(axis=1)

    df.sort_values(['srch_id', 'importance'], ascending=[True, False], inplace=True)

    return df


def apply_ndcg(df_g):
    true_rank = df_g['true_label'].sort_values(ascending=False).values.reshape(1, -1)
    score_rank = df_g['true_label'].values.reshape(1, -1)

    score = ndcg_score(true_rank, score_rank, k=5)
    return score


def rank_score(df_ids):

    srch_grouped = df_ids.groupby('srch_id').apply(apply_ndcg)

    mean_ndcg5 = srch_grouped.mean()

    return mean_ndcg5


def rank_predictions(model_name, validation_pred, ranking_algo=None):
    if ranking_algo is None:
        ranking_algo = simple_rank

    try:
        if validation_pred:
            df_ids = pd.read_csv(VAL_IDS)
            df_ids['true_label'] = np.load('y_val.npy')
            y_proba = load_npy_gzip(f'{VAL_FOLDER}{model_name}_proba.npy.gz')
        else:
            y_proba = load_npy_gzip(f'{TEST_FOLDER}{model_name}_proba.npy.gz')
            df_ids = pd.read_csv(TEST_IDS)
    except FileNotFoundError as e:
        print(e)
        return

    df_ids[['nothing_prob', 'click_prob', 'booking_prob']] = y_proba
    df_ids = ranking_algo(df_ids)

    if validation_pred:
        score = rank_score(df_ids)
        pd.DataFrame([(model_name, score)]).to_csv('val_ndcg_scores.csv', mode='a', index=False, header=False)
    else:
        df_ids[['srch_id', 'prop_id']].to_csv(f'kaggle_predictions/{model_name}.csv', index=False)


TEST_IDS = 'test_ids.csv'
TEST_FOLDER = 'test_predictions/'
VAL_IDS = 'val_ids.csv'
VAL_FOLDER = 'val_predictions/'

MODEL_FOLDER = 'models_saved/'
model_names = [fn.split('.pkl')[0] for fn in os.listdir(MODEL_FOLDER) if fn.endswith('.pkl')]
model_names = ['GradientBoostingClassifier_nodownsample']

for mdl_name in tqdm(model_names):
    rank_predictions(mdl_name, validation_pred=False)

