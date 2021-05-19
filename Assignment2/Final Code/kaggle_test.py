import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from model_dispatcher import load_npy_gzip
from sklearn.metrics import ndcg_score


def weight_rank(df, weights):
    df['importance'] = weights[0] * df['nothing_prob'] + weights[1] * df['click_prob'] + weights[2] * df['booking_prob']

    df.sort_values(['srch_id', 'importance'], ascending=[True, False], inplace=True)

    return df


def simple_rank(df):

    return weight_rank(df, (0, 1, 5))
    #
    # df['importance'] = df['booking_prob'] * 5 + df['click_prob']
    # # df['importance'] = pd.concat([df['booking_prob']*5, df['click_prob']], axis=1).max(axis=1)
    #
    # df.sort_values(['srch_id', 'importance'], ascending=[True, False], inplace=True)
    #
    # return df


def apply_ndcg(df_g):
    true_rank = df_g['true_label'].sort_values(ascending=False).values.reshape(1, -1)
    score_rank = df_g['true_label'].values.reshape(1, -1)

    score = ndcg_score(true_rank, score_rank, k=5)
    return score


def rank_score(df_ids):

    srch_grouped = df_ids.groupby('srch_id').apply(apply_ndcg)

    mean_ndcg5 = srch_grouped.mean()

    return mean_ndcg5


def rank_predictions(model_name, validation_pred, ranking_algo=None, rank_name=None):
    if ranking_algo is None:
        ranking_algo = simple_rank
    if rank_name is None:
        rank_name = ranking_algo.__name__

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
        print(score)
        pd.DataFrame([(f'{model_name}_{rank_name}', score)]).to_csv('val_ndcg_scores.csv', mode='a', index=False, header=False)
    else:
        df_ids[['srch_id', 'prop_id']].to_csv(f'kaggle_predictions/{model_name}_{rank_name}.csv', index=False)


TEST_IDS = 'test_ids.csv'
TEST_FOLDER = 'test_predictions/'
VAL_IDS = 'val_ids.csv'
VAL_FOLDER = 'val_predictions/'
MODEL_FOLDER = 'models_saved/'
RUN_NAME = 'top8_downsample'


def main():
    model_names = [fn.split('.pkl')[0] for fn in os.listdir(MODEL_FOLDER) if fn.endswith(f'{RUN_NAME}.pkl')]
    # model_names = ['HistGradientBoostingClassifier_cv_test']
    # model_names = ['GradientBoostingClassifier_nodownsample']

    grey_wolf_weights = [0.8450970154270735, 0.9139184920050212, 1.3436247928017258]
    for mdl_name in tqdm(model_names):
        # rank_predictions(mdl_name, validation_pred=False,
        #                  ranking_algo=lambda df: weight_rank(df, grey_wolf_weights), rank_name='gwo')
        rank_predictions(mdl_name, validation_pred=False)


if __name__ == '__main__':
    main()


