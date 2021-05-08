import pandas as pd

import pickle
import scipy.sparse as sp
from sklearn.model_selection import GroupShuffleSplit
from preprocessing import create_label_column
from sklearn.preprocessing import StandardScaler

def create_label_column(df):
    df['label'] = 0
    book_indices = df['booking_bool'] == 1
    click_indices = df['click_bool'] == 1

    df['label'].loc[click_indices] = 1
    df['label'].loc[book_indices] = 2


    return df

def preprocess2dot0(data_name):
    '''
    Preprocesses the input data to a dataframe with all id columns one-hot encoded
    :return: DataFrame with features and one-hot encoded categories. List of embedding dimensions needed for DeepFM
    '''
    print('Loading data...')

    df = pd.read_pickle(f'{data_name}.pkl')

    df = create_label_column(df)
    df.drop(['click_bool', 'booking_bool', 'date_time'], axis=1, inplace=True)

    id_cols = pd.read_csv(f'{data_name}_idcols.csv')['id_columns'].values.tolist()
    boolean_cols = pd.read_csv(f'{data_name}_booleancols.csv')['boolean_columns'].values.tolist()
    non_id_cols = list(set(df.columns) - set(id_cols) - set(boolean_cols) - {'label'})

    scaled_features = StandardScaler().fit_transform(df[non_id_cols].values)
    df[non_id_cols] = pd.DataFrame(scaled_features, index=df[non_id_cols].index, columns=non_id_cols)

    embedding_dims = [df[col].max()+1 for col in id_cols]

    df_cols = non_id_cols+boolean_cols+id_cols+['label']
    df = df.reindex(columns=df_cols)

    with open('embedding_dims.txt', 'wb') as fp:
        pickle.dump(embedding_dims, fp)

    train_inds, val_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7).split(df, groups=df['srch_id']))

    trainset = df.iloc[train_inds]
    valset = df.iloc[val_inds]

    trainset.to_pickle(f'{data_name}_train.pkl')
    valset.to_pickle(f'{data_name}_val.pkl')

if __name__ == '__main__':
    preprocess2dot0('df_temporary')
    # preprocess2('df_features')
