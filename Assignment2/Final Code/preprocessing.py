import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_validate, GroupShuffleSplit


TRAIN_SET = 'training_set_VU_DM.pkl'
TEST_SET = 'test_set_VU_DM.pkl'


def prepare_train_set():

    df_train = pd.read_pickle(TRAIN_SET)

    missing = df_train.isnull().sum()
    describ = df_train.describe()

    gss = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7).split(df_train, groups=df_train['srch_id'])
    idx_train, idx_val = next(gss)

    X, y = df_train.loc[:, df_train.columns != 'label'].to_numpy(), df_train['label'].to_numpy()
    X = X.astype(float)
    y = y.astype(float)

    X_train, X_val = X[idx_train, :], X[idx_val, :]
    y_train, y_val = y[idx_train], y[idx_val]

    train_ids = df_train.loc[idx_train, ['srch_id', 'prop_id']]
    val_ids = df_train.loc[idx_val, ['srch_id', 'prop_id']]

    train_ids.to_csv('train_ids.csv', index=False)
    val_ids.to_csv('val_ids.csv', index=False)

    sc_X = MinMaxScaler()
    X_train = sc_X.fit_transform(X_train)
    X_val = sc_X.transform(X_val)

    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)

    return sc_X


def prepare_test_set(sc_X):
    df_test = pd.read_pickle(TEST_SET)

    describ = df_test.describe()

    test_ids = df_test[['srch_id', 'prop_id']]
    test_ids.to_csv('test_ids.csv', index=False)
    X_test = df_test.to_numpy().astype(float)

    X_test = sc_X.transform(X_test)
    np.save('X_test.npy', X_test)


def main():
    sc_X = prepare_train_set()
    prepare_test_set(sc_X)

    # TODO: Prepare train set once again without validation split


if __name__ == '__main__':
    main()



# TODO: collect all classifiers
# TODO: Train some classifiers
# TODO: Report scores on validation set

# TODO: Run Predictions on vu test set
# TODO: Write simple ranking
# TODO: Write Kaggle Output file

# TODO: Upload to kaggle

