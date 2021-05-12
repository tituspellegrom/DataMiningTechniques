import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_validate, GroupShuffleSplit

import scipy.sparse as sp
import dask.array as da
from dask_ml.wrappers import Incremental
from dask_ml.model_selection import GridSearchCV
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import gc
import time
import dill
from collections import ChainMap
import joblib
import gzip

from collections import namedtuple


def save_model(model, model_name=None):
    if model_name is None:
        model_name = type(model).__name__

    if hasattr(model, 'cv_results_'):
        cv_result = pd.DataFrame(model.cv_results_)
        cv_result.to_excel(f"cv_output/{model_name}.xlsx", index=False)

    with open(f"models_saved/{model_name}.pkl", 'wb') as file:
        dill.dump(model, file)


Classifier = namedtuple('Classifier', 'name scikit_class sparse_support dask_support proba_support')


def obtain_scikit_classifiers(verbose=True, skip_classifiers={'ClassifierChain', 'MultiOutputClassifier',
                                                              'StackingClassifier', 'VotingClassifier',
                                                              'CategoricalNB', 'CalibratedClassifierCV',
                                                              'MultinomialNB', 'OneVsOneClassifier',
                                                              'OneVsRestClassifier', 'OutputCodeClassifier',
                                                              },
                              scikit_inconsistent_sparse={'CategoricalNB'}):

    estimators = all_estimators(type_filter='classifier')

    classifiers = []
    for name, ClassifierClass in estimators:
        if name in skip_classifiers:
            continue

        docstring = getattr(ClassifierClass, 'fit').__doc__

        sparse_support = docstring is not None and "X : {array-like, sparse matrix}" in docstring
        # Scikit is not consistent with docstrings & true support
        if name in scikit_inconsistent_sparse:
            sparse_support = False

        dask_support = hasattr(ClassifierClass, 'partial_fit')
        proba_support = hasattr(ClassifierClass, 'predict_proba')

        classifier = Classifier(name, ClassifierClass, sparse_support, dask_support, proba_support)
        classifiers.append(classifier)

        if verbose:
            print(f"{name}:\n"
                  f"\tsupports sparsity: {sparse_support}\n"
                  f"\tsupports dask training: {dask_support}\n"
                  f"\tsupports predict_proba(): {proba_support}")

    return classifiers


def prepare_data(dataset_name):
    arr = sp.load_npz(f"../{dataset_name}_data_merged.npz").tocsr()
    groups = np.load(f"../{dataset_name}_groups.npy")
    embedding_dims = np.genfromtxt(f'../{dataset_name}_embedding_dims.csv', delimiter=',')
    non_embedded = np.sum(embedding_dims[:-1] == 1)

    arr_train, arr_test, idx_train, idx_test = split_groups(arr, groups)
    del arr
    gc.collect()

    X_train = arr_train[:, :-1]
    y_train = np.ravel(arr_train[:, -1].todense(), 'C')
    groups_train = groups[idx_train]

    X_test = arr_test[:, :-1]
    y_test = np.ravel(arr_test[:, -1].todense(), 'C')
    groups_test = groups[idx_test]

    sc_X = MaxAbsScaler()

    X_train[:, :non_embedded] = sc_X.fit_transform(X_train[:, :non_embedded])
    X_test[:, :non_embedded] = sc_X.transform(X_test[:, :non_embedded])

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def split_groups(arr, groups):
    gss = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7).split(arr, groups=groups)

    idx_train, idx_test = next(gss)

    data_train, data_test = arr[idx_train, :], arr[idx_test, :]
    return data_train, data_test, idx_train, idx_test


def prepare_data_nonhot(dataset_name):
    arr = np.load(f"../{dataset_name}_nonhot.npy", allow_pickle=True)
    groups = np.load(f"../{dataset_name}_groups.npy")

    arr_train, arr_test, idx_train, idx_test = split_groups(arr, groups)

    del arr
    gc.collect()
    time.sleep(5)

    X_train= arr_train[:, :-1]
    y_train = np.ravel(arr_train[:, -1], 'C')

    groups_train = groups[idx_train]

    X_test = arr_test[:, :-1]
    y_test = np.ravel(arr_test[:, -1], 'C')
    groups_test = groups[idx_test]

    sc_X = MinMaxScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def prepare_classifier(classifier, model_params, hyper_opt=False):
    print(f"Preparing {classifier.name}")

    default_params = model_params.get('default_params', dict())
    hyper_params = model_params.get('hyper_params', dict())

    clf = classifier.scikit_class(**default_params)

    if hyper_opt:
        hyper_params_incl_default = [{**default_params, **hyper_dict}
                                     for hyper_dict in hyper_params]
        clf = GridSearchCV(clf, hyper_params_incl_default)

    return clf


def save_npy_gzip(file_name, arr):
    with gzip.GzipFile(file_name+'.gz', "w") as f:
        np.save(f, arr)


def load_npy_gzip(file_name):
    with gzip.GzipFile(file_name, 'r') as f:
        arr = np.load(f)

    return arr


def train_and_save(clfs_nt, clfs, X_train, X_test, y_train, y_test, groups_train, groups_test, dataset_name, start_after=None):
    # TODO: control memory usage => https://stackoverflow.com/questions/24406937/scikit-learn-joblib-bug-multiprocessing-pool-self-value-out-of-range-for-i-fo
    # https://github.com/scikit-learn/scikit-learn/issues/936

    save_npy_gzip(f"test_predictions/{dataset_name}_srchgroups.npy", groups_test)
    save_npy_gzip(f"test_predictions/{dataset_name}_testlabels.npy", groups_test)

    scores = []
    active = False
    for clf_nt, clf in tqdm(list(zip(clfs_nt, clfs))):
        print(f"Fitting: {type(clf)}")
        if not active:
            active = clf_nt.name == start_after
            continue

        # if parallel:
        #     with joblib.parallel_backend('dask'):
        #         clf.fit(X_train, y_train)
        #     continue

        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(e)
            continue

        # TODO: wrap CalibratedClassifierCV if no predict_proba()
        if clf_nt.proba_support:
            try:
                y_prob = clf.predict_proba(X_test)
                save_npy_gzip(f"test_predictions/{dataset_name}_{clf_nt.name}.npy", y_prob)
            except Exception as e:
                print(e)

            # arr = load_npy_gzip(f"test_predictions/{dataset_name}_{clf_nt.name}.npy.gz")
            # print(arr)

        score = clf.score(X_test, y_test)
        print(f"Test Score: {score}")

        scores += (clf_nt.name, score)
        save_model(clf, model_name=clf_nt.name)

    df_scores = pd.DataFrame(scores, columns=['classifier', 'Test Score'])
    df_scores.to_excel(f'test_predictions/{dataset_name}_scores.xlsx', index=False)


def main():
    from model_parameters import params

    DATASET_NAME = 'df_features'

    client = Client(processes=False)

    classifiers = obtain_scikit_classifiers()
    # slow_models = ['AdaBoostClassifier', 'ExtraTreesClassifier', 'MLPClassifier']
    slow_models = ['KNeighborsClassifier', 'LogisticRegressionCV', 'RadiusNeighborsClassifier', 'SVC']
    ignore_models = ['GaussianProcessClassifier', 'LabelPropagation', 'LabelSpreading', 'QuadraticDiscriminantAnalysis']
    clfs_nt = list(filter(lambda clf_nt: clf_nt.name not in slow_models + ignore_models,
                          classifiers))

    clfs = [prepare_classifier(clf_nt, model_params=params.get(clf_nt.name, dict()), hyper_opt=False)
            for clf_nt in clfs_nt]
    train_and_save(clfs_nt, clfs, *prepare_data_nonhot(DATASET_NAME), dataset_name=DATASET_NAME+'_dense',
                   start_after='RidgeClassifierCV')


    #
    #
    # dense_clfs_nt = list(filter(lambda clf_nt: clf_nt.sparse_support is False, clfs_nt))
    # sparse_clfs_nt = list(filter(lambda clf_nt: clf_nt.sparse_support is True, clfs_nt))
    #
    # dense_clfs = [prepare_classifier(clf_nt, model_params=params.get(clf_nt.name, dict()), hyper_opt=False)
    #               for clf_nt in dense_clfs_nt]
    #
    # # This takes way to long => use dense input for all models?
    # sparse_clfs = [prepare_classifier(clf_nt, model_params=params.get(clf_nt.name, dict()), hyper_opt=False)
    #                for clf_nt in sparse_clfs_nt]
    #
    # # QuadraticDiscriminanAnalysis predicts NaNs => because of collinear?
    # train_and_save(dense_clfs_nt, dense_clfs, *prepare_data_nonhot(DATASET_NAME), dataset_name=DATASET_NAME+'_dense')
    # train_and_save(sparse_clfs_nt, sparse_clfs, *prepare_data(DATASET_NAME), dataset_name=DATASET_NAME+'_sparse')

    # TODO: use some voting ensemble for best models
    # TODO: HyperOpt best models


if __name__ == '__main__':
    main()


# TODO: use truly chunked
# if classifier.dask_support:
#     # parallel_X_train = da.from_array(X_train, (1_000, X_train.shape[1]))
#     # parallel_y_train = da.from_array(y_train, (1_000, y_train.shape[1]))
#     cl = Incremental(cl)
#     with ProgressBar():
#         cl.fit(X_train, y_train, classes=(0, 1, 5))    # dask needs classes explicit
# else:
#     cl.fit(X_train, y_train)

