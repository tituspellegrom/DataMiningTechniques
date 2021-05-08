import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_validate

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


def prepare_data(input_file_name):
    arr = sp.load_npz(input_file_name).tocsr()
    # arr = da.from_array(arr)
    X = arr[:, :-1]
    y = np.ravel(arr[:, -1].todense(), 'C')

    del arr
    gc.collect()

    time.sleep(5)  # give gc some time

    embedding_dims = np.genfromtxt(f'../df_features_embedding_dims.csv', delimiter=',')
    non_embedded = np.sum(embedding_dims[:-1] == 1)

    # TODO: keep searches together
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=1)

    sc_X = MaxAbsScaler()

    X_train[:, :non_embedded] = sc_X.fit_transform(X_train[:, :non_embedded])
    X_test[:, :non_embedded] = sc_X.transform(X_test[:, :non_embedded])

    return X_train, X_test, y_train, y_test


def prepare_data_nonhot(input_file_name):
    arr = np.load(input_file_name)
    X = arr[:, :-1]
    y = arr[:, -1]
    del arr
    gc.collect()
    time.sleep(5)  # give gc some time

    # TODO: keep searches together
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=1)

    sc_X = MinMaxScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_classifier(classifier, X_train, y_train, model_params, hyper_opt=False):
    print(f"Fitting {classifier.name}")

    default_params = model_params.get('default_params', dict())
    hyper_params = model_params.get('hyper_params', dict())

    cl = classifier.scikit_class(**default_params)

    # TODO: use truly chunked
    # if classifier.dask_support:
    #     # parallel_X_train = da.from_array(X_train, (1_000, X_train.shape[1]))
    #     # parallel_y_train = da.from_array(y_train, (1_000, y_train.shape[1]))
    #     cl = Incremental(cl)
    #     with ProgressBar():
    #         cl.fit(X_train, y_train, classes=(0, 1, 5))    # dask needs classes explicit
    # else:
    #     cl.fit(X_train, y_train)

    if hyper_opt:
        hyper_params_incl_default = [{**default_params, **hyper_dict}
                                     for hyper_dict in hyper_params]
        cl = GridSearchCV(cl, hyper_params_incl_default)

    cl.fit(X_train, y_train)

    # TODO: wrap CalibratedClassifierCV if needed and activate predict_proba()
    save_model(cl, model_name=classifier.name)

    return cl


def main():
    from model_parameters import params

    client = Client(processes=False)

    classifiers = obtain_scikit_classifiers()

    slow_models = ['AdaBoostClassifier', 'ExtraTreesClassifier']

    for classifier in tqdm(classifiers):
        # ignore for now
        if classifier.name in slow_models:
            continue

        # reload every loop to avoid memory persisting
        if classifier.sparse_support:
            X_train, X_test, y_train, y_test = prepare_data('../df_temporary_data_merged.npz')
        else:
            X_train, X_test, y_train, y_test = prepare_data_nonhot('../df_temporary_nonhot.npy')

        cl = train_classifier(classifier, X_train, y_train,
                              model_params=params.get(classifier.name, dict()), hyper_opt=True)
        y_prob = cl.predict_proba(X_test)
        score = cl.score(X_test, y_test)

        print(f"{classifier.name}: {score}")

        del X_train, X_test, y_train, y_test
        gc.collect()
        time.sleep(5)

    # TODO: use some voting ensemble for best models


if __name__ == '__main__':
    main()


# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]





