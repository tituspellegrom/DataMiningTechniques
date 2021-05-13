import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.utils import all_estimators


from dask_ml.model_selection import GridSearchCV
from dask.distributed import Client
from dask.diagnostics import ProgressBar
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


def prepare_classifier(CLF, model_params, hyper_opt=False):
    default_params = model_params.get('default_params', dict())
    hyper_params = model_params.get('hyper_params', dict())

    clf = CLF(**default_params)

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


def train_and_save(classifiers_prepped, X_train, X_test, y_train, y_test, start_after=None):

    active = False
    for clf_nm, clf in tqdm(classifiers_prepped):
        print(f"Fitting: {clf_nm}")
        if not active and start_after is not None:
            active = clf_nm == start_after
            continue

        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(e)
            continue

        # TODO: wrap CalibratedClassifierCV if no predict_proba()
        try:
            y_prob = clf.predict_proba(X_test)
            save_npy_gzip(f"val_predictions/{clf_nm}_proba.npy", y_prob)
            # arr = load_npy_gzip(f"test_predictions/{dataset_name}_{clf_nt.name}.npy.gz")
        except Exception as e:
            print(e)

        try:
            y_pred = clf.predict(X_test)
            save_npy_gzip(f"val_predictions/{clf_nm}.npy", y_pred)
        except Exception as e:
            print(e)

        score = clf.score(X_test, y_test)
        print(f"Test Score: {score}")

        pd.DataFrame([(clf_nm, score)]).to_csv('val_predictions/scores.csv', mode='a', index=False, header=False)
        save_model(clf, model_name=clf_nm)


def main():
    from model_parameters import params

    client = Client(processes=False)

    X_train, X_val = np.load('X_train.npy'), np.load('X_val.npy')
    y_train, y_val = np.load('y_train.npy'), np.load('y_val.npy')

    skip_classifiers = {'ClassifierChain', 'MultiOutputClassifier',
                        'StackingClassifier', 'VotingClassifier',
                        'CategoricalNB', 'CalibratedClassifierCV',
                        'MultinomialNB', 'OneVsOneClassifier',
                        'OneVsRestClassifier', 'OutputCodeClassifier',
                        'CategoricalNB', 'GaussianProcessClassifier', 'LabelPropagation',
                        'LabelSpreading', 'QuadraticDiscriminantAnalysis'}
    slow_models = {'KNeighborsClassifier', 'LogisticRegressionCV', 'RadiusNeighborsClassifier', 'SVC'}

    classifiers = [(nm, CLF) for nm, CLF in all_estimators(type_filter='classifier')
                  if nm not in skip_classifiers | slow_models]

    classifiers_prepped = [(clf_nm, prepare_classifier(CLF, model_params=params.get(clf_nm, dict()), hyper_opt=False))
                           for clf_nm, CLF in classifiers]

    train_and_save(classifiers_prepped, X_train, X_val, y_train, y_val, start_after=None)

    # TODO: use some voting ensemble for best models
    # TODO: HyperOpt best models


if __name__ == '__main__':
    main()

