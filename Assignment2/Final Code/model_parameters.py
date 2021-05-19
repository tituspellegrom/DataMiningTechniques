from itertools import permutations
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC

params = {'AdaBoostClassifier': {
              'default_params': {},
              'hyper_params': [
                                {
                                    'base_estimator': [None, LogisticRegression(solver='saga')],
                                    'learning_rate': [0.01, 0.1, 1, 10],
                                    'n_estimators': [5, 50, 100, 250]
                                },
                               ]
          },
          'BaggingClassifier': {
              'default_params': {},
              'hyper_params': [
                                {
                                    'n_estimators': list(range(5, 50, 5)),
                                    'max_samples': [.5, .6, .7, .8, .9, 1.],
                                    'bootstrap': [True, False],
                                }
                              ]
          },
          'BernoulliNB': {
              'default_params': {},
              'hyper_params': [
                  {
                      'alpha': [.2, .4, .6, .8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                      'binarize': [0.0, .2, .4, .6, .8, 1.0]
                  }
              ]
          },
          'ComplementNB': {
              'default_params': {},
              'hyper_params': [
                  {
                      'alpha': [.2, .4, .6, .8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
                  }
              ]
          },
          'DecisionTreeClassifier': {
              'default_params': {},
              'hyper_params': [
                  {
                      'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'max_features': ['auto', 'sqrt', 'log2']
                  }
              ]
          },
          'DummyClassifier': {
              'default_params': {},
              'hyper_params': [
                  {
                      'strategy': ['stratified', 'most_frequent', 'prior', 'uniform'],
                  }
              ]
          },
          'ExtraTreesClassifier': {
              'default_params': {},
              'hyper_params': [
                  {
                      'n_estimators': [range(10, 200, 10)],
                      'criterion': ['gini', 'entropy'],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'bootstrap': [False, True]
                  }
              ]
          },
          'GaussianNB': {
              'default_params': {},
              'hyper_params': [
                  {
                      'var_smoothing': [10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2],
                  }
              ]
          },
          'GaussianProcessClassifier': {
              'default_params': {
                  'copy_X_train': False
              },
              'hyper_params': [
                  {}
              ]
          },
          'GradientBoostingClassifier': {
              'default_params': {},
              'hyper_params': [
                  {
                      'learning_rate': [0.01, 0.1, 1, 10, 100],
                      'n_estimators': [5, 50, 250, 500],
                      'max_depth': [1, 3, 5, 7, 9]
                  }
              ]
          },
          'HistGradientBoostingClassifier': {
              'default_params': {},
              'hyper_params': [
                  {
                      # 'learning_rate': [.2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.],
                      'learning_rate': [0.01, 0.1, 1, 10, 100],
                      'max_depth': [1, 3, 5, 7, 9]
                      # 'l2_regularization': [.1, .2, 3., .4, .5, .6, .7, .8, .9, 1.],
                      # 'l2_regularization': [.5, 1.]
                  }
              ]
          },
          'KNeighborsClassifier': {
              'default_params': {},
              'hyper_params': [
                  {
                        'n_neighbors': list(range(1, 10)),
                        'weights': ['uniform', 'distance'],
                        'leaf_size': list(range(10, 50, 5)),
                        'p': list(range(4))
                  }]
          },
          'LabelPropagation': {
              'default_params': {},
              'hyper_params': [
                  {
                        'kernel': ['rbf'],
                        'gamma': list(range(1, 100, 5))
                  },
                  {
                        'kernel': ['knn'],
                        'gamma': list(range(1, 25, 1))
                  }
              ]
          },
          'LabelSpreading': {
              'default_params': {},
              'hyper_params': [
                  {
                      'kernel': ['rbf'],
                      'gamma': list(range(1, 100, 5)),
                      'alpha': [0., .2, .4, .6, .8, 1.]
                  },
                  {
                      'kernel': ['knn'],
                      'gamma': list(range(1, 25, 1)),
                      'alpha': [0., .2, .4, .6, .8, 1.]
                  }
              ]
          },
          'LinearDiscriminantAnalysis': {
              'default_params': {},
              'hyper_params': [
                  {}
              ]
          },
          'LinearSVC': {
              'default_params': {},
              'hyper_params': [
                  {
                      'penalty': ['l1', 'l2'],
                      'loss': ['hinge', 'squared_hinge'],
                      'dual': [True, False],
                      'C': [1, 10, 100, 1000]
                  }
              ]
          },
          'LogisticRegression': {
              'default_params': {
                  'solver': 'saga'
              },
              'hyper_params': [
                  {
                      'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                      'dual': [True, False],
                      'C': [1, 10, 100, 1000],
                      'solver': ['saga'],
                      'max_iter': [100, 200, 500]
                  }
              ]
          },
          'LogisticRegressionCV': {
              'default_params': {
                  'solver': 'saga'
              },
              'hyper_params': [
                  {
                      'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                      'dual': [True, False],
                      'Cs': [1, 10, 100, 1000],
                      'solver': ['saga'],
                      'max_iter': [100, 200, 500]
                  }
              ]
          },
          'MLPClassifier': {
              'default_params': {},
              'hyper_params': [
                  # {
                  #     # 'hidden_layer_sizes': [(5, ), (50, ), (100,), (200,) tuple(lay)
                  #     #                        for i in range(1, 3)
                  #     #                        for lay in permutations([5, 50, 250], i)],
                  #     'hidden_layer_sizes': [(5, 5), (50, 50), (100, 100)],
                  #     'alpha': [0.0001, 0.05],
                  #     'learning_rate': ['constant', 'adaptive']
                  # },
                  {
                      'hidden_layer_sizes': [(5, 5), (50, 50), (100, )],
                      'alpha': [0.0001, 0.05],
                      'learning_rate': ['constant', 'adaptive']
                  },
                  # {
                  #     'hidden_layer_sizes': [(5, 5)],
                  #     'alpha': [0.0001],
                  #     'learning_rate': ['adaptive']
                  # },
              ]
          },
          'NearestCentroid': {
              'default_params': {},
              'hyper_params': []
          },
          'NuSVC': {
              'default_params': {
                  'probability': True
              },
              'hyper_params': [
                  {
                      'nu': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
                      'kernel': ['linear', 'rbf', 'sigmoid'],
                      'gamma': ['scale', 'auto'],
                      'shrinking': [True, False]
                  },
                  {
                      'nu': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
                      'kernel': ['poly'],
                      'degree': [2, 3, 4, 5],
                      'gamma': ['scale', 'auto'],
                      'shrinking': [True, False]
                  }
              ]
          },
          'PassiveAggressiveClassifier': {
              'default_params': {},
              'hyper_params': [
                  {
                      'C': [1., 10., 100., 1000.],
                      'early_stopping': [False, True],
                      'shuffle': [False, True]
                  },
              ]
          },
          'Perceptron': {
              'default_params': {},
              'hyper_params': [
                  {
                      'penalty': ['l2', 'l1', 'elasticnet', None],
                      'early_stopping': [False, True]
                  }
              ]
          },
          'QuadraticDiscriminantAnalysis': {
              'default_params': {},
              'hyper_params': [
                  {}
              ]
          },
          'RadiusNeighborsClassifier': {
              'default_params': {},
              'hyper_params': [
                  {
                      'radius': [.5, .75, 1., 1.25, 1.5],
                      'weights': ['uniform', 'distance'],
                      'leaf_size': list(range(10, 50, 5)),
                      'p': list(range(4)),
                      'outlier_label': ['most_frequent']
                  }
              ]
          },
          'RandomForestClassifier': {
              'default_params': {},
              'hyper_params': [
                  {
                      'n_estimators': [range(10, 200, 10)],
                      'criterion': ['gini', 'entropy'],
                      'max_features': ['auto', 'sqrt', 'log2'],
                      'bootstrap': [False, True]
                  }
              ]
          },
          'RidgeClassifier': {
            'default_params': {},
            'hyper_params': [
                {
                    'alpha': [.1, .5, 1., 1.5, 2., 5., 10.],
                    'solver': ['saga']
                }
            ]
          },
          'RidgeClassifierCV': {
            'default_params': {},
            'hyper_params': [
                {
                }
            ]
          },
          'SGDClassifier': {
            'default_params': {},
            'hyper_params': [
                {
                    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'early_stopping': [False, True]
                }
            ]
          },
          'SVC': {
              'default_params': {
                  'probability': True
              },
              'hyper_params': [
                  {
                      'C': [1, 10, 100, 1000],
                      'kernel': ['linear', 'rbf', 'sigmoid'],
                      'gamma': ['scale', 'auto'],
                      'shrinking': [True, False]
                  },
                  {
                      'C': [1, 10, 100, 1000],
                      'kernel': ['poly'],
                      'degree': [2, 3, 4, 5],
                      'gamma': ['scale', 'auto'],
                      'shrinking': [True, False]
                  }
              ]
          }
}


