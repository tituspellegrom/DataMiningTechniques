params = {'AdaBoostClassifier': {
              'default_params': {},
              'hyper_params': [
                                {
                                    'n_estimators': list(range(10, 200, 10)),
                                    'learning_rate': [.6, .8, 1., 1.2, 1.4, 1.6]
                                }
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
          'GradientBoostingClassifier': {
              'default_params': {},
              'hyper_params': [
                  {
                      'loss': ['deviance', 'exponential'],
                      'learning_rate': [.2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.],
                      'n_estimators': range(20, 200, 20),
                      'sub_sample': [.5, .6, .7, .8, .9, 1.],
                      'criterion': ['friedman_mse', 'mse', 'mae'],
                      'min_samples_split': list(range(1, 5)),
                      'max_depth': list(range(2, 10, 1)),
                      'max_features': ['auto', 'sqrt', 'log2']
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


