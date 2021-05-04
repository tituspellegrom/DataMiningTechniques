import pandas as pd

df = pd.read_pickle('../../df_features.pkl')


missing_percent = 100 * df.isnull().sum() / df.shape[0]
print("% missing:")

