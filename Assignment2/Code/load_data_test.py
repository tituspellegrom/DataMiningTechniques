import pandas as pd

from preprocessing import preprocess
import time

preprocess()

start = time.time()
data = pd.read_pickle('data_merged.pkl')
end = time.time()
print(end-start)




start = time.time()
data = pd.read_hdf('data_merged.h5')
end = time.time()
print(end-start)

