'''
Z value for the identification of outliers 
'''
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv('Data.csv')

'''
Creating the numpy array from the dataframe with respective Z values

'''
z = np.abs(stats.zscore(df))

'''
Creating theframe for the vertical append

'''
z_frame = pd.DataFrame(data=z, columns=["zscore"])

'''
Vertical append
'''

df_new=pd.concat([df,z_frame], axis=1)

'''
Just apply filter above a threshold: 2 or 95%
'''

new=df_new[df_new.zscore<2]


    