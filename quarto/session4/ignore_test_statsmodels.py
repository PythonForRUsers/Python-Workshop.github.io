import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices

df = sm.datasets.get_rdataset("Guerry", "HistData").data

vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']

df = df[vars]

df = df.dropna()


df[-5:]

y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')

mod = sm.OLS(y, X)    

