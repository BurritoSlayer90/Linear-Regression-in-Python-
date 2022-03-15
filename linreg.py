import pandas as pd
import statsmodels.api as sm
import numpy as np

#making our dataframe for the regression
df = pd.read_csv('nbastuff.csv')
dataframe = pd.DataFrame(df, columns = ['PTS','Rk','FG'])
dataframe = dataframe.dropna()

#outcome variable
Y = dataframe[['PTS']]
#regressors
X = dataframe[['Rk','FG']]
#adding a constant term
X = sm.add_constant(X)
#the model
model = sm.OLS(Y, X).fit()
#pprinting the summary
print(model.summary())
