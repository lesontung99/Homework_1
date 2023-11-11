import numpy as np


filepath = 'data\univariateData.dat'
alldata = np.loadtxt(filepath, delimiter=',')
x = np.matrix(alldata[:,:-1])
y = np.matrix((alldata[:,-1])).T

# Better initialize it here

initial = np.matrix(np.zeros((d,1)))
n,d = x.shape

from test_linreg_univariate import plotRegLine1D
from linreg import LinearRegression
x = np.c_[np.ones((n,1)), x]
lr_model = LinearRegression(init_theta = initial, alpha = 0.01, n_iter=1500)
lr_model.fit(x,y)
plotRegLine1D(lr_model,x,y)


