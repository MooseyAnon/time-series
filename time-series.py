import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller

# from matplotlib.pylab import rcParams

plt.rcParams['figure.figsize'] = 15, 6


dateparse = lambda date: pd.datetime.strptime(date, '%Y-%m')

data = pd.read_csv('AirPassengers.csv', parse_dates='Month', index_col='Month', date_parser=dateparse)


ts = data['#Passengers']

l = np.log(ts)

plt.plot(l)
plt.show(block=True)

def stationary_check(timeseries):

	rolmean = pd.rolling_mean(timeseries, window = 12)
	rolstd = pd.rolling_std(timeseries, window = 12)

	orig = plt.plot(timeseries, color='blue', label='Original')
	mean = plt.plot(rolmean, color='red', label='Rolling Mean')
	std = plt.plot(rolstd, color='green', label='Rolling Std')

	plt.legend(loc='best')

	plt.title('Rolling Mean and Std')
	
	



	print 'Results of Dicky Fuller Test: '

	dftest = adfuller(timeseries, autolag='AIC')
	dfout = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

	for key, value in dftest[4].items():
		dfout['Critical Value (%s)'%key] = value
	print dfout
	plt.show()

# stationary_check(ts)
