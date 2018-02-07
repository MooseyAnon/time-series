import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller


plt.rcParams['figure.figsize'] = 15, 6


dateparse = lambda date: pd.datetime.strptime(date, '%Y-%m')

data = pd.read_csv('AirPassengers.csv', parse_dates='Month', index_col='Month', date_parser=dateparse)


ts = data['#Passengers']

l = np.log(ts)



moving_av = pd.rolling_mean(l, 12)

# plt.plot(l)
# plt.plot(moving_av)
# plt.show(block=True)

# tma = moving_av - l
# tma.dropna(inplace=True)
# print tma.head(12)

# exponentially weighted moving average
ewa = pd.ewma(l, halflife=12)

log_ewa_diff = l - ewa








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
		dfout['Critical Value ({0})'.format(key)] = value
	print dfout
	plt.show()

# stationary_check(log_ewa_diff)


decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(l)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(l, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()



#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(log_ewa_diff, nlags=20)
lag_pacf = pacf(log_ewa_diff, nlags=20, method='ols')


#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(log_ewa_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(log_ewa_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(log_ewa_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(log_ewa_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

plt.show()







from statsmodels.tsa.arima_model import ARIMA

# AR Model
model = ARIMA(l, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(log_ewa_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-log_ewa_diff)**2))

plt.show()

# MA Model
model = ARIMA(l, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(log_ewa_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-log_ewa_diff)**2))

plt.show()

# Combined Model
model = ARIMA(l, order=(2, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(log_ewa_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-log_ewa_diff)**2))

plt.show()





