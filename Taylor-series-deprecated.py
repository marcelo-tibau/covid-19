import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 



brazilAltPred = pd.read_csv('brazil-alt-prediction.csv', sep = ',')

brazilDeathPred = pd.read_csv('brazil-deathBrazil.csv', sep = ',')







# Taylor series
#Using the first and second derivatives of the continuous approximation of your usage data


import scipy.interpolate
import matplotlib
import scipy.misc


# Normalize time series data


series = brazilAltPred
deathSeries = brazilDeathPred

series['date'] = pd.to_datetime(series['date']).astype('int64')
max_a = series.date.max()
min_a = series.date.min()
min_norm = -1
max_norm =1
series['NORMA'] = (series.date- min_a) *(max_norm - min_norm) / (max_a-min_a) + min_norm

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15,16, 17, 18, 19, 20, 21,
                  22, 23]).astype('int64')

x = np.array(series.number_days)
    
y = np.array(series.total_cases)

z = np.array(series.NORMA)

z = np.array(deathSeries.number_days)

w = np.array(deathSeries.total_deaths)

# interpolate to approximate a continuous version of hard drive usage over time
f = scipy.interpolate.interp1d(x, y, kind='quadratic')

# approximate the first and second derivatives near the last point (2015)
dx = 0.01
x0 = x[-1] - 2*dx
first = scipy.misc.derivative(f, x0, dx=dx, n=1)
second = scipy.misc.derivative(f, x0, dx=dx, n=2)

# taylor series approximation near x[-1]
forecast = lambda x_new: np.poly1d([second/2, first, f(x[-1])])(x_new - x[-1])

forecast(23)  # 11.9

xs = np.arange(152, 185)
ys = forecast(xs)


# interpolate to approximate a continuous version of hard drive usage over time
f = scipy.interpolate.interp1d(z, w, kind='quadratic')

# approximate the first and second derivatives near the last point (2015)
dx = 0.01
x0 = x[-1] - 2*dx
first = scipy.misc.derivative(f, x0, dx=dx, n=1)
second = scipy.misc.derivative(f, x0, dx=dx, n=2)

# taylor series approximation near x[-1]
forecast = lambda z_new: np.poly1d([second/2, first, f(z[-1])])(z_new - z[-1])

zs = np.arange(152, 185)
ws = forecast(zs)

# needed to prevent matplotlib from putting the x-axis in scientific notation
x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)  
plt.gca().xaxis.set_major_formatter(x_formatter)

plt.plot(x, y, xs, ys)

# Predictions - plot

brazilPred = pd.read_csv('brazil-prediction.csv', sep = ',')

#Plotting data

brazilPred = brazilPred.sort_values('date', ascending=True)
brazilPred.Timestamp = pd.to_datetime(brazilPred.date,format='%Y-%m-%d') 
brazilPred.index = brazilPred.Timestamp
brazilPred.total_cases.plot(figsize=(15,8), title= 'Daily spreading', fontsize=14)

brazilPred.total_deaths.plot(figsize=(15,8), title= 'Daily death toll', fontsize=14, color='green')

