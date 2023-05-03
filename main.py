"""
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path

#Select database
data_file_path= 'input/household.csv'
dataset=pd.read_csv(data_file_path,low_memory=False )

#Checking database
#print(dataset.info())
#tmp_str = "Feature(attribute)     DataType"; print(tmp_str+"\n"+"-"*len(tmp_str)); print(dataset.dtypes)
#print("Shape of the data: {} --> n_rows = {}, n_cols = {}".format(dataset.shape, dataset.shape[0],dataset.shape[1]))
#print (dataset.head(10))
#print (dataset.tail(10))


dataset = dataset[:-1]
#print (dataset.tail(10))
dataset.columns = [col.replace(' [kW]', '') for col in dataset.columns]
dataset['sum_Furnace'] = dataset[['Furnace 1','Furnace 2']].sum(axis=1)
dataset['avg_Kitchen'] = dataset[['Kitchen 12','Kitchen 14','Kitchen 38']].mean(axis=1)
dataset = dataset.drop(['Kitchen 12','Kitchen 14','Kitchen 38'], axis=1)
dataset = dataset.drop(['Furnace 1','Furnace 2'], axis=1)
dataset = dataset.drop(columns=['House overall',  'Wine cellar','summary', 'icon', 'use', 'gen','precipProbability', 'sum_Furnace', 'avg_Kitchen', 'windBearing', 'precipIntensity', 'dewPoint', 'apparentTemperature', 'pressure', 'windSpeed', 'cloudCover', 'humidity', 'visibility', 'Solar', 'Garage door', 'Barn', 'Well'
 ])

dataset = dataset.drop(['Dishwasher'], axis=1)
dataset = dataset.drop(['Home office'], axis=1)
dataset = dataset.drop(['Fridge'], axis=1)
dataset = dataset.drop(['Microwave'], axis=1)
dataset = dataset.drop(['Living room'], axis=1)
#dataset = dataset.drop(['temperature'], axis=1)



#print(dataset['time'].head())
start_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(dataset['time'].iloc[0])))
time_index = pd.date_range(start_time, periods=len(dataset),  freq='min')
time_index = pd.DatetimeIndex(time_index)
dataset = dataset.set_index(time_index)
dataset = dataset.drop(['time'], axis=1)
#print(dataset.shape)
#dataset.info()
#print(dataset.isna().sum())

#print(dataset.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))
#print(dataset.cloudCover)
#dataset['cloudCover'].replace(['cloudCover'], method='bfill', inplace=True)
#dataset['cloudCover'] = dataset['cloudCover'].astype('float')
#print(dataset.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))


dataset = dataset.resample('H', group_keys=True).mean()


filepath = Path('output/temperature.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(filepath)



#print (dataset.head(pd.set_option('display.max_columns', None)))


#print("Shape of hourly dataset: {} --> n_rows = {}, n_cols = {}".format(dataset.shape, dataset.shape[0],dataset.shape[1]))


''''
#dataset['temperature'].plot(figsize=(16,5))
#dataset['temperature'].resample(rule='D').mean().plot(figsize=(16,5))
#plt.rcParams["figure.figsize"] = (25,5)
#plt.show()

#dataset['Microwave'].resample("h").mean().iloc[:24].plot()
#dataset.groupby(dataset.index.hour).mean()['Microwave'].plot(xticks=np.arange(24)).set(xlabel='Daily Hours', ylabel='Microwave Usage (kW)')
#plt.show()


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(25, 5))
    plt.title("Moving average with window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

        # find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


n_samples = 24 * 30  # 1 month
cols = ['use']
#plotMovingAverage(dataset[cols][:n_samples], window=6)  # A window of 6 hours
#plotMovingAverage(dataset[cols][:n_samples], window=12) # A window of 12 hours
#plotMovingAverage(dataset[cols][:n_samples], window=24, plot_intervals=True, plot_anomalies=True)


def exponential_smoothing(series, alpha):
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result


def plotExponentialSmoothing(series, alphas):
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(25, 5))
        for alpha in alphas:
            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
        plt.plot(series.values, "c", label="Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);
        plt.show()

n_samples = 24*30 # 1 month
col = 'use'
#plotExponentialSmoothing(dataset[col][:n_samples], [0.3, 0.05])

def forcast_ts(data, tt_ratio):
    X = data.values
    size = int(len(X) * tt_ratio)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('progress:%',round(100*(t/len(test))),'\t predicted=%f, expected=%f' % (yhat, obs), end="\r")
    error = mean_squared_error(test, predictions)
    print('\n Test MSE: %.3f' % error)

    plt.rcParams["figure.figsize"] = (25,10)
    preds = np.append(train, predictions)
    plt.plot(list(preds), color='green', linewidth=3, label="Predicted Data")
    plt.plot(list(data), color='blue', linewidth=2, label="Original Data")
    plt.axvline(x=int(len(data)*tt_ratio)-1, linewidth=5, color='red')
    plt.legend()
    plt.show()

col = 'sum_Furnace'
data = dataset[col].resample('d').mean()
data.shape
tt_ratio = 0.70 # Train to Test ratio
forcast_ts(data, tt_ratio)
 '''


















