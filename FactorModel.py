import pandas as pd
import numpy as np

# Read in the data
data_dir = '~/Dropbox/DataSci/PycharmProjects/SP500/'
stock_data = pd.read_csv(data_dir + 'SP500.csv', usecols=['Date', 'Ticker', 'Close'], index_col='Date')
ETF_data = pd.read_csv(data_dir + 'ETF.csv', usecols=['Date', 'Ticker', 'Close'], index_col='Date')
stock_tickers = pd.unique(stock_data['Ticker'])

# Cut them down to six months
stock_data.index = pd.to_datetime(stock_data.index, format='%Y-%m-%d').date
ETF_data.index = pd.to_datetime(ETF_data.index, format='%Y-%m-%d').date
last_dt = ETF_data.index.max()
first_dt = last_dt - pd.DateOffset(months=6)
stock_data = stock_data.loc[(stock_data.index>first_dt) & (stock_data.index<=last_dt)]
ETF_data = ETF_data.loc[(ETF_data.index>first_dt) & (ETF_data.index<=last_dt)]

# Check that dates are identical
tick0_dates = stock_data[stock_data['Ticker']==stock_tickers[0]].index
errors = []
for ticker in stock_tickers[1:]:
    if (stock_data[stock_data['Ticker']==ticker].index != tick0_dates).any():
        errors.append(ticker)
        raise Exception('Dates are inconsistent - please fix your data!')

# Extract returns and normalize by mean and std
def to_return_array(df):
    df.set_index('Ticker', append=True, inplace=True)
    df_unstack = df['Close'].unstack(level=1)
    df['Return'] = (np.log(df_unstack) - np.log(df_unstack.shift(1))).stack()
    df.dropna(inplace=True)
    df.drop('Close', axis=1, inplace=True)
    df = df.unstack(level=1)
    df.columns = df.columns.droplevel()
    df = (df - np.mean(df, axis=0)) / np.std(df, axis=0)
# It might be a good idea to think about whether it makes sense to
# scale the standard deviations like this
    return df

stock_data = to_return_array(stock_data)
ETF_data = to_return_array(ETF_data)

# Do the Gram Schmidt Regression Process