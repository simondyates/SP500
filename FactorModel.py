import pandas as pd
import numpy as np, numpy.linalg as l

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

# Extract returns
def to_return_array(df):
    df.set_index('Ticker', append=True, inplace=True)
    df_unstack = df['Close'].unstack(level=1)
    df['Return'] = (np.log(df_unstack) - np.log(df_unstack.shift(1))).stack()
    df.dropna(inplace=True)
    df.drop('Close', axis=1, inplace=True)
    df = df.unstack(level=1)
    df.columns = df.columns.droplevel()
    return df

stock_data = to_return_array(stock_data)
ETF_data = to_return_array(ETF_data)

# Normalize both the attributes and targets to mean zero and
# vol equal to SPY (this is maybe a bit quirky)
SPY = ETF_data.loc[:, 'SPY']
SPY_vol = np.std( (SPY - np.mean(SPY)), axis=0, ddof=1)
Y = stock_data.to_numpy()
X = ETF_data.to_numpy()
Y_mean = np.mean(Y, axis=0)
Y = (Y - Y_mean) * SPY_vol / np.std(Y, axis=0)
X = (X - np.mean(X, axis=0)) * SPY_vol / np.std(X, axis=0)

# Main Part of the Code
# Use univariate linear regression to choose most explanatory ETF
# Gram Schmidt orthonormalise to this and repeat until tstat < 2

# Function to do multiple univariate regressions
def lin_reg(X, Y, p):
    n_attrs= X.shape[1] # ETFs
    n_tgts = Y.shape[1] # Stocks
    beta_hat = np.zeros([n_attrs, n_tgts]) # rows for ETFs, cols for Stocks
    t_stats = np.zeros([n_attrs, n_tgts])
    xTx_inv = l.inv(X.T @ X) # square matrix size n_attrs
    beta_hat = xTx_inv @ X.T @ Y # rectangular n_attrs, n_tgts
    Y_hat = X @ beta_hat # N x n_tgts
    MSE = np.diag((Y - Y_hat).T @ (Y - Y_hat)) / (N-p) # MSE is an array length n_tgts
    MSE = np.reshape(MSE, [1, -1]) # make it a 1 row matrix
    xTx_diag = np.reshape(np.diag(xTx_inv), [-1, 1]) # and this a 1 col matrix
    se_beta_hat = np.sqrt(xTx_diag @ MSE) # rectangular n_attrs, n_tgts
    t_stats = abs(beta_hat / se_beta_hat)
    return(beta_hat, t_stats)

def build_R(X, p, n):
# Returns a matrix R which implements the first n steps (from 1)
# of a modified Gram Schmidt process.  i.e. post-multiplying
# the input vector X (with p columns) by R will produce n
# orthonormal basis vectors and then the original vectors,
# residualized to the n GS vectors and normalized.
    R = np.identity(p)
# Do the Gram Schmidt upper left triangle
    for i in range(n):
        for j in range(i):
            E = X @ R
            step_R = np.identity(p)
            step_R[j, i] = - (E[:, i].T @ E[:, j]) / (E[:, j].T @ E[:, j])
            R = R @ step_R
# Now do the residualization of the remaining vectors
    E = X @ R
    for i in range(n, p):
        for j in range(n):
            R[j, i] = - (E[:, i].T @ E[:, j]) / (E[:, j].T @ E[:, j])
# Then normalize with respect to X @ R
    R = R / l.norm((X @ R), axis=0)
    return(R)

# Initialise output variables
N, p = X.shape
Q = np.identity(p)  # will track permutations

# Let's try it once for fun
beta, tstats = lin_reg(X, Y, p)