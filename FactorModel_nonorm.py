# OK there's a bad bug here.  The second GS matrix is losing the orthogonalisation
# to the first vector.  Need to spend some time to bug fix this.


# Takes a set of prices for S&P500 stocks and also for 17 major ETFs
# Runs a Gram Schmidt forward regression to identify the ETFs that
# have the highest beta coefficients.
# NO normalization of variance is done at any stage, so lower volatility ETFs are preferred

# TO-DO List:
# Replace t stat with Driscoll Kraay version

import pandas as pd
import numpy as np
import numpy.linalg as lin
import datetime as dt

# Read in the data
stock_data = pd.read_csv('SP500.csv', usecols=['Date', 'Ticker', 'Adj Close'], index_col='Date')
ETF_data = pd.read_csv('ETF.csv', usecols=['Date', 'Ticker', 'Adj Close'], index_col='Date')
stock_tickers = pd.unique(stock_data['Ticker'])

# Cut them down to six months
stock_data.index = pd.to_datetime(stock_data.index, format='%Y-%m-%d').date
ETF_data.index = pd.to_datetime(ETF_data.index, format='%Y-%m-%d').date
last_dt = dt.date(2020, 2, 20) # Let's look at data before the whole Covid thing hit
first_dt = last_dt - pd.DateOffset(months=6)
stock_data = stock_data.loc[(stock_data.index > first_dt) & (stock_data.index <= last_dt)]
ETF_data = ETF_data.loc[(ETF_data.index > first_dt) & (ETF_data.index <= last_dt)]

# Check that dates are identical
tick0_dates = stock_data[stock_data['Ticker'] == stock_tickers[0]].index
errors = []
for ticker in stock_tickers[1:]:
    if (stock_data[stock_data['Ticker'] == ticker].index != tick0_dates).any():
        errors.append(ticker)
        raise Exception('Dates are inconsistent - please fix your data!')


# Convert closing prices to returns
def to_return_array(df):
    df.set_index('Ticker', append=True, inplace=True)
    df_unstack = df['Adj Close'].unstack(level=1)
    df['Return'] = (np.log(df_unstack) - np.log(df_unstack.shift(1))).stack()
    df.dropna(inplace=True)
    df.drop('Adj Close', axis=1, inplace=True)
    df = df.unstack(level=1)
    df.columns = df.columns.droplevel()
    return df


stock_data = to_return_array(stock_data)
ETF_data = to_return_array(ETF_data)

# Normalize both the attributes and targets to mean zero
Y = stock_data.to_numpy()
X = ETF_data.to_numpy()
Y_mean = np.mean(Y, axis=0)
Y = (Y - Y_mean)
X = (X - np.mean(X, axis=0))

# Exploration: Look at the return correlation matrix for the attributes
norms = lin.norm(X, axis = 0)
corr = X.T @ X / norms / norms.reshape([-1, 1])
corr_df = pd.DataFrame(corr, columns=ETF_data.columns, index=ETF_data.columns)
# print(corr_df)
# HOLY COW!  Covid-19 has blown up my data.  Everything is basically 100% correlated now.

# Main Part of the Code
# Use linear regression to choose most explanatory ETF
# Gram Schmidt orthogonalise to this and repeat until tstat < 2

# Function to do multivariate linear regression
def lin_reg(attribs, targets, ddof):
    xTx_inv = lin.inv(attribs.T @ attribs)  # square matrix size n_attribs
    beta_hat = xTx_inv @ attribs.T @ targets  # rectangular n_attribs, n_targets
    targets_hat = attribs @ beta_hat  # N x n_targets
    MSE = np.diag((targets - targets_hat).T @ (targets - targets_hat)) / (N - ddof)  # MSE is an array length n_targets
    MSE = np.reshape(MSE, [1, -1])  # make it a 1 row matrix
    xTx_diag = np.reshape(np.diag(xTx_inv), [-1, 1])  # and this a 1 col matrix
    se_beta_hat = np.sqrt(xTx_diag @ MSE)  # rectangular n_attribs, n_targets
    t_stats = abs(beta_hat / se_beta_hat)
    return (beta_hat, t_stats)


def build_R(vectors, n_steps):
    # Possible bug:  at step 2 need to remove not just proj on vec 2 but
    # proj on ADJUSTED vec 2 - i.e. you need the result of the previous GS step

    # Returns a matrix R which implements the first n_steps
    # of a modified Gram Schmidt process.  i.e. post-multiplying
    # the input vector vectors by R will produce n_steps
    # orthonormal basis vectors and then the original vectors,
    # residualized to the n_steps GS vectors and normalized.
    n_vecs = X.shape[1]
    R = np.identity(n_vecs)
    # Do the Gram Schmidt upper left triangle
    for i in range(n_steps):
        for j in range(i):
            E = vectors @ R
            step_R = np.identity(n_vecs)
            step_R[j, i] = - (E[:, i].T @ E[:, j]) / (E[:, j].T @ E[:, j])
            R = R @ step_R
    # Now do the residualization of the remaining vectors
    E = vectors @ R
    step_R = np.identity(n_vecs)
    for i in range(n_steps, n_vecs):
        for j in range(n_steps):
            step_R[j, i] = - (E[:, i].T @ E[:, j]) / (E[:, j].T @ E[:, j])
    R = R @ step_R
    return (R)


# Initialise output variables
N, p = X.shape
Q = np.identity(p)  # will track permutations of ETFs
R = np.identity(p)  # will track Gram Schmidt process on ETFs

# Cycle through the attributes until average t stat drops below 1 (arbitrary!)
for step in range(p):
    beta_hat, t_stats = lin_reg(X @ Q @ R, Y, step)
    beta_norm = np.sqrt(np.diag(beta_hat @ beta_hat.T) / p)
    t_avg = np.mean(t_stats, axis=1)
    pos = np.argmax(beta_norm[step:]) + step  # Since we don't normalize, this will benefit low vol ETFs in selection
    if t_avg[pos] < 1 and step>0:
        print('Break at step {0}'.format(step))
        break
    Q[:, [step, pos]] = Q[:, [pos, step]]
    R = build_R(X @ Q, step + 1)

# Zero out unused attributes
for i in range(step, p):
    Q[i, i] = 0
    for j in range(p):
        R[j, i] = 0

# Print order of attributes
index = (range(p) @ Q).astype('int')
for i in range(step):
    print(ETF_data.columns[index[i]])

# Demonstrate that X are orthogonal up to step
for i in range(step):
    for j in range(step):
        e_ij = (X @ Q @ R)[:, i].T @ (X @ Q @ R)[:, j]
        print('{0:.2f}'.format(e_ij), end='\t')
    print('')
