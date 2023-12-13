# Mean Variance Optimization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

comps = ["andritz", "ATS", "Bawag", "CA_Immo", "DOCO", "Erste_Group", "Immofinanz", "Lenzing", "MayrMelnhof", "OMV", "OPost", "Raiffeisen", "SImmo", "Schoeller-Bleckmann", "TelekomAT", "Uniqa", "VIG", "Verbund", "voest","Wienerberger"]
tickers = ["ANDR","ATS", "BG", "CAI", "DOC", "EBS", "IIA", "LNZ", "MMK", "OMV", "POST", "RBI", "SPI", "SBO", "TKA", "UQA", "VIG", "VER", "VOE", "WIE"]
atx_comps = pd.DataFrame()

for i in range(0,19):
    comp = pd.read_csv("atx/"+str(comps[i]) + ".csv", delimiter = ";", decimal = ",",
        names = ["Date", str(tickers[i])], header = 1, index_col = "Date", parse_dates=True)

    atx_comps = pd.merge(atx_comps, comp, left_index = True, right_index=True, how = "outer") 

atx_comps = atx_comps["2010-10-20":"2020-10-20"]

atx =  pd.read_csv("indices/atx_tr.csv", delimiter = ";", decimal = ".", names = ["Date", "ATX_TR"], usecols = [0,1], header = 0, index_col = "Date", parse_dates=True)
sp500 = pd.read_csv("indices/^GSPC.csv", names = ["Date", "SP500"], parse_dates=True, usecols = [0,1], index_col = "Date", header = 1)
dax = pd.read_csv("indices/^GDAXI.csv", names = ["Date", "DAX"], parse_dates=True, usecols = [0,1], index_col = "Date", header = 1)
nasdaq = pd.read_csv("indices/^IXIC.csv", names = ["Date", "NASDAQ"], parse_dates=True, usecols = [0,1], index_col = "Date", header = 1)

indices = pd.concat([atx, sp500, dax, nasdaq], join = "outer", axis = 1)['2010-10-19':]
indices.head()

indices_rets = indices.pct_change()
indices_cumrets = indices_rets.add(1).cumprod().sub(1)*100

fig = px.line(indices_cumrets, x=indices_cumrets.index, y=indices_cumrets.columns, title='Cumulative Returns of Indices (2010-2020)')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Cumulative Return in %')
fig.show()

atx_comps_returns = atx_comps.pct_change()
atx_comps_rets_cumprod = atx_comps_returns.add(1).cumprod().sub(1)*100

fig = px.line(atx_comps_rets_cumprod, x=atx_comps_rets_cumprod.index, y=atx_comps_rets_cumprod.columns, title='Cumulative Returns of ATX Stocks (2010-2020)')

fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Cumulative Return in %')

fig.show()

from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import exp_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier
from pypfopt.plotting import plot_weights
from pypfopt.cla import CLA

train = atx_comps_returns[:"2018-10-21"]
test = atx_comps_returns["2018-10-21":]

mu = expected_returns.ema_historical_return(train, returns_data = True, span = 500)
Sigma = risk_models.exp_cov(train, returns_data = True, span = 180)


ret_ef = np.arange(0, 0.879823, 0.01)
vol_ef = []
for i in np.arange(0, 0.879823, 0.01):
    ef = EfficientFrontier(mu, Sigma)
    ef.efficient_return(i)
    vol_ef.append(ef.portfolio_performance()[1])

ef = EfficientFrontier(mu, Sigma)
ef.min_volatility()
min_vol_ret = ef.portfolio_performance()[0]
min_vol_vol = ef.portfolio_performance()[1]

ef.max_sharpe(risk_free_rate=0.009)
max_sharpe_ret = ef.portfolio_performance()[0]
max_sharpe_vol = ef.portfolio_performance()[1]



sns.set()

fig, ax = plt.subplots(figsize = [15,10])

sns.lineplot(x = vol_ef, y = ret_ef, label = "Efficient Frontier", ax = ax)
sns.scatterplot(x = [min_vol_vol], y = [min_vol_ret], ax = ax, label = "Minimum Variance Portfolio", color = "purple", s = 100)
sns.scatterplot(x = [max_sharpe_vol], y = [max_sharpe_ret], ax = ax, label = "Maximum Sharpe Portfolio", color = "green", s = 100)
sns.lineplot(x = [0, max_sharpe_vol, 1], y = [0.009, max_sharpe_ret, 3.096], label = "Capital Market Line", ax = ax, color = "r")

ax.set(xlim = [0, 0.4])
ax.set(ylim = [0, 1])
ax.set_xlabel("Volatility")
ax.set_ylabel("Mean Return")
plt.legend(fontsize='large')
plt.title("Efficient Frontier", fontsize = '20')

ax.figure.savefig("EffFront_big.png", dpi = 300)


ef = EfficientFrontier(mu, Sigma)
raw_weights_minvar_exp = ef.min_volatility()

plot_weights(raw_weights_minvar_exp)
ef.portfolio_performance(verbose = True, risk_free_rate = 0.009)

ef = EfficientFrontier(mu, Sigma)
raw_weights_maxsharpe_exp = ef.max_sharpe(risk_free_rate=0.009)

plot_weights(raw_weights_maxsharpe_exp)
ef.portfolio_performance(verbose = True, risk_free_rate = 0.009)

weights_minvar_exp = list(raw_weights_minvar_exp.values())
weights_maxsharpe_exp = list(raw_weights_maxsharpe_exp.values())

ret_1 = test.dot(weights_minvar_exp).add(1).cumprod().subtract(1).multiply(100)
ret_2 = test.dot(weights_maxsharpe_exp).add(1).cumprod().subtract(1).multiply(100)

ind_ret = indices["2018-10-23":]["ATX_TR"].pct_change().add(1).cumprod().subtract(1).multiply(100)

back = pd.DataFrame({"MinVar":ret_1, "MaxSharpe":ret_2})
back = pd.concat([back, ind_ret],  join = "outer", axis = 1)
back.drop(back.tail(1).index,inplace=True)

back.interpolate(method = "linear", inplace = True)

fig = px.line(back, x = back.index, y = back.columns, title = "Portfolio Performance (2018-2020)")
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Cumulative Return in %')

fig.show()