import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import streamlit as st
import plotly.graph_objects as go
from plotly.offline import iplot, init_notebook_mode
import matplotlib.pyplot as plt
import plotly.io as pio
import seaborn as sn
import riskfolio as rp
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import HRPOpt
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import objective_functions

warnings.filterwarnings("ignore")
#pd.options.display.float_format = '{:.4%}'.format



st.title("Criando um Portifólio de Commodities") 

st.header("Selecione Algumas Commodities Para Criarmos o Portifólio:")
# Date range
start='2020-01-01'
end='2022-9-25'


# Tickers of commodities
assets = ['GC=F','MGC=F','SI=F','SIL=F','PL=F','HG=F','PA=F','CL=F','HO=F','NG=F','BZ=F','ZC=F','ZO=F','KE=F','ZR=F','ZM=F','ZL=F','ZS=F','GF=F','HE=F','LE=F','CC=F','KC=F','CT=F','LBS=F','OJ=F','SB=F'
]

# Downloading data
data = yf.download(assets, start = start, end = end)
#data = yf.download(assets, period="max")
#data = data["Adj Close"].dropna(how="all")
data = data.loc[:,('Adj Close', slice(None))]
data.columns = assets


data.rename(columns={'GC=F':'Gold', 
                           'MGC=F':'Micro Gold Futures',
                           'SI=F':'Silver',
                           'SIL=F':'Micro Silver',
                           'PL=F':'Platinum',
                           'HG=F':'Copper',
                           'PA=F':'Palladium',
                           'CL=F':'Crude Oil',
                           'HO=F':'Heating Oil',
                           'NG=F':'Natural Gas',
                           'BZ=F':'Brent Crude Oil',
                           'ZC=F':'Corn Futures',
                           'ZO=F':'Oat Futures',
                           'KE=F':'KC HRW Wheat Futures',
                           'ZR=F':'Rough Rice Futures',
                           'ZM=F':'Soybean Meal Futures',
                           'ZL=F':'Soybean Oil Futures',
                           'ZS=F':'Soybean Futures',
                           'GF=F':'Feeder Cattle Futures',
                           'HE=F':'Lean Hogs Futures',
                           'LE=F':'Live Cattle Futures',
                           'CC=F':'Cocoa',
                           'KC=F':'Coffee',
                           'CT=F':'Cotton',
                           'LBS=F':'Lumber',
                           'OJ=F':'Orange Juice',
                           'SB=F':'Sugar'}, inplace=True)

opcoes = st.multiselect('Selecione as Commodities Desejadas',
     ['Micro Gold Futures','Silver','Micro Silver','Platinum','Copper','Palladium','Crude Oil','Heating Oil','Natural Gas',
      'Brent Crude Oil','Corn Futures','Oat Futures','KC HRW Wheat Futures','Rough Rice Futures',
      'Soybean Meal Futures','Soybean Oil Futures','Soybean Futures','Feeder Cattle Futures','Lean Hogs Futures','Live Cattle Futures',
      'Cocoa','Coffee','Cotton','Lumber','Orange Juice','Sugar'])

st.write('Você selecionou:', opcoes)

df = pd.DataFrame(data, columns = opcoes)


#st.dataframe(df)

st.header("Gráfico das Commodities Escolhidas - Preços do Fechamento Ajustado ao Longo do Periodo Max")

chart_data = pd.DataFrame(df,
     columns=opcoes)

st.line_chart(chart_data)

aceito = st.checkbox('Eu confirmo as Commodities selecionadas')
if aceito:
	Y = df.pct_change().dropna()
	mu = expected_returns.mean_historical_return(df,frequency=252)
	S = risk_models.sample_cov(df)
	ef = EfficientFrontier(mu,S)
	ef.add_objective(objective_functions.L2_reg, gamma=0.1)
	weights = ef.max_sharpe()
	cleaned_weights = ef.clean_weights()
	metrica = ef.portfolio_performance(verbose=False,risk_free_rate=0.1365)
	st.write(str)
	st.write('Seguem as Commodities e Seus Pesos')
	cleaned_weights
	st.write('Expected annual return:',metrica[0]*100,'%')
	st.write('Annual volatility:',metrica[1]*100,'%')
	st.write('Sharpe Ratio:', metrica[2])
	st.header("Agora Vamos Fazer as Alocações Para as Commodities Selecionadas de acordo Com o Investimento")
	totalinvesti= st.number_input('Qual o valor de investimento?')
	if totalinvesti:
		latest_prices = get_latest_prices(df)
		weights = cleaned_weights
		da = DiscreteAllocation(weights,latest_prices,totalinvesti)
		allocation, leftover = da.lp_portfolio()
		st.write('Alocação Discreta Para Cada Commodities selecionada:', allocation)
		st.write('Resto:', leftover) 
	else:
		st.write('Por favor Digite um Valor')





#else:
	#st.write('Por favor escolha as Commodities')

st.header('Utilizando o Metodo de Otimização: Paridade de Risco Hierárquico (HRP)')
criar = st.checkbox('Criar Portifólio HRP?')
if criar:
	rets = expected_returns.returns_from_prices(df)
	hrp = HRPOpt(rets)
	hrp.optimize()
	metrica2 = hrp.portfolio_performance(verbose=False,risk_free_rate=0.1365)
	st.write('Expected annual return:',metrica2[0] *100,'%')
	st.write('Annual volatility:',metrica2[1]*100,'%')
	st.write('Sharpe Ratio:', metrica2[2])
	weights2 = hrp.clean_weights()
	st.write('Seguem as Commodities e seus pesos - otimização HRP')
	st.write(weights2)
	latest_prices = get_latest_prices(df)
	da = DiscreteAllocation(weights2,latest_prices,totalinvesti)
	allocation, leftover = da.lp_portfolio()
	st.write('Alocação discreta:', allocation)
	st.write('Resto:', leftover)

	
	


	