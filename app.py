# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 04:38:48 2023

@author: Admin
"""


import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt



st.set_page_config(page_title="Forecasting stock price", layout="wide")


st.title('Forecasted Prices')

 # Load the time series data
data = pd.read_csv('Modeldata.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)
   
df=np.log(data)

# Define the parameters for the Holt-Winters model
alpha = 0.4 # smoothing parameter for level
beta = 0.45   # smoothing parameter for trend
gamma = 0.15  # smoothing parameter for seasonality
season_length =260  # length of seasonal cycle

# Fit the Holt-Winters model to the data
model = ExponentialSmoothing(df, seasonal_periods=season_length, trend='add', seasonal='add')
fitted_model = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)


#model = pickle.load(open('model.pkl','rb'))
st.sidebar.title('Apple stock Forecasting')

st.sidebar.header('Select the number of days to Forecast')
days=st.sidebar.number_input("",1,30)
forecast = fitted_model.forecast(days)
pred = pd.DataFrame(forecast, columns=['Adj_close_imp'])
if st.sidebar.button("Forecast"):
    st.dataframe(np.exp(pred))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Chart Zoomed Out')
        fig1, ax1 = plt.subplots()
        ax1.plot(df.index, np.exp(df.values))
        ax1.plot(forecast.index, np.exp(forecast.values))
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Adj_close_imp')
        st.pyplot(fig1)
        
    with col2:
        st.subheader('Chart Zoomed In')
        last=df['2019':]
        last1=np.exp(last)
        fig2, ax2 = plt.subplots()
        ax2.plot(last.index, last1.values)
        ax2.plot(forecast.index, np.exp(forecast.values))
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Adj_close_imp')
        st.pyplot(fig2)
        
        
    col3, col4 = st.columns(2)
    with col3:
        st.subheader('Percentage change')
        fig3, ax3 = plt.subplots()
        pct_change1 = np.exp(forecast).pct_change()
        ax3.plot(pct_change1.index,pct_change1.values,color="orange")
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Adj_close_imp')
        st.pyplot(fig3)
        total_pct_change = (abs(np.exp(forecast).iloc[1]-np.exp(forecast).iloc[-1])/(np.exp(forecast).iloc[1]))*100
        st.subheader('Expected returns in percentage')
        st.write(total_pct_change)
            
    with col4:
        st.subheader('Cumulative percentage change')
        last=df['2019':]
        last1=np.exp(last)
        fig4, ax4 = plt.subplots()
        cumulative_pct_change=np.exp(forecast).pct_change().add(1).cumprod().sub(1)
        ax4.plot(cumulative_pct_change.index,cumulative_pct_change.values,color="orange")
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Adj_close_imp')
        st.pyplot(fig4)
        
        
        
        
        

       
        




