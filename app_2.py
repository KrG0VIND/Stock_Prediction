import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('stock price prediction.keras')

st.markdown('<h1 style="color:#804A17;">Stock Market Predictor</h1>', unsafe_allow_html=True)
st.markdown('<hr style="border:1px solid black;">', unsafe_allow_html=True  )

popular_stocks = [
    'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA',
    'META', 'NFLX', 'NVDA', 'JPM', 'V', 'DIS', 'BABA', 'TSM'
]

stock_label = st.markdown('<span style="color:black; font-size: 24px">Select Stock Symbol</span>', unsafe_allow_html=True)
stock = st.text_input('Enter Stock Symbol', st.selectbox('Popular Stocks', popular_stocks))


start = '2014-06-02'
end = '2024-06-02'

data = yf.download(stock, start, end)

st.markdown('<h2 style="color:green;">Stock Data</h2>', unsafe_allow_html=True)
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)


st.markdown('<h2 style="color:green;">Price vs MA50</h2>', unsafe_allow_html=True)
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

#st.subheader('Price vs MA50 vs MA100')
st.markdown('<h2 style="color:green;">Price vs MA50 vs MA100</h2>', unsafe_allow_html=True)
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

#st.subheader('Price vs MA100 vs MA200')
st.markdown('<h2 style="color:green;">Price vs MA100 vs MA200</h2>', unsafe_allow_html=True)
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale
y5=y

#st.subheader('Original Price vs Predicted Price')
st.markdown('<h2 style="color:green;">Original Price vs Predicted Price</h2>', unsafe_allow_html=True)
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')

plt.show()
st.pyplot(fig4)








from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

x_train_rf = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
x_test_rf = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))

rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(x_train_rf, y)

y_pred_rf = rf_model.predict(x_test_rf)

y_pred_rf_rescaled = y_pred_rf * (1 / scaler.scale_[0])
y_test_rescaled = y * (1 / scaler.scale_[0])

mse_rf = mean_squared_error(y_test_rescaled, y_pred_rf_rescaled)
print(f'Random Forest Mean Squared Error: {mse_rf}')

print(predict)
print(predict.shape)
print(predict.min(), predict.max())

predict_rescaled = scaler.inverse_transform(predict)

st.markdown('<h2 style="color:green;">Original Price vs LSTM Predicted Price vs Random Forest Predicted Price</h2>', unsafe_allow_html=True)
fig5 = plt.figure(figsize=(8,6))
plt.plot(y_test_rescaled, 'g', label='Original Price')
plt.plot(predict_rescaled, 'r', label='LSTM Predicted Price')
plt.plot(y_pred_rf_rescaled, 'b', label='Random Forest Predicted Price')
plt.legend()
plt.title('Stock Price Prediction Comparison')
plt.show()
st.pyplot(fig5)





st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://wallpaperaccess.com/full/1393718.jpg");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)