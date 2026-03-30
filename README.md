# 📈 Stock Market Trend Prediction (SMTP)

## 📌 Overview
This project focuses on predicting stock price trends using historical data and machine learning techniques.  
It explores how time-series data can be used to model and forecast stock prices.


---


## 🎯 Objectives
- Analyze historical stock market data
- Build a predictive model for stock prices
- Visualize actual vs predicted trends
- Understand strengths and limitations of ML in financial forecasting

---

## 📊 Dataset
- Source: Yahoo Finance (via yfinance)
- Data includes:
  - Open price
  - Close price
  - High price
  - Low price
  - Volume

---

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- TensorFlow / Keras
- yfinance

---

## 🤖 Model
- Model Used: LSTM (Long Short-Term Memory)
- LSTM is a type of Recurrent Neural Network (RNN) that is well-suited for time-series prediction.
- The model is trained on past stock prices to predict future trends.

---

## 📈 Results
The model predicts stock prices based on historical patterns.

- Visualization compares:
  - Actual stock prices
  - Predicted stock prices

<img width="1005" height="525" alt="SMTP_OriginalVsPredicted" src="https://github.com/user-attachments/assets/a42a5aad-a92d-4b98-a733-208742220d71" />

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/abhinan2058/SMTP.git
   cd SMTP

2. Install dependencies:
    pip install -r requirements.txt
