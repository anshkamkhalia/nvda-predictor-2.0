import pandas as pd
import yfinance as yf
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from mangum import Mangum
# os.environ["TF_METAL_USE_GPU"] = "0"  # completely disable Metal
ticker = "NVDA"

from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Dense, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam

df = yf.download(ticker, start="1800-01-01", end="2025-09-12")

# Flatten MultiIndex if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

# Reset index so 'Date' becomes a column
df = df.reset_index()

# Drop volume as it shows future data
df.drop(['Volume', 'High', 'Low'], axis=1, inplace=True) 
df["Pct_Change"] = df["Close"].pct_change() # Create target column
df = df.dropna() # Drop null values

# Get current quarter
df["Quarter"] = df["Date"].dt.quarter

# Extract month, day, year from Date
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Year'] = df['Date'].dt.year

# One-hot encode
quarter_dummies = pd.get_dummies(df["Quarter"], prefix="Quarter")
df = pd.concat([df, quarter_dummies.astype(int)], axis=1)

df.drop("Quarter", axis=1, inplace=True) # Drop quarters

X_train, X_test = train_test_split(df, test_size=0.1, shuffle=False) # Split data

y_train = X_train.pop("Pct_Change") # Get targets
y_test = X_test.pop("Pct_Change")

# 1 if stock increased today, 0 if not
X_train['Increased_From_Yesterday'] = (y_train > 0).astype(int)
X_test['Increased_From_Yesterday'] = (y_test > 0).astype(int)

# Add normalized dates
X_train['Month_sin'] = np.sin(2 * np.pi * X_train['Month'] / 12)
X_train['Month_cos'] = np.cos(2 * np.pi * X_train['Month'] / 12)

# Day: 1–31
X_train['Day_sin'] = np.sin(2 * np.pi * X_train['Day'] / 31)
X_train['Day_cos'] = np.cos(2 * np.pi * X_train['Day'] / 31)

# Same for test set
X_test['Month_sin'] = np.sin(2 * np.pi * X_test['Month'] / 12)
X_test['Month_cos'] = np.cos(2 * np.pi * X_test['Month'] / 12)
X_test['Day_sin'] = np.sin(2 * np.pi * X_test['Day'] / 31)
X_test['Day_cos'] = np.cos(2 * np.pi * X_test['Day'] / 31)

# Add rolling averages
X_train['MA_3'] = X_train['Close'].rolling(3).mean()
X_train['MA_5'] = X_train['Close'].rolling(5).mean()
X_train['MA_10'] = X_train['Close'].rolling(10).mean()

X_test['MA_3'] = X_test['Close'].rolling(3).mean()
X_test['MA_5'] = X_test['Close'].rolling(5).mean()
X_test['MA_10'] = X_test['Close'].rolling(10).mean()

X_train.dropna(inplace=True)
y_train = y_train[-len(X_train):]  # Align y_train with X_train
X_test.dropna(inplace=True)
y_test = y_test[-len(X_test):]

# Drop month and day cols
X_train.drop(['Month', 'Day', 'Year',], axis=1, inplace=True)
X_train.dropna(inplace=True)

# Keep all numeric columns + binary features
features_to_use = [
    'Open', 'MA_3', 'MA_5', 'MA_10',
    'Increased_From_Yesterday',
    'Quarter_1', 'Quarter_2', 'Quarter_3', 'Quarter_4',
    'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
]


# Scale values between [0-1]

# Create scalers
input_scaler = MinMaxScaler()   
output_scaler = MinMaxScaler()

# y_train is a Series → reshape to 2D
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# Fit scalers
input_scaler.fit(X_train[['Open', 'Close', 'MA_3', 'MA_5', 'MA_10']])
output_scaler.fit(y_train)

numeric_cols = ['Open', 'MA_3', 'MA_5', 'MA_10']

# Transform
X_train[numeric_cols] = input_scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = input_scaler.transform(X_test[numeric_cols])

X_train = X_train[features_to_use]
X_test = X_test[features_to_use]

y_train = output_scaler.transform(y_train)
y_test = output_scaler.transform(y_test)

# Convert to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Create model

@register_keras_serializable(package='custom_model')
class NVDAStockPredictor(Model):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.lstm1 = LSTM(128, return_sequences=True)
        self.lstm2 = LSTM(64,)
        
        self.batch_norm = BatchNormalization()
        
        self.dense64 = Dense(64, activation='relu')
        # self.dense256 = Dense(256, activation='relu')



        self.pct_change = Dense(1)

    def call(self, x):
        x = self.lstm1(x)
        # x = self.lstm2(x)
        # x = self.lstm3(x)

        x = self.batch_norm(x)

        x = self.dense64(x)
        # x = self.dense256(x)

        x = self.pct_change(x)

        return x

model = NVDAStockPredictor()

# Create sequences for LSTM
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_train, y_train = create_sequences(X_train, y_train, time_steps=20)
X_test, y_test = create_sequences(X_test, y_test, time_steps=20)

# Step 2: Ensure float32
X_train = np.ascontiguousarray(X_train, dtype=np.float32)
X_test  = np.ascontiguousarray(X_test, dtype=np.float32)
y_train = np.ascontiguousarray(y_train, dtype=np.float32)
y_test  = np.ascontiguousarray(y_test, dtype=np.float32)

time_steps = X_train.shape[1]
features = X_train.shape[2]  # this will now be 16

# NO model.build() call 

# Create early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=45,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',      # watch validation loss
    factor=0.75,              # multiply lr by amount
    patience=5,              # wait 5 epochs with no improvement
    verbose=1,               # show messages
    mode='min',              # 'min' because lower val_loss is better
    min_lr=1e-6              # don't go below this learning rate
)

# Created my own callback for sounds 
class MyCallback(Callback):

    def __init__(self):
        super().__init__()
        self.best_val_loss = float("inf")
    
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss') is not None:
            if logs['val_loss'] < self.best_val_loss:
                self.best_val_loss = logs['val_loss']
                os.system("afplay /System/Library/Sounds/Tink.aiff")
            else:
                os.system("afplay /System/Library/Sounds/Sosumi.aiff")


    def on_train_end(self, logs=None):
        os.system("afplay /System/Library/Sounds/Submarine.aiff")

# Create model checkpointing to use the best model
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min" 
)

# Convert to float to avoid metal M4 GPU errors
X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test  = y_test.astype(np.float32)


time_steps = X_train.shape[1]
features = X_train.shape[2]

# Compile model with mse, adam and maa
model.compile(
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0),
    loss='mse',          
    metrics=['mae'] 
)

# # TRAIN MODEL
# history = model.fit(
#     X_train,
#     y_train,
#     validation_data=[X_test, y_test],
#     epochs=100,
#     batch_size=32,
#     callbacks=[checkpoint, early_stop, MyCallback(), reduce_lr],
#     shuffle=False
# )

nvda_model = load_model('model/best_model.keras')
nvda_model.summary()

import datetime

# Get today's date dynamically
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

# Function to fetch and preprocess stock data
def prepare_data(ticker):
    data = yf.download(ticker, start=yesterday - datetime.timedelta(days=40), end=today + datetime.timedelta(days=1))
    data = data.reset_index()

    # Flatten MultiIndex if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    # Calculate rolling averages
    data['MA_3'] = data['Close'].rolling(3).mean()
    data['MA_5'] = data['Close'].rolling(5).mean()
    data['MA_10'] = data['Close'].rolling(10).mean()

    # Compute percent change
    data['Pct_Change'] = data['Close'].pct_change()

    # Extract date features
    data['Quarter'] = data['Date'].dt.quarter
    quarter_dummies = pd.get_dummies(data['Quarter'], prefix="Quarter")
    for q in ['Quarter_1', 'Quarter_2', 'Quarter_3', 'Quarter_4']:
        if q not in quarter_dummies.columns:
            quarter_dummies[q] = 0
    data = pd.concat([data, quarter_dummies], axis=1)

    data['Month_sin'] = np.sin(2 * np.pi * data['Date'].dt.month / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Date'].dt.month / 12)
    data['Day_sin'] = np.sin(2 * np.pi * data['Date'].dt.day / 31)
    data['Day_cos'] = np.cos(2 * np.pi * data['Date'].dt.day / 31)

    # Increased from yesterday
    data['Increased_From_Yesterday'] = (data['Pct_Change'] > 0).astype(int)

    # Keep features in correct order
    seq_data = data[features_to_use].iloc[-10:]  # last 10 days for LSTM
    seq_data[numeric_cols] = input_scaler.transform(seq_data[numeric_cols])

    # Final sequence
    seq = np.expand_dims(seq_data.values.astype(np.float32), axis=0)
    return seq

# Prepare data
nvda_seq = prepare_data("NVDA")

# Predict NVDA
nvda_final = nvda_model.predict(nvda_seq)[0][0][0]
print(nvda_final)
# nvda_final = output_scaler.inverse_transform(nvda_final)
# print(nvda_final)

# WEB

from flask import Flask, render_template, request
import yagmail

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    # Send email

    if request.method == "POST":

        # --- CONFIG ---
        EMAIL_ADDRESS = "wannabananaboat@gmail.com"
        EMAIL_PASSWORD = "fcfwkyabwldfqcqj"
        TO_EMAIL = "9172085437@vtext.com"
        SUBJECT = "NVDA Percent Change Predictions"
        BODY = f"NVDA: {nvda_final:.2f}%\n\nWe love you soo much!!!! - From Ansh and Arjun"

        try:
            yag = yagmail.SMTP(EMAIL_ADDRESS, EMAIL_PASSWORD)
            yag.send(
                to=TO_EMAIL,
                subject=SUBJECT,
                contents=BODY
            )
            print("✅ Email sent successfully with yagmail!")
        except Exception as e:
            print("❌ Error:", e)


    return render_template('home_page.html')

handler = Mangum(app)