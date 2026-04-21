import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

engine = create_engine('postgresql://postgres:yeni_sifre@localhost:5432/travelsafe')

print("LSTM için veriler çekiliyor...")
query = 'SELECT "Date", "Primary Type", "Latitude", "Longitude", "harm_weight" FROM crime ORDER BY "Date" '
df = pd.read_sql(query, engine)

df['Month'] = pd.to_datetime(df['Date']).dt.month
le = LabelEncoder()
y = le.fit_transform(df['Primary Type'])
X = df[['Month', 'Latitude', 'Longitude', 'harm_weight']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LSTM 3D Input
X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Zaman serisi olduğu için sondaki verileri test olarak ayırıyoruz (shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, shuffle=False)

print("LSTM Modeli Oluşturuluyor...")
model = Sequential([
    LSTM(64, input_shape=(1, X.shape[1]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=512, validation_split=0.1)