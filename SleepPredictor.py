
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class SleepPredictor:
    def __init__(self, data_path="msleep.csv"):
        # Veriyi yükleme ve işleme
        self.df = pd.read_csv(data_path)
        self.model = None
        self.x = None
        self.y = None
        self.x_scaled = None

    def process_data(self):
        # Eksik değerlerin doldurulması ve veri işleme
        self.df["sleep_rem"].fillna(self.df["sleep_rem"].mean(), inplace=True)
        self.df["brainwt"].fillna(self.df["brainwt"].mean(), inplace=True)
        self.df['brain_body_ratio'] = self.df['brainwt'] / self.df['bodywt']
        self.df['brain_body_ratio'].fillna(self.df['brain_body_ratio'].mean(), inplace=True)
        self.df["sleep_cycle"].fillna(self.df["sleep_cycle"].mean(), inplace=True)

        # One-hot encoding işlemi
        self.df = pd.get_dummies(self.df, columns=["vore", "conservation", "order"], drop_first=True)

        # Log dönüşümleri
        self.df['log_bodywt'] = np.log1p(self.df['bodywt'])
        self.df['log_brainwt'] = np.log1p(self.df['brainwt'])

        # Hedef ve özellikleri ayırma
        self.y = self.df["sleep_total"]
        self.x = self.df[[
                             "log_bodywt", "log_brainwt", "sleep_rem", "sleep_cycle", "brain_body_ratio"
                         ] + [col for col in self.df.columns if
                              'vore_' in col or 'conservation_' in col or 'order_' in col]]

        # Özellikleri ölçeklendirme
        scaler = StandardScaler()
        self.x_scaled = scaler.fit_transform(self.x)

    def train_model(self):
        # Eğitim ve test verilerini ayırma
        x_train, x_test, y_train, y_test = train_test_split(self.x_scaled, self.y, test_size=0.2, random_state=42)

        # Model tanımlama ve eğitme
        self.model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        self.model.fit(x_train, y_train)

        # Performans metrikleri hesaplama
        y_pred = self.model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'MSE: {mse}, MAE: {mae}, R² Score: {r2}')

        # Tüm veride tahmin yapma
        self.df["predicted_sleep_total"] = self.model.predict(self.x_scaled)
        self.df["error"] = (self.df["sleep_total"] - self.df["predicted_sleep_total"]).abs()

    def get_results(self):
        # Tahmin sonuçlarını döndürme
        return self.df[["name", "sleep_total", "predicted_sleep_total", "error"]]
