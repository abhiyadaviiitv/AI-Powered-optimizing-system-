# ============================================================================
# FILE: src/ml_models/cpu_burst_predictor.py
# PURPOSE: Predict next CPU burst duration (without neural networks)
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


class BurstPredictorTrainer:
    """Train CPU burst predictor using scikit-learn regressor"""
    
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def prepare_sequences(self, df):
        """Prepare rolling sequences from process CPU usage"""
        sequences = []
        targets = []
        
        for pid, group in df.groupby('pid'):
            cpu_values = group['cpu_percent'].values
            
            if len(cpu_values) < self.sequence_length + 1:
                continue
            
            for i in range(len(cpu_values) - self.sequence_length):
                seq = cpu_values[i:i + self.sequence_length]
                target = cpu_values[i + self.sequence_length]
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def train(self, df, test_size=0.2):
        """Train scikit-learn model"""
        print("🧠 Preparing data for burst prediction (no deep learning)...")
        
        X, y = self.prepare_sequences(df)
        if len(X) == 0:
            print("❌ Not enough data for training")
            return
        
        print(f"📊 Samples: {len(X)}, Features per sample: {X.shape[1]}")
        
        # Normalize data
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        print("🚀 Training Gradient Boosting Regressor...")
        self.model.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print("\n✅ Training Completed!")
        print(f"📉 MAE:  {mae:.2f}")
        print(f"📈 RMSE: {rmse:.2f}")
        
    def predict_next_burst(self, recent_cpu_series):
        """Predict next CPU burst duration"""
        if len(recent_cpu_series) < self.sequence_length:
            raise ValueError("Not enough data points for prediction")
        
        X_input = np.array(recent_cpu_series[-self.sequence_length:]).reshape(1, -1)
        X_scaled = self.scaler.transform(X_input)
        return self.model.predict(X_scaled)[0]
    
    def save_model(self, path='models/burst_predictor.pkl'):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'sequence_length': self.sequence_length
        }, path)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path='models/burst_predictor.pkl'):
        """Load trained model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.sequence_length = data['sequence_length']
        print(f"✅ Model loaded from {path}")
