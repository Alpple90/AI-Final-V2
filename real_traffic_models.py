# real_traffic_models.py
"""
REAL LSTM, GRU, and XGBoost implementations for traffic prediction
Uses .keras format for better compatibility
"""

import numpy as np
import pandas as pd
from datetime import timedelta
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# XGBoost
import xgboost as xgb

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class RealTrafficPredictor:
    def __init__(self, seq_length=12, batch_size=32, learning_rate=0.001):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.models = {}
        self.scaler = StandardScaler()
        self.training_history = {}
        
    def load_data_from_excel(self, excel_file='Scats Data October 2006.xls'):
        """Load traffic data from Excel file"""
        print("\n" + "=" * 60)
        print("LOADING TRAFFIC DATA FROM EXCEL")
        print("=" * 60)
        
        df = pd.read_excel(excel_file, sheet_name='Data', header=1)
        print(f"Loaded {len(df)} rows of traffic data")
        
        volume_cols = [col for col in df.columns if str(col).startswith('V') and str(col)[1:].isdigit()]
        volume_cols = sorted(volume_cols, key=lambda x: int(x[1:]))
        print(f"Found {len(volume_cols)} volume columns (15-min intervals)")
        
        all_volumes = []
        timestamps = []
        
        for idx, row in df.iterrows():
            scats_num = row.get('SCATS Number')
            if pd.isna(scats_num):
                continue
            
            volumes = []
            for col in volume_cols:
                vol = row.get(col, 0)
                if pd.isna(vol):
                    vol = 0
                volumes.append(int(vol))
            
            date_val = row.get('Date', None)
            if pd.notna(date_val):
                all_volumes.extend(volumes)
                base_time = pd.to_datetime(date_val)
                for i in range(len(volumes)):
                    timestamps.append(base_time + timedelta(minutes=15 * i))
        
        print(f"Total volume samples collected: {len(all_volumes)}")
        
        hours = [ts.hour for ts in timestamps]
        day_of_week = [ts.dayofweek for ts in timestamps]
        
        # Prepare sequences
        X, y = [], []
        for i in range(len(all_volumes) - self.seq_length - 1):
            X.append(all_volumes[i:i + self.seq_length])
            y.append(all_volumes[i + self.seq_length])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        print(f"Created {len(X)} training sequences")
        
        time_features = np.column_stack([
            hours[self.seq_length:-1],
            day_of_week[self.seq_length:-1]
        ])
        
        # Split
        split_idx = int(len(X) * 0.8)
        
        X_train_seq = X[:split_idx]
        X_test_seq = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        time_train = time_features[:split_idx]
        time_test = time_features[split_idx:]
        
        # Normalize
        X_train_flat = X_train_seq.reshape(-1, self.seq_length)
        X_test_flat = X_test_seq.reshape(-1, self.seq_length)
        
        self.scaler.fit(X_train_flat)
        X_train_norm = self.scaler.transform(X_train_flat)
        X_test_norm = self.scaler.transform(X_test_flat)
        
        # For LSTM/GRU
        X_train_lstm = X_train_norm.reshape(-1, self.seq_length, 1)
        X_test_lstm = X_test_norm.reshape(-1, self.seq_length, 1)
        
        # For XGBoost
        X_train_xgb = np.column_stack([X_train_norm, time_train])
        X_test_xgb = np.column_stack([X_test_norm, time_test])
        
        print(f"Training samples: {len(X_train_lstm)}")
        print(f"Test samples: {len(X_test_lstm)}")
        
        return {
            'X_train_lstm': X_train_lstm,
            'X_test_lstm': X_test_lstm,
            'X_train_xgb': X_train_xgb,
            'X_test_xgb': X_test_xgb,
            'y_train': y_train,
            'y_test': y_test,
        }
    
    def _build_lstm(self):
        """Build LSTM model - optimized for traffic prediction"""
        model = Sequential([
            Input(shape=(self.seq_length, 1)),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def _build_gru(self):
        """Build GRU model - optimized for traffic prediction"""
        model = Sequential([
            Input(shape=(self.seq_length, 1)),
            GRU(128, return_sequences=True),
            Dropout(0.2),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def train_lstm(self, X_train, y_train, X_test, y_test, epochs=50, verbose=True):
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING LSTM MODEL")
            print("=" * 60)
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_test)}")
        
        model = self._build_lstm()
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=verbose
        )
        
        os.makedirs('saved_models', exist_ok=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1 if verbose else 0
        )
        
        self.models['lstm'] = model
        self.training_history['lstm'] = history.history
        
        if verbose:
            self._evaluate_model('lstm', model, X_test, y_test)
        
        return model
    
    def train_gru(self, X_train, y_train, X_test, y_test, epochs=50, verbose=True):
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING GRU MODEL")
            print("=" * 60)
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_test)}")
        
        model = self._build_gru()
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=verbose
        )
        
        os.makedirs('saved_models', exist_ok=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1 if verbose else 0
        )
        
        self.models['gru'] = model
        self.training_history['gru'] = history.history
        
        if verbose:
            self._evaluate_model('gru', model, X_test, y_test)
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, verbose=True):
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING XGBOOST MODEL")
            print("=" * 60)
            print(f"Training samples: {len(X_train)}")
            print(f"Feature count: {X_train.shape[1]}")
        
        params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'rmse'
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        
        self.models['xgboost'] = model
        
        if verbose:
            self._evaluate_model('xgboost', model, X_test, y_test)
            self._print_feature_importance(model)
        
        return model
    
    def _evaluate_model(self, name, model, X_test, y_test):
        if name in ['lstm', 'gru']:
            y_pred = model.predict(X_test, verbose=0).flatten()
        else:
            y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{name.upper()} Performance:")
        print(f"  MAE:  {mae:.2f} vehicles/15min")
        print(f"  RMSE: {rmse:.2f} vehicles/15min")
        print(f"  R²:   {r2:.4f}")
        
        if name not in self.training_history:
            self.training_history[name] = {}
        self.training_history[name]['test_mae'] = mae
        self.training_history[name]['test_rmse'] = rmse
        self.training_history[name]['test_r2'] = r2
    
    def _print_feature_importance(self, model):
        importance = model.feature_importances_
        print("\nXGBoost Feature Importance:")
        print(f"  Past 12 traffic volumes: {importance[:self.seq_length].sum():.3f}")
        if importance.shape[0] > self.seq_length:
            print(f"  Hour of day:            {importance[self.seq_length]:.3f}")
        if importance.shape[0] > self.seq_length + 1:
            print(f"  Day of week:            {importance[self.seq_length + 1]:.3f}")
    
    def predict(self, model_name, last_sequence, hour_of_day=12, day_of_week=2):
        """Predict next traffic volume using trained model"""
        if last_sequence is None:
            return self._fallback_prediction(hour_of_day)
        
        if model_name not in self.models:
            return self._fallback_prediction(hour_of_day)
        
        model = self.models[model_name]
        
        # Handle sequence length
        if len(last_sequence) < self.seq_length:
            last_sequence = [last_sequence[-1]] * (self.seq_length - len(last_sequence)) + last_sequence
        elif len(last_sequence) > self.seq_length:
            last_sequence = last_sequence[-self.seq_length:]
        
        # Normalize
        seq_array = np.array(last_sequence[-self.seq_length:]).reshape(1, -1)
        seq_norm = self.scaler.transform(seq_array)
        
        if model_name in ['lstm', 'gru']:
            seq_input = seq_norm.reshape(1, self.seq_length, 1)
            pred_norm = model.predict(seq_input, verbose=0)[0, 0]
            pred_volume = pred_norm * self.scaler.scale_[0] + self.scaler.mean_[0]
        else:  # xgboost
            time_features = np.array([[hour_of_day, day_of_week]])
            features = np.column_stack([seq_norm, time_features])
            pred_volume = model.predict(features)[0]
        
        return max(5, int(pred_volume))
    
    def _fallback_prediction(self, hour_of_day):
        if 7 <= hour_of_day <= 9:
            return 180 + (hour_of_day - 7) * 50
        elif 16 <= hour_of_day <= 19:
            return 160 + (hour_of_day - 16) * 40
        elif hour_of_day >= 22 or hour_of_day <= 5:
            return 30
        elif 10 <= hour_of_day <= 15:
            return 100
        else:
            return 70
    
    def save_models(self, folder='saved_models'):
        """Save models using .keras format for best compatibility"""
        os.makedirs(folder, exist_ok=True)
        
        # Save LSTM and GRU using .keras format (more compatible than .h5)
        if 'lstm' in self.models:
            self.models['lstm'].save(f'{folder}/lstm_model.keras', save_format='keras')
            print(f"✓ LSTM model saved to {folder}/lstm_model.keras")
        
        if 'gru' in self.models:
            self.models['gru'].save(f'{folder}/gru_model.keras', save_format='keras')
            print(f"✓ GRU model saved to {folder}/gru_model.keras")
        
        # Save XGBoost using joblib (better than pickle for XGBoost)
        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost'], f'{folder}/xgboost_model.joblib')
            print(f"✓ XGBoost model saved to {folder}/xgboost_model.joblib")
        
        # Save scaler
        joblib.dump(self.scaler, f'{folder}/scaler.joblib')
        print(f"✓ Scaler saved to {folder}/scaler.joblib")
    
    def load_models(self, folder='saved_models'):
        """Load models from disk"""
        if not os.path.exists(folder):
            print(f"Folder {folder} not found. Will train new models.")
            return False
        
        loaded = False
        
        # Load LSTM
        lstm_path = f'{folder}/lstm_model.keras'
        if os.path.exists(lstm_path):
            self.models['lstm'] = load_model(lstm_path, compile=False)
            # Recompile with original settings
            self.models['lstm'].compile(optimizer=Adam(learning_rate=self.learning_rate), 
                                        loss='mse', metrics=['mae'])
            print(f"✓ LSTM model loaded from {lstm_path}")
            loaded = True
        
        # Load GRU
        gru_path = f'{folder}/gru_model.keras'
        if os.path.exists(gru_path):
            self.models['gru'] = load_model(gru_path, compile=False)
            self.models['gru'].compile(optimizer=Adam(learning_rate=self.learning_rate), 
                                       loss='mse', metrics=['mae'])
            print(f"✓ GRU model loaded from {gru_path}")
            loaded = True
        
        # Load XGBoost
        xgb_path = f'{folder}/xgboost_model.joblib'
        if os.path.exists(xgb_path):
            self.models['xgboost'] = joblib.load(xgb_path)
            print(f"✓ XGBoost model loaded from {xgb_path}")
            loaded = True
        
        # Load scaler
        scaler_path = f'{folder}/scaler.joblib'
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded from {scaler_path}")
        
        return loaded


# ============================================================
# TRAINING SCRIPT
# ============================================================

def train_all_models():
    print("\n" + "=" * 60)
    print("TRAINING REAL TRAFFIC PREDICTION MODELS")
    print("=" * 60)
    
    predictor = RealTrafficPredictor(seq_length=12, batch_size=32, learning_rate=0.001)
    
    data = predictor.load_data_from_excel('Scats Data October 2006.xls')
    
    print("\nStarting training (this may take 5-10 minutes)...")
    
    # Train LSTM
    predictor.train_lstm(
        data['X_train_lstm'], data['y_train'],
        data['X_test_lstm'], data['y_test'],
        epochs=30
    )
    
    # Train GRU
    predictor.train_gru(
        data['X_train_lstm'], data['y_train'],
        data['X_test_lstm'], data['y_test'],
        epochs=30
    )
    
    # Train XGBoost
    predictor.train_xgboost(
        data['X_train_xgb'], data['y_train'],
        data['X_test_xgb'], data['y_test']
    )
    
    # Save models
    predictor.save_models()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("Models saved in 'saved_models/' folder")
    print("=" * 60)
    
    return predictor


if __name__ == "__main__":
    predictor = train_all_models()