# real_traffic_models.py - LSTM, GRU and XGBoost traffic predictors

import numpy as np
import pandas as pd
from datetime import timedelta
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class RealTrafficPredictor:
    def __init__(self, seq_len=12, batch_size=32, lr=0.001):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.models = {}
        self.scaler = StandardScaler()
        self.training_history = {}

    # read the SCATS Excel file, flatten it into sequences and split train/test
    def load_data(self, excel_file='Scats Data October 2006.xls'):
        print("--- Loading traffic data from Excel ---")

        df = pd.read_excel(excel_file, sheet_name='Data', header=1)
        print(f"Loaded {len(df)} rows of traffic data")

        # grab volume columns (V0, V1, ... V95 etc.)
        vol_cols = [col for col in df.columns if str(col).startswith('V') and str(col)[1:].isdigit()]
        vol_cols = sorted(vol_cols, key=lambda x: int(x[1:]))
        print(f"Found {len(vol_cols)} volume columns (15-min intervals)")

        all_volumes = []
        timestamps = []

        for idx, row in df.iterrows():
            scats_num = row.get('SCATS Number')
            if pd.isna(scats_num):
                continue

            volumes = []
            for col in vol_cols:
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

        # build sliding window sequences
        X, y = [], []
        for i in range(len(all_volumes) - self.seq_len - 1):
            X.append(all_volumes[i:i + self.seq_len])
            y.append(all_volumes[i + self.seq_len])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        print(f"Created {len(X)} training sequences")

        time_features = np.column_stack([
            hours[self.seq_len:-1],
            day_of_week[self.seq_len:-1]
        ])

        split_idx = int(len(X) * 0.8)

        x_train_seq = X[:split_idx]
        x_test_seq = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        time_train = time_features[:split_idx]
        time_test = time_features[split_idx:]

        # normalize flow values
        x_train_flat = x_train_seq.reshape(-1, self.seq_len)
        x_test_flat = x_test_seq.reshape(-1, self.seq_len)

        self.scaler.fit(x_train_flat)
        x_train_norm = self.scaler.transform(x_train_flat)
        x_test_norm = self.scaler.transform(x_test_flat)

        # reshape for LSTM/GRU
        x_train_lstm = x_train_norm.reshape(-1, self.seq_len, 1)
        x_test_lstm = x_test_norm.reshape(-1, self.seq_len, 1)

        # append time features for XGBoost
        x_train_xgb = np.column_stack([x_train_norm, time_train])
        x_test_xgb = np.column_stack([x_test_norm, time_test])

        print(f"Training samples: {len(x_train_lstm)}")
        print(f"Test samples: {len(x_test_lstm)}")

        return {
            'X_train_lstm': x_train_lstm,
            'X_test_lstm': x_test_lstm,
            'X_train_xgb': x_train_xgb,
            'X_test_xgb': x_test_xgb,
            'y_train': y_train,
            'y_test': y_test,
        }

    # define and compile a stacked LSTM network for traffic volume prediction
    def build_lstm(self):
        model = Sequential([
            Input(shape=(self.seq_len, 1)),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        optimizer = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    # define and compile a stacked GRU network for traffic volume prediction
    def build_gru(self):
        model = Sequential([
            Input(shape=(self.seq_len, 1)),
            GRU(128, return_sequences=True),
            Dropout(0.2),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        optimizer = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    # train the LSTM model with early stopping and store it internally
    def train_lstm(self, x_train, y_train, x_test, y_test, epochs=50, verbose=True):
        if verbose:
            print("--- Training LSTM model ---")
            print(f"Training samples: {len(x_train)}")
            print(f"Validation samples: {len(x_test)}")

        model = self.build_lstm()

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=verbose
        )

        os.makedirs('saved_models', exist_ok=True)

        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1 if verbose else 0
        )

        self.models['lstm'] = model
        self.training_history['lstm'] = history.history

        if verbose:
            self.eval_model('lstm', model, x_test, y_test)

        return model

    # train the GRU model with early stopping and store it internally
    def train_gru(self, x_train, y_train, x_test, y_test, epochs=50, verbose=True):
        if verbose:
            print("--- Training GRU model ---")
            print(f"Training samples: {len(x_train)}")
            print(f"Validation samples: {len(x_test)}")

        model = self.build_gru()

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=verbose
        )

        os.makedirs('saved_models', exist_ok=True)

        history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1 if verbose else 0
        )

        self.models['gru'] = model
        self.training_history['gru'] = history.history

        if verbose:
            self.eval_model('gru', model, x_test, y_test)

        return model

    # fit an XGBoost regressor with tuned hyperparameters and store it internally
    def train_xgb(self, x_train, y_train, x_test, y_test, verbose=True):
        if verbose:
            print("--- Training XGBoost model ---")
            print(f"Training samples: {len(x_train)}")
            print(f"Feature count: {x_train.shape[1]}")

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
        model.fit(x_train, y_train, verbose=False)

        self.models['xgboost'] = model

        if verbose:
            self.eval_model('xgboost', model, x_test, y_test)
            self.print_feature_importance(model)

        return model

    # compute MAE, RMSE and R2 on the test set and print them
    def eval_model(self, name, model, x_test, y_test):
        if name in ['lstm', 'gru']:
            y_pred = model.predict(x_test, verbose=0).flatten()
        else:
            y_pred = model.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"\n{name.upper()} Performance:")
        print(f"  MAE:  {mae:.2f} vehicles/15min")
        print(f"  RMSE: {rmse:.2f} vehicles/15min")
        print(f"  R2:   {r2:.4f}")

        if name not in self.training_history:
            self.training_history[name] = {}
        self.training_history[name]['test_mae'] = mae
        self.training_history[name]['test_rmse'] = rmse
        self.training_history[name]['test_r2'] = r2

    # print how much each input feature contributed to the XGBoost model
    def print_feature_importance(self, model):
        importance = model.feature_importances_
        print("\nXGBoost Feature Importance:")
        print(f"  Past 12 traffic volumes: {importance[:self.seq_len].sum():.3f}")
        if importance.shape[0] > self.seq_len:
            print(f"  Hour of day:            {importance[self.seq_len]:.3f}")
        if importance.shape[0] > self.seq_len + 1:
            print(f"  Day of week:            {importance[self.seq_len + 1]:.3f}")

    # run the chosen model on the given traffic sequence and return a flow prediction
    def predict(self, model_name, last_seq, hour_of_day=12, day_of_week=2):
        # fall back to time-of-day heuristic if no model or sequence
        if last_seq is None:
            return self.fallback_predict(hour_of_day)

        if model_name not in self.models:
            return self.fallback_predict(hour_of_day)

        model = self.models[model_name]

        # pad or trim sequence to the right length
        if len(last_seq) < self.seq_len:
            last_seq = [last_seq[-1]] * (self.seq_len - len(last_seq)) + last_seq
        elif len(last_seq) > self.seq_len:
            last_seq = last_seq[-self.seq_len:]

        seq_array = np.array(last_seq[-self.seq_len:]).reshape(1, -1)
        seq_norm = self.scaler.transform(seq_array)

        if model_name in ['lstm', 'gru']:
            seq_input = seq_norm.reshape(1, self.seq_len, 1)
            pred_norm = model.predict(seq_input, verbose=0)[0, 0]
            pred_volume = pred_norm * self.scaler.scale_[0] + self.scaler.mean_[0]
        else:  # xgboost
            time_features = np.array([[hour_of_day, day_of_week]])
            features = np.column_stack([seq_norm, time_features])
            pred_volume = model.predict(features)[0]

        return max(5, int(pred_volume))

    # estimate traffic flow from the hour of day when no trained model is available
    def fallback_predict(self, hour_of_day):
        # rough hourly traffic profile when no model is available
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

    # write all trained models and the scaler to disk
    def save_models(self, folder='saved_models'):
        os.makedirs(folder, exist_ok=True)

        if 'lstm' in self.models:
            self.models['lstm'].save(f'{folder}/lstm_model.keras', save_format='keras')
            print(f"LSTM model saved to {folder}/lstm_model.keras")

        if 'gru' in self.models:
            self.models['gru'].save(f'{folder}/gru_model.keras', save_format='keras')
            print(f"GRU model saved to {folder}/gru_model.keras")

        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost'], f'{folder}/xgboost_model.joblib')
            print(f"XGBoost model saved to {folder}/xgboost_model.joblib")

        joblib.dump(self.scaler, f'{folder}/scaler.joblib')
        print(f"Scaler saved to {folder}/scaler.joblib")

    # load previously saved models and scaler from disk, returns False if nothing found
    def load_models(self, folder='saved_models'):
        if not os.path.exists(folder):
            print(f"Folder {folder} not found. Will train new models.")
            return False

        loaded = False

        lstm_path = f'{folder}/lstm_model.keras'
        if os.path.exists(lstm_path):
            self.models['lstm'] = load_model(lstm_path, compile=False)
            # recompile so we can keep training later if needed
            self.models['lstm'].compile(optimizer=Adam(learning_rate=self.lr),
                                        loss='mse', metrics=['mae'])
            print(f"LSTM model loaded from {lstm_path}")
            loaded = True

        gru_path = f'{folder}/gru_model.keras'
        if os.path.exists(gru_path):
            self.models['gru'] = load_model(gru_path, compile=False)
            self.models['gru'].compile(optimizer=Adam(learning_rate=self.lr),
                                       loss='mse', metrics=['mae'])
            print(f"GRU model loaded from {gru_path}")
            loaded = True

        xgb_path = f'{folder}/xgboost_model.joblib'
        if os.path.exists(xgb_path):
            self.models['xgboost'] = joblib.load(xgb_path)
            print(f"XGBoost model loaded from {xgb_path}")
            loaded = True

        scaler_path = f'{folder}/scaler.joblib'
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")

        return loaded


# convenience function to train all three models in one go and save them
def train_all_models():
    print("--- Training all traffic prediction models ---")

    predictor = RealTrafficPredictor(seq_len=12, batch_size=32, lr=0.001)

    data = predictor.load_data('Scats Data October 2006.xls')

    print("\nStarting training (this may take 5-10 minutes)...")

    predictor.train_lstm(
        data['X_train_lstm'], data['y_train'],
        data['X_test_lstm'], data['y_test'], epochs=30
    )

    predictor.train_gru(
        data['X_train_lstm'], data['y_train'],
        data['X_test_lstm'], data['y_test'], epochs=30
    )

    predictor.train_xgb(
        data['X_train_xgb'], data['y_train'],
        data['X_test_xgb'], data['y_test']
    )

    predictor.save_models()

    print("--- Training complete. Models saved in 'saved_models/' ---")

    return predictor


if __name__ == "__main__":
    predictor = train_all_models()
