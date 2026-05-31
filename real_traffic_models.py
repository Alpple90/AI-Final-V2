# real_traffic_models.py - LSTM, GRU and XGBoost traffic predictors
# Rewritten using Parse/Model/Train/GetFlow approach with sin/cos encoding

import numpy as np
import pandas as pd
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import datetime as dt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Time column names representing 15-min intervals across a day
TIME_COLS = [f'{h:02d}:{m:02d}:00' for h in range(24) for m in (0, 15, 30, 45)]

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


class RealTrafficPredictor:
    def __init__(self, seqLen=12, batchSize=256, lr=0.001):
        self.seqLen = seqLen
        self.batchSize = batchSize
        self.lr = lr
        self.models = {}
        self.scaler = None        # MinMaxScaler for flow values
        self.scatsScaler = None   # MinMaxScaler for SCATS numbers
        self.trainingHistory = {}
        self.predictionCache = {}

    # -------------------------------------------------------------------------
    # Parse logic
    # -------------------------------------------------------------------------

    def loadData(self, excelFile='TrafficDataCopy.xlsx'):
        print("--- Loading traffic data from Excel (Parse logic) ---")

        df = pd.read_excel(excelFile)
        print(f"Loaded {len(df)} rows")

        # Identify time columns present in the DataFrame
        timeCols = [c for c in TIME_COLS if c in df.columns]
        print(f"Found {len(timeCols)} time columns")

        # Group by SCATS Number + Date, summing time columns
        df['Date'] = pd.to_datetime(df['Date'])
        grouped = df.groupby(['SCATS Number', 'Date'])[timeCols].sum().reset_index()

        # Add Day of week (0=Monday ... 6=Sunday)
        grouped['Day of week'] = grouped['Date'].dt.dayofweek

        # Extract day-of-month for splitting
        grouped['_day'] = grouped['Date'].dt.day

        # Split: train = day <= 24, test = day >= 24 (overlap at 24 for windowing)
        trainDf = grouped[grouped['_day'] <= 24].copy()
        testDf  = grouped[grouped['_day'] >= 24].copy()

        print(f"Train rows: {len(trainDf)}, Test rows: {len(testDf)}")

        # Fit scalers on training data
        allScatsNums = grouped['SCATS Number'].unique().astype(float)
        self.scatsScaler = MinMaxScaler(feature_range=(0, 1))
        self.scatsScaler.fit(allScatsNums.reshape(-1, 1))

        trainFlows = trainDf[timeCols].values.flatten().reshape(-1, 1)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(trainFlows)

        # Build windows for train and test sets
        xTrain, yTrain = self._buildWindows(trainDf, timeCols, fit=False)
        xTest,  yTest  = self._buildWindows(testDf,  timeCols, fit=False)

        # Shuffle training windows
        idx = np.random.permutation(len(xTrain))
        xTrain = xTrain[idx]
        yTrain = yTrain[idx]

        # Save reference CSVs (needed by buildReference / precomputePredictions)
        self._saveReference(trainDf, timeCols, 'data_train_reference.csv')
        self._saveReference(testDf,  timeCols, 'data_test_reference.csv')

        print(f"Training windows: {len(xTrain)}, Test windows: {len(xTest)}")
        print(f"Window shape: {xTrain.shape}")  # (N, 12, 6)

        # Flatten for XGBoost
        xTrainFlat = xTrain.reshape(len(xTrain), -1)   # (N, 72)
        xTestFlat  = xTest.reshape(len(xTest),  -1)

        return {
            'x_train': xTrain,
            'x_test':  xTest,
            'x_train_flat': xTrainFlat,
            'x_test_flat':  xTestFlat,
            'y_train': yTrain,
            'y_test':  yTest,
        }

    def _meltToLong(self, df, timeCols):
        """Melt a grouped df to long format with Time and Flow columns."""
        melted = df.melt(
            id_vars=['SCATS Number', 'Date', 'Day of week'],
            value_vars=timeCols,
            var_name='Time',
            value_name='Flow'
        )
        # Convert Time string to datetime.time object
        melted['Time'] = melted['Time'].apply(
            lambda s: dt.time(int(s[:2]), int(s[3:5]), int(s[6:8]))
        )
        melted = melted.sort_values(['SCATS Number', 'Date', 'Time']).reset_index(drop=True)
        return melted

    def _addCyclicalFeatures(self, melted):
        """Add sin/cos encoding for Day of week (period 7) and Time slot (period 96)."""
        # Time slot index 0-95
        melted['_timeIdx'] = melted['Time'].apply(
            lambda t: t.hour * 4 + t.minute // 15
        )
        dow = melted['Day of week'].values
        tidx = melted['_timeIdx'].values

        melted['day_sin']  = np.sin(2 * np.pi * dow  / 7)
        melted['day_cos']  = np.cos(2 * np.pi * dow  / 7)
        melted['time_sin'] = np.sin(2 * np.pi * tidx / 96)
        melted['time_cos'] = np.cos(2 * np.pi * tidx / 96)
        return melted

    def _buildWindows(self, df, timeCols, fit=False):
        """Build sliding windows of shape (seqLen, 6) per site."""
        melted = self._meltToLong(df, timeCols)
        melted = self._addCyclicalFeatures(melted)

        # Normalize scats numbers and flow
        scatsNorm = self.scatsScaler.transform(
            melted['SCATS Number'].values.astype(float).reshape(-1, 1)
        ).flatten()
        flowNorm = self.scaler.transform(
            melted['Flow'].values.reshape(-1, 1)
        ).flatten()

        melted['_scats_norm'] = scatsNorm
        melted['_flow_norm']  = flowNorm

        X, y = [], []
        featureCols = ['_scats_norm', '_flow_norm', 'day_sin', 'day_cos', 'time_sin', 'time_cos']

        for scat, siteData in melted.groupby('SCATS Number'):
            siteData = siteData.reset_index(drop=True)
            features = siteData[featureCols].values  # (T, 6)
            flows    = siteData['_flow_norm'].values  # (T,)

            for i in range(self.seqLen, len(features)):
                X.append(features[i - self.seqLen:i])   # (12, 6)
                y.append(flows[i])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _saveReference(self, df, timeCols, filename):
        """Melt df to long and save to CSV for GetFlow reference."""
        melted = self._meltToLong(df, timeCols)
        melted = self._addCyclicalFeatures(melted)
        melted.to_csv(filename, index=False)

    # -------------------------------------------------------------------------
    # Model builders (Model.py logic)
    # -------------------------------------------------------------------------

    def _buildSequentialModel(self, layerType):
        """Build 2-layer LSTM or GRU with Dropout, sigmoid output."""
        LayerCls = LSTM if layerType == 'lstm' else GRU
        model = Sequential([
            Input(shape=(self.seqLen, 6)),
            LayerCls(64, return_sequences=True),
            Dropout(0.2),
            LayerCls(64),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        return model

    # -------------------------------------------------------------------------
    # Train logic
    # -------------------------------------------------------------------------

    def _trainDeepModel(self, modelName, xTrain, yTrain, xTest, yTest, epochs=600, verbose=True):
        if verbose:
            print(f"--- Training {modelName.upper()} model ---")
            print(f"Training samples: {len(xTrain)}, Validation samples: {len(xTest)}")

        model = self._buildSequentialModel(modelName)
        model.compile(loss='mse', optimizer='adam', metrics=['mape'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)
        ]

        os.makedirs('saved_models', exist_ok=True)

        history = model.fit(
            xTrain, yTrain,
            validation_split=0.05,
            epochs=epochs,
            batch_size=self.batchSize,
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )

        self.models[modelName] = model
        self.trainingHistory[modelName] = history.history

        if verbose:
            self.evalModel(modelName, model, xTest, yTest)

        return model

    def trainLSTM(self, xTrain, yTrain, xTest, yTest, epochs=600, verbose=True):
        return self._trainDeepModel('lstm', xTrain, yTrain, xTest, yTest, epochs, verbose)

    def trainGRU(self, xTrain, yTrain, xTest, yTest, epochs=600, verbose=True):
        return self._trainDeepModel('gru', xTrain, yTrain, xTest, yTest, epochs, verbose)

    def trainXGB(self, xTrainFlat, yTrain, xTestFlat, yTest, verbose=True):
        if verbose:
            print("--- Training XGBoost model ---")
            print(f"Training samples: {len(xTrainFlat)}, Features: {xTrainFlat.shape[1]}")

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
        model.fit(xTrainFlat, yTrain, verbose=False)

        self.models['xgboost'] = model

        if verbose:
            self.evalModel('xgboost', model, xTestFlat, yTest)

        return model

    # -------------------------------------------------------------------------
    # GetFlow logic
    # -------------------------------------------------------------------------

    def _buildReference(self):
        """Read data_test_reference.csv, return meta df with (SCATS Number, Day of week, Time)."""
        refDf = pd.read_csv('data_test_reference.csv')
        # Convert Time back to string (it was saved as HH:MM:SS)
        refDf['Time'] = refDf['Time'].astype(str)
        return refDf

    def _buildTestWindows(self, refDf):
        """Rebuild x_test windows from the reference CSV in the same order as training."""
        featureCols = ['_scats_norm', '_flow_norm', 'day_sin', 'day_cos', 'time_sin', 'time_cos']

        # Re-normalise (refDf already has the computed columns from _saveReference)
        # But CSV load may have lost them — recompute if missing
        if '_scats_norm' not in refDf.columns:
            scatsNorm = self.scatsScaler.transform(
                refDf['SCATS Number'].values.astype(float).reshape(-1, 1)
            ).flatten()
            refDf['_scats_norm'] = scatsNorm

        if '_flow_norm' not in refDf.columns:
            flowNorm = self.scaler.transform(
                refDf['Flow'].values.reshape(-1, 1)
            ).flatten()
            refDf['_flow_norm'] = flowNorm

        if 'day_sin' not in refDf.columns:
            dow  = refDf['Day of week'].values
            tidx = refDf['_timeIdx'].values
            refDf['day_sin']  = np.sin(2 * np.pi * dow  / 7)
            refDf['day_cos']  = np.cos(2 * np.pi * dow  / 7)
            refDf['time_sin'] = np.sin(2 * np.pi * tidx / 96)
            refDf['time_cos'] = np.cos(2 * np.pi * tidx / 96)

        xWindows, meta = [], []
        for scat, siteData in refDf.groupby('SCATS Number'):
            siteData = siteData.reset_index(drop=True)
            features = siteData[featureCols].values
            for i in range(self.seqLen, len(features)):
                xWindows.append(features[i - self.seqLen:i])
                row = siteData.iloc[i]
                meta.append({
                    'SCATS Number': int(scat),
                    'Day of week':  int(row['Day of week']),
                    'Time':         str(row['Time']),
                })

        return np.array(xWindows, dtype=np.float32), pd.DataFrame(meta)

    def precomputePredictions(self, folder='saved_models'):
        """GetFlow precompute: run model.predict on full x_test, cache results."""
        print("--- Precomputing predictions for all models ---")
        os.makedirs(folder, exist_ok=True)

        refDf = self._buildReference()
        xTest, metaDf = self._buildTestWindows(refDf)

        self.predictionCache = {}

        for modelName, model in self.models.items():
            print(f"  {modelName}...")

            if modelName in ('lstm', 'gru'):
                yPredScaled = model.predict(xTest, verbose=0).flatten()
            else:
                xFlat = xTest.reshape(len(xTest), -1)
                yPredScaled = model.predict(xFlat).flatten()

            # Inverse-transform flow predictions
            yPred = self.scaler.inverse_transform(yPredScaled.reshape(-1, 1)).flatten()

            for i, row in metaDf.iterrows():
                scatsNum = row['SCATS Number']
                dow      = row['Day of week']
                timeStr  = row['Time']  # 'HH:MM:SS'
                hour     = int(timeStr[:2])

                cacheKey = (modelName, scatsNum, dow, hour)
                flow = max(0, float(yPred[i]))

                # Average if multiple predictions for same key
                if cacheKey in self.predictionCache:
                    existing = self.predictionCache[cacheKey]
                    self.predictionCache[cacheKey] = (existing + flow) / 2
                else:
                    self.predictionCache[cacheKey] = flow

        # Convert to int
        self.predictionCache = {k: max(5, int(v)) for k, v in self.predictionCache.items()}

        joblib.dump(self.predictionCache, f'{folder}/prediction_cache.joblib')
        joblib.dump(self.scaler,       f'{folder}/scaler.joblib')
        joblib.dump(self.scatsScaler,  f'{folder}/scats_scaler.joblib')
        print(f"  Saved {len(self.predictionCache)} predictions to {folder}/")

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------

    def predict(self, modelName, scatsNum, hourOfDay=12, dayOfWeek=2):
        """Look up precomputed prediction. Key: (modelName, scatsNum_int, dow_int, hour_int)."""
        scatsInt = int(scatsNum) if not isinstance(scatsNum, int) else scatsNum
        key = (modelName, scatsInt, dayOfWeek, hourOfDay)
        if key in self.predictionCache:
            return self.predictionCache[key]
        # Fall back to average across available days
        available = [
            self.predictionCache[(modelName, scatsInt, d, hourOfDay)]
            for d in range(7)
            if (modelName, scatsInt, d, hourOfDay) in self.predictionCache
        ]
        return int(np.mean(available)) if available else 100

    # -------------------------------------------------------------------------
    # Eval
    # -------------------------------------------------------------------------

    def evalModel(self, name, model, xTest, yTest):
        if name in ('lstm', 'gru'):
            yPred = model.predict(xTest, verbose=0).flatten()
        else:
            yPred = model.predict(xTest)

        # Inverse-transform if scaler is available
        if self.scaler is not None:
            yPredInv = self.scaler.inverse_transform(yPred.reshape(-1, 1)).flatten()
            yTestInv = self.scaler.inverse_transform(np.array(yTest).reshape(-1, 1)).flatten()
        else:
            yPredInv = yPred
            yTestInv = yTest

        mae  = mean_absolute_error(yTestInv, yPredInv)
        rmse = np.sqrt(mean_squared_error(yTestInv, yPredInv))
        r2   = r2_score(yTestInv, yPredInv)

        print(f"\n{name.upper()} Performance:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2:   {r2:.4f}")

        if name not in self.trainingHistory:
            self.trainingHistory[name] = {}
        self.trainingHistory[name]['test_mae']  = mae
        self.trainingHistory[name]['test_rmse'] = rmse
        self.trainingHistory[name]['test_r2']   = r2

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------

    def saveModels(self, folder='saved_models'):
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

        if self.scaler is not None:
            joblib.dump(self.scaler,      f'{folder}/scaler.joblib')
            joblib.dump(self.scatsScaler, f'{folder}/scats_scaler.joblib')
            print(f"Scalers saved to {folder}/")

        self.precomputePredictions(folder)

    def loadModels(self, folder='saved_models'):
        if not os.path.exists(folder):
            print(f"Folder {folder} not found. Will train new models.")
            return False

        loaded = False

        lstmPath = f'{folder}/lstm_model.keras'
        if os.path.exists(lstmPath):
            self.models['lstm'] = load_model(lstmPath, compile=False)
            self.models['lstm'].compile(loss='mse', optimizer='adam', metrics=['mape'])
            print(f"LSTM model loaded from {lstmPath}")
            loaded = True

        gruPath = f'{folder}/gru_model.keras'
        if os.path.exists(gruPath):
            self.models['gru'] = load_model(gruPath, compile=False)
            self.models['gru'].compile(loss='mse', optimizer='adam', metrics=['mape'])
            print(f"GRU model loaded from {gruPath}")
            loaded = True

        xgbPath = f'{folder}/xgboost_model.joblib'
        if os.path.exists(xgbPath):
            self.models['xgboost'] = joblib.load(xgbPath)
            print(f"XGBoost model loaded from {xgbPath}")
            loaded = True

        scalerPath = f'{folder}/scaler.joblib'
        if os.path.exists(scalerPath):
            self.scaler = joblib.load(scalerPath)
            print(f"Flow scaler loaded from {scalerPath}")

        scatsScalerPath = f'{folder}/scats_scaler.joblib'
        if os.path.exists(scatsScalerPath):
            self.scatsScaler = joblib.load(scatsScalerPath)
            print(f"SCATS scaler loaded from {scatsScalerPath}")

        cachePath = f'{folder}/prediction_cache.joblib'
        if os.path.exists(cachePath):
            self.predictionCache = joblib.load(cachePath)
            print(f"Prediction cache loaded ({len(self.predictionCache)} entries)")
        elif loaded and self.scaler is not None:
            self.precomputePredictions(folder)

        return loaded


# convenience function to train all three models in one go and save them
def trainAllModels():
    print("--- Training all traffic prediction models ---")

    predictor = RealTrafficPredictor(seqLen=12, batchSize=256, lr=0.001)

    data = predictor.loadData('TrafficDataCopy.xlsx')

    print("\nStarting training (this may take a while)...")

    predictor.trainLSTM(
        data['x_train'], data['y_train'],
        data['x_test'],  data['y_test'], epochs=600
    )

    predictor.trainGRU(
        data['x_train'], data['y_train'],
        data['x_test'],  data['y_test'], epochs=600
    )

    predictor.trainXGB(
        data['x_train_flat'], data['y_train'],
        data['x_test_flat'],  data['y_test']
    )

    predictor.saveModels()

    print("--- Training complete. Models saved in 'saved_models/' ---")

    return predictor


if __name__ == "__main__":
    predictor = trainAllModels()
