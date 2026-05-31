# real_traffic_models.py - LSTM, GRU and XGBoost traffic predictors
# Rewritten using Parse/Model/Train/GetFlow approach with sin/cos encoding

import numpy as np
import pandas as pd
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

import datetime as dt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

    # load and preprocess Excel data, fit scalers, return train/test windows
    def loadData(self, excelFile='TrafficDataCopy.xlsx'):
        print("--- Loading traffic data from Excel (Parse logic) ---")

        df = pd.read_excel(excelFile)
        print(f"Loaded {len(df)} rows")

        # Rename datetime.time column headers to 'HH:MM:00' strings
        import datetime as _dt
        df.columns = [
            f'{c.hour:02d}:{c.minute:02d}:00' if isinstance(c, _dt.time) else c
            for c in df.columns
        ]

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
        xTrain, yTrain = self.buildWindows(trainDf, timeCols, fit=False)
        xTest,  yTest  = self.buildWindows(testDf,  timeCols, fit=False)

        # Shuffle training windows
        idx = np.random.permutation(len(xTrain))
        xTrain = xTrain[idx]
        yTrain = yTrain[idx]

        # Save reference CSVs (needed by buildReference / precomputePredictions)
        self.saveReference(trainDf, timeCols, 'data_train_reference.csv')
        self.saveReference(testDf,  timeCols, 'data_test_reference.csv')

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

    # melt grouped df from wide to long format with Time and Flow columns
    def meltToLong(self, df, timeCols):
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

    # add sin/cos encoding for day of week and time slot index
    def addCyclicalFeatures(self, melted):
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

    # build sliding windows of shape (seqLen, 6) per site
    def buildWindows(self, df, timeCols, fit=False):
        melted = self.meltToLong(df, timeCols)
        melted = self.addCyclicalFeatures(melted)

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

    # melt df to long format and save as CSV for precompute reference
    def saveReference(self, df, timeCols, filename):
        melted = self.meltToLong(df, timeCols)
        melted = self.addCyclicalFeatures(melted)
        melted.to_csv(filename, index=False)

    # -------------------------------------------------------------------------
    # Model builders (Model.py logic)
    # -------------------------------------------------------------------------

    # build 2-layer LSTM or GRU with dropout and sigmoid output
    def buildSequentialModel(self, layerType):
        LayerCls = LSTM if layerType == 'lstm' else GRU
        model = Sequential()
        model.add(LayerCls(64, input_shape=(self.seqLen, 6), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LayerCls(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    # -------------------------------------------------------------------------
    # Train logic
    # -------------------------------------------------------------------------

    # compile, train and evaluate a deep model with early stopping
    def trainDeepModel(self, modelName, xTrain, yTrain, xTest, yTest, epochs=600, verbose=True):
        if verbose:
            print(f"--- Training {modelName.upper()} model ---")
            print(f"Training samples: {len(xTrain)}, Validation samples: {len(xTest)}")

        model = self.buildSequentialModel(modelName)
        model.compile(loss='mse', optimizer='adam', metrics=['mape'])

        early = EarlyStopping(
            monitor='val_loss', patience=30, verbose=1,
            mode='auto', restore_best_weights=True
        )
        reduceLearning = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10,
            verbose=1, min_lr=1e-6
        )

        os.makedirs('saved_models', exist_ok=True)

        history = model.fit(
            xTrain, yTrain,
            batch_size=self.batchSize,
            epochs=epochs,
            validation_split=0.05,
            callbacks=[early, reduceLearning],
            verbose=1 if verbose else 0
        )

        self.models[modelName] = model
        self.trainingHistory[modelName] = history.history

        # Save loss history CSV (matching Train.py behaviour)
        pd.DataFrame.from_dict(history.history).to_csv(
            f'saved_models/{modelName}_loss.csv', encoding='utf-8', index=False
        )

        if verbose:
            self.evalModel(modelName, model, xTest, yTest)

        return model

    def trainLSTM(self, xTrain, yTrain, xTest, yTest, epochs=600, verbose=True):
        return self.trainDeepModel('lstm', xTrain, yTrain, xTest, yTest, epochs, verbose)

    # train the GRU model
    def trainGRU(self, xTrain, yTrain, xTest, yTest, epochs=600, verbose=True):
        return self.trainDeepModel('gru', xTrain, yTrain, xTest, yTest, epochs, verbose)

    # train the XGBoost model on flattened sequence windows
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

    # read test reference CSV and return per-site meta rows
    def buildReference(self):
        ref = pd.read_csv('data_test_reference.csv')
        sites = ref['SCATS Number'].unique()

        reference = []
        for site in sites:
            siteRef = ref[ref['SCATS Number'] == site].reset_index(drop=True)
            for i in range(self.seqLen, len(siteRef)):
                row = siteRef.iloc[i]
                reference.append({
                    'SCATS Number': site,
                    'Day of week':  row['Day of week'],
                    'Time':         str(row['Time']),
                })

        return pd.DataFrame(reference).reset_index(drop=True)

    # rebuild x_test windows from the reference CSV
    def buildTestWindows(self, refDf):
        featureCols = ['_scats_norm', '_flow_norm', 'day_sin', 'day_cos', 'time_sin', 'time_cos']

        fullRef = pd.read_csv('data_test_reference.csv')

        if '_scats_norm' not in fullRef.columns:
            fullRef['_scats_norm'] = self.scatsScaler.transform(
                fullRef['SCATS Number'].values.astype(float).reshape(-1, 1)
            ).flatten()

        if '_flow_norm' not in fullRef.columns:
            fullRef['_flow_norm'] = self.scaler.transform(
                fullRef['Flow'].values.reshape(-1, 1)
            ).flatten()

        if '_timeIdx' not in fullRef.columns:
            fullRef['_timeIdx'] = fullRef['Time'].apply(
                lambda t: int(str(t)[:2]) * 4 + int(str(t)[3:5]) // 15
            )

        if 'day_sin' not in fullRef.columns:
            dow  = fullRef['Day of week'].values
            tidx = fullRef['_timeIdx'].values
            fullRef['day_sin']  = np.sin(2 * np.pi * dow  / 7)
            fullRef['day_cos']  = np.cos(2 * np.pi * dow  / 7)
            fullRef['time_sin'] = np.sin(2 * np.pi * tidx / 96)
            fullRef['time_cos'] = np.cos(2 * np.pi * tidx / 96)

        xWindows, meta = [], []
        for scat, siteData in fullRef.groupby('SCATS Number'):
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

    # run all models over x_test once and cache results by (model, site, day, timeSlot)
    def precomputePredictions(self, folder='saved_models'):
        print("--- Precomputing predictions for all models ---")
        os.makedirs(folder, exist_ok=True)

        xTest, metaDf = self.buildTestWindows(None)

        cacheCounts = {}
        for modelName, model in self.models.items():
            print(f"  {modelName}...")

            if modelName in ('lstm', 'gru'):
                yPredScaled = model.predict(xTest, verbose=0).flatten()
            else:
                yPredScaled = model.predict(xTest.reshape(len(xTest), -1)).flatten()

            # Inverse-transform (matching GetFlow scaler.inverse_transform)
            yPred = self.scaler.inverse_transform(yPredScaled.reshape(-1, 1)).flatten()

            # Save per-model scaled predictions (matches GetFlow np.save pattern)
            np.save(f'{folder}/y_pred_scaled_{modelName}.npy', yPredScaled)

            for i, row in metaDf.iterrows():
                scatsNum = row['SCATS Number']
                dow      = row['Day of week']
                t        = str(row['Time'])
                timeSlot = int(t[:2]) * 4 + int(t[3:5]) // 15

                cacheKey = (modelName, scatsNum, dow, timeSlot)
                flow = max(0, float(yPred[i]))

                if cacheKey in self.predictionCache:
                    n = cacheCounts[cacheKey]
                    self.predictionCache[cacheKey] = (self.predictionCache[cacheKey] * n + flow) / (n + 1)
                    cacheCounts[cacheKey] = n + 1
                else:
                    self.predictionCache[cacheKey] = flow
                    cacheCounts[cacheKey] = 1

        self.predictionCache = {k: max(5, int(v)) for k, v in self.predictionCache.items()}

        joblib.dump(self.predictionCache, f'{folder}/prediction_cache.joblib')
        joblib.dump(self.scaler,          f'{folder}/scaler.joblib')
        joblib.dump(self.scatsScaler,     f'{folder}/scats_scaler.joblib')
        print(f"  Saved {len(self.predictionCache)} predictions to {folder}/")

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------

    # look up a precomputed prediction from the cache by 15-minute time slot (0-95)
    def predict(self, modelName, scatsNum, timeSlot=48, dayOfWeek=2):
        scatsInt = int(scatsNum) if not isinstance(scatsNum, int) else scatsNum
        key = (modelName, scatsInt, dayOfWeek, timeSlot)
        if key in self.predictionCache:
            return self.predictionCache[key]
        # fall back to average across available days
        available = [
            self.predictionCache[(modelName, scatsInt, d, timeSlot)]
            for d in range(7)
            if (modelName, scatsInt, d, timeSlot) in self.predictionCache
        ]
        return int(np.mean(available)) if available else 100

    # -------------------------------------------------------------------------
    # Eval
    # -------------------------------------------------------------------------

    # compute and print MAE, RMSE and R2 for a model on a test set
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

    # save all trained models and scalers to disk
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

    # load saved models, scalers and prediction cache from disk
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
            _s = joblib.load(scalerPath)
            # Validate: must be a 1-feature scaler (flow-only MinMaxScaler)
            if hasattr(_s, 'n_features_in_') and _s.n_features_in_ == 1:
                self.scaler = _s
                print(f"Flow scaler loaded from {scalerPath}")
            else:
                print(f"Flow scaler at {scalerPath} is incompatible (expected 1 feature) — will retrain")

        scatsScalerPath = f'{folder}/scats_scaler.joblib'
        if os.path.exists(scatsScalerPath):
            _ss = joblib.load(scatsScalerPath)
            if hasattr(_ss, 'n_features_in_') and _ss.n_features_in_ == 1:
                self.scatsScaler = _ss
                print(f"SCATS scaler loaded from {scatsScalerPath}")
            else:
                print(f"SCATS scaler at {scatsScalerPath} is incompatible — will retrain")

        cachePath = f'{folder}/prediction_cache.joblib'
        if os.path.exists(cachePath):
            self.predictionCache = joblib.load(cachePath)
            print(f"Prediction cache loaded ({len(self.predictionCache)} entries)")
        elif loaded and self.scaler is not None:
            self.precomputePredictions(folder)
        elif loaded and self.scaler is None:
            print("WARNING: Models loaded but scaler is missing or incompatible.")
            print("         Delete saved_models/ and retrain: python real_traffic_models.py")

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['lstm', 'gru', 'xgboost', 'all'], default='all',
                        help='Which model to train (default: all)')
    args = parser.parse_args()

    if args.model == 'all':
        trainAllModels()
    else:
        predictor = RealTrafficPredictor(seqLen=12, batchSize=256, lr=0.001)
        data = predictor.loadData('TrafficDataCopy.xlsx')
        predictor.loadModels()  # load existing models so others are preserved
        if args.model == 'lstm':
            predictor.trainLSTM(data['x_train'], data['y_train'], data['x_test'], data['y_test'], epochs=600)
        elif args.model == 'gru':
            predictor.trainGRU(data['x_train'], data['y_train'], data['x_test'], data['y_test'], epochs=600)
        elif args.model == 'xgboost':
            predictor.trainXGB(data['x_train_flat'], data['y_train'], data['x_test_flat'], data['y_test'])
        predictor.saveModels()
        print(f"--- {args.model.upper()} training complete ---")
