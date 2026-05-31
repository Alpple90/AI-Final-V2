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
    def __init__(self, seqLen=12, batchSize=32, lr=0.001):
        self.seqLen = seqLen
        self.batchSize = batchSize
        self.lr = lr
        self.models = {}
        self.scaler = StandardScaler()
        self.trainingHistory = {}

    # read the SCATS Excel file, flatten it into sequences and split train/test
    def loadData(self, excelFile='Scats Data October 2006.xls'):
        print("--- Loading traffic data from Excel ---")

        df = pd.read_excel(excelFile, sheet_name='Data', header=1)
        print(f"Loaded {len(df)} rows of traffic data")

        # grab volume columns (V0, V1, ... V95 etc.)
        volCols = [col for col in df.columns if str(col).startswith('V') and str(col)[1:].isdigit()]
        volCols = sorted(volCols, key=lambda x: int(x[1:]))
        print(f"Found {len(volCols)} volume columns (15-min intervals)")

        allVolumes = []
        timestamps = []

        for idx, row in df.iterrows():
            scatsNum = row.get('SCATS Number')
            if pd.isna(scatsNum):
                continue

            volumes = []
            for col in volCols:
                vol = row.get(col, 0)
                if pd.isna(vol):
                    vol = 0
                volumes.append(int(vol))

            dateVal = row.get('Date', None)
            if pd.notna(dateVal):
                allVolumes.extend(volumes)
                baseTime = pd.to_datetime(dateVal)
                for i in range(len(volumes)):
                    timestamps.append(baseTime + timedelta(minutes=15 * i))

        print(f"Total volume samples collected: {len(allVolumes)}")

        hours = [ts.hour for ts in timestamps]
        dayOfWeek = [ts.dayofweek for ts in timestamps]

        # build sliding window sequences
        X, y = [], []
        for i in range(len(allVolumes) - self.seqLen - 1):
            X.append(allVolumes[i:i + self.seqLen])
            y.append(allVolumes[i + self.seqLen])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        print(f"Created {len(X)} training sequences")

        timeFeatures = np.column_stack([
            hours[self.seqLen:-1],
            dayOfWeek[self.seqLen:-1]
        ])

        splitIdx = int(len(X) * 0.8)

        xTrainSeq = X[:splitIdx]
        xTestSeq = X[splitIdx:]
        yTrain = y[:splitIdx]
        yTest = y[splitIdx:]
        timeTrain = timeFeatures[:splitIdx]
        timeTest = timeFeatures[splitIdx:]

        # normalize flow values
        xTrainFlat = xTrainSeq.reshape(-1, self.seqLen)
        xTestFlat = xTestSeq.reshape(-1, self.seqLen)

        self.scaler.fit(xTrainFlat)
        xTrainNorm = self.scaler.transform(xTrainFlat)
        xTestNorm = self.scaler.transform(xTestFlat)

        # reshape for LSTM/GRU
        xTrainLstm = xTrainNorm.reshape(-1, self.seqLen, 1)
        xTestLstm = xTestNorm.reshape(-1, self.seqLen, 1)

        # append time features for XGBoost
        xTrainXgb = np.column_stack([xTrainNorm, timeTrain])
        xTestXgb = np.column_stack([xTestNorm, timeTest])

        print(f"Training samples: {len(xTrainLstm)}")
        print(f"Test samples: {len(xTestLstm)}")

        return {
            'X_train_lstm': xTrainLstm,
            'X_test_lstm': xTestLstm,
            'X_train_xgb': xTrainXgb,
            'X_test_xgb': xTestXgb,
            'y_train': yTrain,
            'y_test': yTest,
        }

    # define and compile a stacked LSTM network for traffic volume prediction
    def buildLSTM(self):
        model = Sequential([
            Input(shape=(self.seqLen, 1)),
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
    def buildGRU(self):
        model = Sequential([
            Input(shape=(self.seqLen, 1)),
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
    def trainLSTM(self, xTrain, yTrain, xTest, yTest, epochs=50, verbose=True):
        if verbose:
            print("--- Training LSTM model ---")
            print(f"Training samples: {len(xTrain)}")
            print(f"Validation samples: {len(xTest)}")

        model = self.buildLSTM()

        earlyStop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=verbose
        )

        os.makedirs('saved_models', exist_ok=True)

        history = model.fit(
            xTrain, yTrain,
            validation_data=(xTest, yTest),
            epochs=epochs,
            batch_size=self.batchSize,
            callbacks=[earlyStop],
            verbose=1 if verbose else 0
        )

        self.models['lstm'] = model
        self.trainingHistory['lstm'] = history.history

        if verbose:
            self.evalModel('lstm', model, xTest, yTest)

        return model

    # train the GRU model with early stopping and store it internally
    def trainGRU(self, xTrain, yTrain, xTest, yTest, epochs=50, verbose=True):
        if verbose:
            print("--- Training GRU model ---")
            print(f"Training samples: {len(xTrain)}")
            print(f"Validation samples: {len(xTest)}")

        model = self.buildGRU()

        earlyStop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=verbose
        )

        os.makedirs('saved_models', exist_ok=True)

        history = model.fit(
            xTrain, yTrain,
            validation_data=(xTest, yTest),
            epochs=epochs,
            batch_size=self.batchSize,
            callbacks=[earlyStop],
            verbose=1 if verbose else 0
        )

        self.models['gru'] = model
        self.trainingHistory['gru'] = history.history

        if verbose:
            self.evalModel('gru', model, xTest, yTest)

        return model

    # fit an XGBoost regressor with tuned hyperparameters and store it internally
    def trainXGB(self, xTrain, yTrain, xTest, yTest, verbose=True):
        if verbose:
            print("--- Training XGBoost model ---")
            print(f"Training samples: {len(xTrain)}")
            print(f"Feature count: {xTrain.shape[1]}")

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
        model.fit(xTrain, yTrain, verbose=False)

        self.models['xgboost'] = model

        if verbose:
            self.evalModel('xgboost', model, xTest, yTest)
            self.printFeatureImportance(model)

        return model

    # compute MAE, RMSE and R2 on the test set and print them
    def evalModel(self, name, model, xTest, yTest):
        if name in ['lstm', 'gru']:
            yPred = model.predict(xTest, verbose=0).flatten()
        else:
            yPred = model.predict(xTest)

        mae = mean_absolute_error(yTest, yPred)
        rmse = np.sqrt(mean_squared_error(yTest, yPred))
        r2 = r2_score(yTest, yPred)

        print(f"\n{name.upper()} Performance:")
        print(f"  MAE:  {mae:.2f} vehicles/15min")
        print(f"  RMSE: {rmse:.2f} vehicles/15min")
        print(f"  R2:   {r2:.4f}")

        if name not in self.trainingHistory:
            self.trainingHistory[name] = {}
        self.trainingHistory[name]['test_mae'] = mae
        self.trainingHistory[name]['test_rmse'] = rmse
        self.trainingHistory[name]['test_r2'] = r2

    # print how much each input feature contributed to the XGBoost model
    def printFeatureImportance(self, model):
        importance = model.feature_importances_
        print("\nXGBoost Feature Importance:")
        print(f"  Past 12 traffic volumes: {importance[:self.seqLen].sum():.3f}")
        if importance.shape[0] > self.seqLen:
            print(f"  Hour of day:            {importance[self.seqLen]:.3f}")
        if importance.shape[0] > self.seqLen + 1:
            print(f"  Day of week:            {importance[self.seqLen + 1]:.3f}")

    # run the chosen model on the given traffic sequence and return a flow prediction
    def predict(self, modelName, lastSeq, hourOfDay=12, dayOfWeek=2):
        model = self.models[modelName]

        # pad or trim sequence to the right length
        if len(lastSeq) < self.seqLen:
            lastSeq = [lastSeq[-1]] * (self.seqLen - len(lastSeq)) + lastSeq
        elif len(lastSeq) > self.seqLen:
            lastSeq = lastSeq[-self.seqLen:]

        seqArray = np.array(lastSeq[-self.seqLen:]).reshape(1, -1)
        seqNorm = self.scaler.transform(seqArray)

        if modelName in ['lstm', 'gru']:
            seqInput = seqNorm.reshape(1, self.seqLen, 1)
            predNorm = model.predict(seqInput, verbose=0)[0, 0]
            predVolume = predNorm * self.scaler.scale_[0] + self.scaler.mean_[0]
        else:  # xgboost
            timeFeatures = np.array([[hourOfDay, dayOfWeek]])
            features = np.column_stack([seqNorm, timeFeatures])
            predVolume = model.predict(features)[0]

        return max(5, int(predVolume))

    # write all trained models and the scaler to disk
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

        joblib.dump(self.scaler, f'{folder}/scaler.joblib')
        print(f"Scaler saved to {folder}/scaler.joblib")

    # load previously saved models and scaler from disk, returns False if nothing found
    def loadModels(self, folder='saved_models'):
        if not os.path.exists(folder):
            print(f"Folder {folder} not found. Will train new models.")
            return False

        loaded = False

        lstmPath = f'{folder}/lstm_model.keras'
        if os.path.exists(lstmPath):
            self.models['lstm'] = load_model(lstmPath, compile=False)
            # recompile so we can keep training later if needed
            self.models['lstm'].compile(optimizer=Adam(learning_rate=self.lr),
                                        loss='mse', metrics=['mae'])
            print(f"LSTM model loaded from {lstmPath}")
            loaded = True

        gruPath = f'{folder}/gru_model.keras'
        if os.path.exists(gruPath):
            self.models['gru'] = load_model(gruPath, compile=False)
            self.models['gru'].compile(optimizer=Adam(learning_rate=self.lr),
                                       loss='mse', metrics=['mae'])
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
            print(f"Scaler loaded from {scalerPath}")

        return loaded


# convenience function to train all three models in one go and save them
def trainAllModels():
    print("--- Training all traffic prediction models ---")

    predictor = RealTrafficPredictor(seqLen=12, batchSize=32, lr=0.001)

    data = predictor.loadData('Scats Data October 2006.xls')

    print("\nStarting training (this may take 5-10 minutes)...")

    predictor.trainLSTM(
        data['X_train_lstm'], data['y_train'],
        data['X_test_lstm'], data['y_test'], epochs=30
    )

    predictor.trainGRU(
        data['X_train_lstm'], data['y_train'],
        data['X_test_lstm'], data['y_test'], epochs=30
    )

    predictor.trainXGB(
        data['X_train_xgb'], data['y_train'],
        data['X_test_xgb'], data['y_test']
    )

    predictor.saveModels()

    print("--- Training complete. Models saved in 'saved_models/' ---")

    return predictor


if __name__ == "__main__":
    predictor = trainAllModels()
