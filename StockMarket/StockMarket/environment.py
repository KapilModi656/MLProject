import numpy as np
import xgboost as xgb
import joblib
import os
import pandas as pd
import talib

class StockTradingEnvironment:
    """
    A custom Reinforcement Learning environment for stock trading simulation.

    This environment uses pre-trained XGBoost models to predict the next day's
    OHLCV (Open, High, Low, Close, Volume) data based on the current state
    and an action (Buy, Hold, Sell).
    """
    def __init__(self, stock_name):
        self.name = stock_name
        self.action_space = [1, 0, -1]  # Buy, Hold, Sell
        self.observation_space = None  # To be defined after loading data

        # --- FIX 1: Correctly load models and scalers ---
        # Each model must be loaded into a separate Booster instance.
        # Also corrected the path for loading the X_scaler.
        
        # Define paths
        model_path = os.path.join(os.getcwd(),'StockMarket', 'Notebook', 'all_models')
        scaler_path = os.path.join(os.getcwd(),'StockMarket', 'Notebook', 'all_scalers')

        print("Loading models and scalers...")
        
        # Load XGBoost models
        self.model_close = xgb.Booster()
        self.model_close.load_model(os.path.join(model_path, f'{self.name}_close_model.json'))
        
        self.model_open = xgb.Booster()
        self.model_open.load_model(os.path.join(model_path, f'{self.name}_open_model.json'))
        
        self.model_high = xgb.Booster()
        self.model_high.load_model(os.path.join(model_path, f'{self.name}_high_model.json'))
        
        self.model_low = xgb.Booster()
        self.model_low.load_model(os.path.join(model_path, f'{self.name}_low_model.json'))
        
        self.model_volume = xgb.Booster()
        self.model_volume.load_model(os.path.join(model_path, f'{self.name}_volume_model.json'))

        # Load scalers
        self.X_scaler = joblib.load(os.path.join(scaler_path, f'{self.name}_X_scaler.pkl'))
        self.y1_scaler = joblib.load(os.path.join(scaler_path, f'{self.name}_y1_scaler.pkl'))
        self.y2_scaler = joblib.load(os.path.join(scaler_path, f'{self.name}_y2_scaler.pkl'))
        self.y3_scaler = joblib.load(os.path.join(scaler_path, f'{self.name}_y3_scaler.pkl'))
        self.y4_scaler = joblib.load(os.path.join(scaler_path, f'{self.name}_y4_scaler.pkl'))
        self.y5_scaler = joblib.load(os.path.join(scaler_path, f'{self.name}_y5_scaler.pkl'))
        
        print("Environment initialized successfully.")

        self.full_data = None
        self.current_state = None

    def reset(self):
        """
        Resets the environment to the initial state using historical data.
        Returns the initial state.
        """
        # Load the base historical data
        csv_path = os.path.join(os.getcwd(),'StockMarket', 'Notebook', f'{self.name}.csv')
        self.full_data = pd.read_csv(csv_path)
        
        # Drop columns that might be from previous runs
        cols_to_drop = ["Close_", "Open_", "High_", "Low_", "Volume_", "signal", "trend"]
        self.full_data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # --- FIX 2: Calculate features on the entire historical dataset ---
        self.full_data = self._calculate_features(self.full_data)
        
        # Drop rows with NaNs created by indicators with a lookback period
        self.full_data.dropna(inplace=True)
        if self.observation_space is None:
            self.observation_space = self.full_data.shape[1]
        
        # The initial state is the last row of the historical data
        self.current_state = self.full_data.iloc[-1].values
        
        return self.current_state

    def step(self, action):
        """
        Takes an action and simulates the next step in the environment.
        Returns the next state, reward, done flag, and info dictionary.
        """
        # --- FIX 3: Combine current state and action for prediction ---
        # The feature-engineered state is already stored in self.current_state
        state_with_action = np.concatenate((self.current_state.reshape(1, -1), np.array(action).reshape(1, -1)), axis=1)
        
        # Scale the combined vector
        scaled_state = self.X_scaler.transform(state_with_action)
        dmatrix = xgb.DMatrix(scaled_state)

        # Predict the next day's raw OHLCV values
        pred_close = self.model_close.predict(dmatrix)
        pred_open = self.model_open.predict(dmatrix)
        pred_high = self.model_high.predict(dmatrix)
        pred_low = self.model_low.predict(dmatrix)
        pred_volume = self.model_volume.predict(dmatrix)

        # Inverse transform the predictions to their original scale
        next_close = self.y1_scaler.inverse_transform(pred_close.reshape(1, -1))[0][0]
        next_open = self.y2_scaler.inverse_transform(pred_open.reshape(1, -1))[0][0]
        next_high = self.y3_scaler.inverse_transform(pred_high.reshape(1, -1))[0][0]
        next_low = self.y4_scaler.inverse_transform(pred_low.reshape(1, -1))[0][0]
        next_volume = self.y5_scaler.inverse_transform(pred_volume.reshape(1, -1))[0][0]
        
        # --- FIX 4: Correctly update the state for the next step ---
        # Create a new row for the predicted day
        last_timestamp = self.full_data.index[-1]
        next_timestamp = last_timestamp + pd.Timedelta(days=1)
        
        new_row = pd.DataFrame({
            'Open': [next_open], 'High': [next_high], 'Low': [next_low], 
            'Close': [next_close], 'Volume': [next_volume]
        }, index=[next_timestamp])
        
    
        raw_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        updated_data = pd.concat([self.full_data[raw_cols], new_row])

       
        updated_features = self._calculate_features(updated_data)
        
      
        next_state = updated_features.iloc[-1].values
        self.current_state = next_state
        self.full_data = updated_features # Update our history

        # --- FIX 5: Calculate reward based on the action ---
        # A simple reward: the change in closing price.
        # This can be made more sophisticated (e.g., factor in action).
        previous_close = self.full_data.iloc[-2]['Close']
        reward = next_close - previous_close

        # For a simulation, 'done' is usually always False
        done = False
        info = {}

        return next_state, reward, done, info
    def _calculate_features(self,value):
       
        value['SMA']=talib.SMA(value['Close'],timeperiod=14)
        value['EMA']=talib.EMA(value['Close'],timeperiod=14)
        value['upper_band'], value['middle_band'], value['lower_band'] = talib.BBANDS(value['Close'], timeperiod=14)
        value['slowk'], value['slowd'] = talib.STOCH(
            value['High'], 
            value['Low'], 
            value['Close'], 
            fastk_period=14, 
            slowk_period=3, 
            slowk_matype=0, 
            slowd_period=3, 
            slowd_matype=0
        )
        value['RSI']=talib.RSI(value['Close'],timeperiod=14)
        value['MACD'], value['MACD_signal'], value['MACD_hist'] = talib.MACD(value['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        value['ADX']=talib.ADX(value['High'], value['Low'], value['Close'], timeperiod=14)
        value['CCI']=talib.CCI(value['High'], value['Low'], value['Close'], timeperiod=14)
        value['Williams %R']=talib.WILLR(value['High'], value['Low'], value['Close'], timeperiod=14)
        value['ATR']=talib.ATR(value['High'], value['Low'], value['Close'], timeperiod=14)
        value['OBV']=talib.OBV(value['Close'], value['Volume'])
        value['MFI']=talib.MFI(value['High'], value['Low'], value['Close'], value['Volume'], timeperiod=14)
        value['SAR']=talib.SAR(value['High'], value['Low'], acceleration=0, maximum=0)
        value['TRIX']=talib.TRIX(value['Close'], timeperiod=30)
        value['ULTOSC']=talib.ULTOSC(value['High'], value['Low'], value['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
        value['PPO'] = talib.PPO(value['Close'], fastperiod=12, slowperiod=26, matype=0)
        value['KAMA']=talib.KAMA(value['Close'], timeperiod=30)
        # DPO is not available in talib, so we implement it manually
        dpo_period = 20
        dpo_shift = int(dpo_period / 2 + 1)
        dpo_ma = value['Close'].rolling(window=dpo_period, min_periods=0).mean()
        value['DPO'] = value['Close'] - dpo_ma.shift(dpo_shift)
        value['MOM']=talib.MOM(value['Close'], timeperiod=10)
        value['BOP']=talib.BOP(value['Open'], value['High'], value['Low'], value['Close'])
        value['HT_TRENDLINE']=talib.HT_TRENDLINE(value['Close'])
        value['HT_DCPERIOD']=talib.HT_DCPERIOD(value['Close'])
        value['HT_DCPHASE']=talib.HT_DCPHASE(value['Close'])
        value['HT_PHASOR_INPHASE'], value['HT_PHASOR_QUADRATURE'] = talib.HT_PHASOR(value['Close'])
        value['HT_SINE_SINE'], value['HT_SINE_LEADSINE'] = talib.HT_SINE(value['Close'])
        value['HT_TRENDMODE']=talib.HT_TRENDMODE(value['Close'])
        value["SMA_50"]=talib.SMA(value['Close'],timeperiod=50)
        
        
        return value
    