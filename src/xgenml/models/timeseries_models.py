# /src/xgenml/models/timeseries_models.py
"""
시계열 전용 모델들의 scikit-learn 호환 래퍼
"""
import numpy as np
import warnings


class ARIMAWrapper:
    """
    ARIMA 모델을 scikit-learn 스타일로 래핑
    requires: pip install statsmodels
    """
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None
        self._last_values = None
    
    def fit(self, X, y):
        """학습 - y만 사용 (univariate)"""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError(
                "ARIMA requires statsmodels. Install: pip install statsmodels"
            )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fit = self.model.fit(disp=False)
        
        self._last_values = y
        return self
    
    def predict(self, X):
        """예측"""
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_periods = len(X)
        forecast = self.model_fit.forecast(steps=n_periods)
        return np.array(forecast)


class ProphetWrapper:
    """
    Facebook Prophet을 scikit-learn 스타일로 래핑
    requires: pip install prophet
    """
    
    def __init__(self, seasonality_mode='multiplicative', **prophet_kwargs):
        self.seasonality_mode = seasonality_mode
        self.prophet_kwargs = prophet_kwargs
        self.model = None
        self._start_date = None
    
    def fit(self, X, y):
        """학습"""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "Prophet requires prophet. Install: pip install prophet"
            )
        
        import pandas as pd
        
        # Prophet은 'ds'(날짜)와 'y'(값) 필요
        df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(y), freq='D'),
            'y': y
        })
        
        self._start_date = df['ds'].iloc[-1]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                **self.prophet_kwargs
            )
            self.model.fit(df)
        
        return self
    
    def predict(self, X):
        """예측"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        import pandas as pd
        
        # 미래 기간 생성
        future = pd.DataFrame({
            'ds': pd.date_range(
                start=self._start_date + pd.Timedelta(days=1),
                periods=len(X),
                freq='D'
            )
        })
        
        forecast = self.model.predict(future)
        return forecast['yhat'].values