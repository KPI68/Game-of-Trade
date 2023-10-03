import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import alpaca_trade_api as tradeapi
from trade import Trade
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN

def market_data( start, end, tickers ):
    load_dotenv()
    alpaca_api_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')

    # Create the Alpaca API object
    alpaca_api = tradeapi.REST(
       alpaca_api_key,
       alpaca_secret_key,
       api_version = 'v2'
    )
    
    # Set timeframe to "1Day" for Alpaca API
    timeframe = "1Day"
    
    start_date = pd.Timestamp(start,tz='America/New_York')
    end_date = pd.Timestamp(end,tz='America/New_York')
           
    # Get number_of_years' worth of historical data for tickers
    data_df = alpaca_api.get_bars(
        tickers,
        timeframe,
        start = start_date.isoformat(),
        end = end_date.isoformat()
    ).df
    if len(data_df) == 0:
        return []
    return data_df

class Test:
    def __init__ ( self, tickers=['AAPL','AMZN','MSFT','GOOG'], days=15 ):
        today = datetime.today().date()
        date_from = today - timedelta(days=365*3)
        self.data_df = market_data(date_from, today, tickers)
        self.price_df = pd.DataFrame()

        self.dfs = {}
        for ticker in tickers:
            df = data_df[data_df['symbol']==ticker]
            self.dfs[ticker] = df
            self.price_df = pd.concat([self.price_df, 
                                  pd.DataFrame(df.loc[:,['open','high','low','close']].mean(axis=1),
                                               index=df.index)],
                                 axis=1,
                                 join='outer')
        self.price_df.columns = tickers
        self.days = days