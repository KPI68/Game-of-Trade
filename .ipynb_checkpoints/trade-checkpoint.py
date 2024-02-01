import numpy as np
import pandas as pd
import hvplot.pandas
from dotenv import load_dotenv
import os
import alpaca_trade_api as tradeapi
import math
import numpy as np
from fredapi import Fred
from scipy.optimize import minimize
from datetime import datetime, timedelta

prices = {}
market_prices = {}
market_data = pd.DataFrame()

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix.T @ weights
    return np.sqrt(variance)
    
def expected_returns(weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

def sharpe_ratio(weights, log_returns,cov_matrix,risk_free_rate):
    return (expected_returns(weights,log_returns)-risk_free_rate) / standard_deviation(weights,cov_matrix)

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

def get_optimal_weights():
        log_returns = np.log(markect_data / market_data.shift(1)).dropna()
        cov_matrix = log_returns.cov() * 252

        fred_api_key = os.getenv('FRED_STLOUIS_API_KEY')
        fred = Fred(api_key=fred_api_key)
        ten_year_treasurey_rate = fred.get_series_latest_release('GS10')/100
        risk_free_rate = ten_year_treasurey_rate.iloc[-1]
        
        constraints = {'type':'eq','fun':lambda weights:np.sum(weights)-1}
        bounds = [(0,0.5) for _ in range(len(self.tickers))]
        
        initial_weights = np.array([1/len(self.tickers)]*len(self.tickers))
        
        #optimize the weights to maximize sharpe ratio
        optimized_results = minimize(neg_sharpe_ratio, initial_weights, 
                                     args=(log_returns, cov_matrix, risk_free_rate), 
                                     method='SLSQP', constraints=constraints, bounds=bounds)
        optimal_weights = optimized_results.x
        return optimal_weights
    

class Trade:
    def __init__(self, instr, start_amount=10000.00, verbose=1, data=None):
        # instr is a df - 0: date, 1: ticker, 2: action
        # data is a df - date as index, columns of tickers' avg
        
        self.start_amount = start_amount
        
        number_of_tickers = len(set(instr.iloc[:,1]))
        ticker_weights = [1/number_of_tickers] * number_of_tickers
        ticker_names = set(instr.iloc[:,1])
        self.ticker_weights = dict(zip(ticker_names, ticker_weights))
        
        self.verbose = verbose
        self.price_from = 'alpaca'
        
        if ( data != None and
            set(ticker_names) == set(data.columns) and
            instr.iloc[-1:0] <= data.index[-1] ):
            self.price_from = 'data'
            market_data = data
            self.ticker_weights = dict(zip(data.columns, get_optimal_weights()))
        
        else: 
            load_dotenv()
            alpaca_api_key = os.getenv('ALPACA_API_KEY')
            alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')

            # Create the Alpaca API object
            self.alpaca_api = tradeapi.REST(
               alpaca_api_key,
               alpaca_secret_key,
               api_version = 'v2'
            )

            # Set timeframe to "1Day" for Alpaca API
            self.timeframe = "1Day"

    """ If to buy, we can only use start_amount/number_of_tickers at a time, or spend all
        If to sell, sell all shares
    """
    def spend_how_much( self, bal=0, price=0, ticker=None ):
        ticker_portion = bal * self.ticker_weights[ticker]
        if bal > ticker_portion:
            return ticker_portion
        elif bal/price >= 1: 
            return bal
        else:
            return 0
        
    def trade( self, on=pd.to_datetime('today').normalize(), do=0, ticker="", b_bal=0, shares=0 ):
        self.last_action_ticker = ticker
        status = 0
        a_bal = b_bal
        msg = 'success'
        action = 'none'
    
        if len(ticker) == 0:
            status = -1
            msg = 'no ticker'
        else:
            price = self.market_price(on, ticker)
            spend = self.spend_how_much( b_bal, price, ticker )
            if price < 0:
                status = -1
                msg = 'no market price'
            elif do == 1 and spend == 0:
                state = -1
                msg = 'no money to buy'
            elif do == 1 :
                status = 0
                action = 'buy'
                shares = math.floor(spend/price)
                a_bal = round(a_bal - shares * price,2)
            elif do == -1 and shares == 0:
                status = -1
                msg = 'no share to sell'
            elif do == -1:
                status = 0
                action = 'sell'
                a_bal = round(a_bal + shares * price,2)
            else:
                status = -1
                msg = 'no action'
                 
        return { 'action' : action,
                 'price' : price,
                 'bal' : a_bal,
                 'share' : shares,
                 'status' : status,
                 'msg' : msg }

    def market_price( self, on, ticker ):
        #print(market_prices)
        if self.price_from == 'data':
            return market_data.loc[on, ticker]
        
        try:
            prices = market_prices[on]
            try:
                price = prices[ticker]
            except KeyError:
                price = self.get_market_price( on, ticker )
                prices[ticker] = price
            
        except KeyError:    
            price = self.get_market_price( on, ticker )
            prices = {}
            prices[ticker] = price
            market_prices[on] = prices
        
        return price
    
    def get_market_price( self, on, ticker ):
        start_end = pd.Timestamp(on, tz='America/New_York')
        data_df = self.alpaca_api.get_bars(
                [ticker],
                self.timeframe,
                start = start_end.isoformat(),
                end = start_end.isoformat()
            ).df
            
        if len(data_df) == 0:
            return -1
            
        return round( data_df.iloc[-1][['high','low','open','close']].sum()/4,2 )
            

    def trade_action( self ):
        bal = self.start_amount
        shares = {}      
        perf = pd.DataFrame()

        for row in instr.iterrows():
            one_trade = row[1]
            try:
                share = shares[one_trade[1]]
            except KeyError:
                share = 0

            traded = trade.trade(one_trade[0], 
                                 do=one_trade[2], 
                                 ticker=one_trade[1], 
                                 b_bal=bal,
                                 shares=share
                                ) 
            if verbose == 1:
                print(f"On {one_trade[0]} trade {one_trade[1]}")
                print(traded)

            if traded['status'] == 0:
                bal = traded['bal']
                try:
                    if traded['action'] == 'buy':
                        shares[one_trade[1]] += traded['share']
                    if traded['action'] == 'sell':
                        shares[one_trade[1]] -= traded['share']
                except KeyError:
                    shares[one_trade[1]] = traded['share']

            share_worth = 0
            perf_one_row = pd.DataFrame([[one_trade[0], bal]])
            for key in shares:
                share_worth += round(shares[key] * traded['price'],2)
                perf_one_row[key] = shares[key]
            perf_one_row['networth'] = traded['bal'] + share_worth
            perf = pd.concat([perf, perf_one_row], join='outer')

        final_worth = round(traded['bal'] + share_worth,2)
        if verbose == 1:
            print(f"""networth: Cash: {traded['bal']}, 
                      shares: {shares}, 
                      total: {final_worth}""")

        if verbose == 0:
            return final_worth

        perf = perf.rename(columns = { 0:'date', 1:'cash' })
        perf = perf.set_index('date')
        share_plot = (perf.drop(['cash','networth'], axis=1)).hvplot(shared_axes=False, 
                                                                     frame_width=325,
                                                                     ylabel='shares',
                                                                     title='shares in trading'
                                                                    )
        money_plot = perf[['cash','networth']].hvplot(shared_axes=False, 
                                                      frame_width=325,
                                                      ylabel='dollars',
                                                      title='networth and cash'
                                                     )
        if verbose == 1:
            return share_plot + money_plot 

        def market_data( start, end, tickers ):
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
        
class Portfolio:
    
    def __init__(self,
                 number_of_years = 5,
                 tickers = None
                ):
        """
        Provide number_of_years lookback for historical data
        Get historical data for number_of_years till today
        """
        # load env
        load_dotenv()
    
        self.number_of_years = number_of_years
        self.weights = []

        # get portfolio historical data    
        
        # Load Alpaca keys required by the APIs
        alpaca_api_key = os.getenv('ALPACA_API_KEY')
        alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')

        # Create the Alpaca API object
        alpaca_api = tradeapi.REST(
           alpaca_api_key,
           alpaca_secret_key,
           api_version = 'v2'
        )
    
        self.alpaca_api = alpaca_api
        
        # set the start,end date - as pd.timestamp
        today_date = pd.to_datetime('today').normalize()   
        today = pd.Timestamp(today_date,tz='America/New_York')
        start = today - pd.Timedelta(365*number_of_years, 'd')
    
        # prep for Yahoo - as datetime.datetime.timestamp
        end_date = datetime.today()
        start_date = end_date - timedelta(days = number_of_years*365)

        self.alpaca_from = start
        self.alpaca_to = today
        self.yahoo_from = start_date
        self.yahoo_to = end_date
        
        if tickers != None: 
            self.tickers = tickers
            self.data = self.get_data_alpaca( tickers ) 
            if len(self.data) == 0:
                self.data = self.get_data_yahoo( tickers )
       
        else:
            self.tickers = []
            self.data = pd.DataFrame()
            
    def gen_port(self):
        
        if len(self.data) == 0:
            self.data = None
            self.weights = None
            self.daily_return = None
            return None

        self.weights = self.get_optimal_weights()
        
        # calculate daily return of tickers
        daily_return = self.data.pct_change().fillna(0).dot(self.weights)
        
        # calculate and store weighted daily return of the portfolio
        self.daily_return = pd.DataFrame( { 'portfolio': daily_return })
        
        return True
