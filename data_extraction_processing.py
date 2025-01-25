import json
import yfinance as yf
import pandas as pd
import os
import datetime
from dateutil.relativedelta import relativedelta

def retrieve_historical_returns(ETFInfo_path):
    """
    Retrieves return data from the Yahoo Finance API for a given set of ETFs/stocks.

    Parameters:
    ----------
    ETFInfo_path : str
        Path to the JSON file containing ETF information with the following structure:
        {
            "tickers": ["XLC", "XLY", "XLP"],
            "dataParameters" : {
                "sample_time_step" : "1mo",
                "total_sample_period" : "5yr",
                "sample_period_end" : "2024-01-01"
            }
        }
        sample_time_step : dictates the period at which each dataframe is stepped
        total_sample_period : total period to be sampled, must be a multiple of sample_time_step
        sample_period_end : time from which the total_sample_period will step back from

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the monthly percent returns of each ticker.

    Raises:
    ------
    FileNotFoundError:
        If the ETFInfo.json file is not found.
    ValueError:
        If the JSON file contains errors or has invalid date formats.
    KeyError:
        If required keys are missing in the JSON file.
    """
    
    # 1. Ensure the ETFInfo.json file exists before processing
    if not os.path.exists(ETFInfo_path):
        raise FileNotFoundError(f"Error: The file '{ETFInfo_path}' does not exist.")
    
    # 2. Read and parse the JSON file
    with open(ETFInfo_path, 'r') as file:
        try:
            ETFInfo = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file: {e}")

    # 3. Validate presence and correctness of essential data fields
    if "tickers" not in ETFInfo or not isinstance(ETFInfo["tickers"], list):
        raise KeyError("Missing or invalid 'tickers' in JSON file.")
    
    if "dataParameters" not in ETFInfo:
        raise KeyError("Missing 'dataParameters' in JSON file.")
    
    tickers = ETFInfo["tickers"]
    data_params = ETFInfo["dataParameters"]

    # 4. Ensure required keys exist in the dataParameters section
    required_keys = ["sample_time_step", "total_sample_period", "sample_period_end"]
    for key in required_keys:
        if key not in data_params:
            raise KeyError(f"Missing '{key}' in dataParameters.")    

    # 5. Validate the date format to match 'YYYY-MM-DD'
    date_format = "%Y-%m-%d"
    try:
        datetime.datetime.strptime(data_params["sample_period_end"], date_format)
    except ValueError:
        raise ValueError("Invalid date format in sample_period_end, expected YYYY-MM-DD.")
    
    # 6. Validate the time interval format to match '1mo', '1yr', '1d' etc.
    accepted_interval_values = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y']
    for key, value in data_params.items():
        if key != "sample_period_end" and value not in accepted_interval_values:
            raise ValueError("Invalid date format in {key}, accepted values include: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'.")

    # 7. Ensure that sample period data exists for each ETF
        # NOTICE: Assumes total_sample_period will be years, much simpler to implement, trust me this is somehow the best way to do this lmfao
    
    sample_period_end = data_params["sample_period_end"]
    total_sample_period_years = int((data_params["total_sample_period"])[:-1]) # Deletes the 'y' unit from str
    sample_period_start = f'{int(sample_period_end[:4]) - total_sample_period_years}{sample_period_end[4:]}'

    etf_start_years = ([
        yf.download(ticker, start=sample_period_start, end=sample_period_end).index.min().year
        for ticker in tickers
    ]) # Grabs start years for each ETF, upon start period
    
    if max(etf_start_years) != min(etf_start_years) and max(etf_start_years) != int(sample_period_start[:4]):
        bad_etfs = []
        for i in range(len(etf_start_years)):
            if etf_start_years[i] != int(sample_period_start[:4]):
                bad_etfs.append(tickers[i])
        raise ValueError(f"Historical data during specified sample period does not exist for {bad_etfs}.")
    
    # 8. Download and process adjusted close price data for all tickers
    formatted_tickers = " ".join(tickers)
    data = yf.download(formatted_tickers, start=sample_period_start, end=sample_period_end)["Adj Close"]

    # 9. Resample the data to yearly frequency and calculate percentage change
    returns = data.resample('ME').last().pct_change()

    # 10. Handle missing values by filling NaNs with 0 to prevent calculation errors
    returns = returns.fillna(0)

    return returns


def construct_key_metrics(returns):
    """
    Utilizing monthly return data returns the calculated covariance between each ETF and the volatility of each.

    Parameters:
    ----------
    returns : pd dataframe
        Dataframe from consisting of monthly returns over a certain period for a select number of tickers

    Returns:
    -------
    list of pandas.DataFrame
        [
        pandas.DataFrame : Orignial monthly returns for each ETF
        pandas.DataFrame : Volatility (STD) of each ETF caclulated over entire period
        pandas.DataFrame : Covariance matrix (11x11) of each ETF calculated over entire period
        ]
    Raises:
    ------
    RuntimeError
        if errors when calculating pandas values (covarience matrix or volatility)
    """

    # 1. Calculate covariance matrix
    try:
        covariance_matrix = returns.cov()
    except:
        raise RuntimeError('Error calculating covariance matrix. Returns matrix of shape: {returns.shape}')

    # 2. Calculate volatilities
    try:
        volatilities = returns.std()
    except:
        raise RuntimeError('Error calculating volatilities. Returns matrix of shape: {returns.shape}')

    # 3. Return list of dataframes
    return [returns, volatilities, covariance_matrix]

def backtest_resample(returns, duration_months):
    """
    Calculates construct_key_metrics() from a limited timeframe, utilized for backtesting. 

    Parameters:
    ----------
    returns : pd dataframe
        Dataframe from consisting of monthly returns over a certain period for a select number of tickers 
            Must be the full initial dataframe from retrieve_historical_returns()
    duration_months : int
        An integer representing how many months from the initial period to calculate key metrics upon
            Must not exceed the total duration of the returns dataframe

    Returns:
    -------
    list of pandas.DataFrame
        [
        pandas.DataFrame : Shortened monthly returns for each ETF within the duration period
        pandas.DataFrame : Volatility (STD) of each ETF caclulated over the duration period
        pandas.DataFrame : Covariance matrix (11x11) of each ETF calculated over the duration period
        ]
    Raises:
        ValueError : if duration_months is not a positive integer
                     if duration_months exceeds the total duration of the returns dataframe

    ------
    """
    # 1. Check duration_months is a positive integer
    if duration_months <= 0 or type(duration_months) != int:
        raise ValueError(f"Duration must be a positive integer. Currently of value {duration_months} and type {type(duration_months)}")

    # 1. Find the nearest date in data after adding the date_duration
    start_date = returns.index[0]
    full_dates = (returns.index).tolist()
    end_date = (start_date + relativedelta(months=duration_months))

    # 2. Check that duration_months does not exceed total returns duration
    if returns.index[-1] < end_date and duration_months != len((returns.index).tolist()):
        raise ValueError("Duration exceeds the length of the returns DataFrame.")

    closest_date = min(full_dates, key=lambda x: abs(x - end_date))   
                
    # 2. Reduce initial returns to end at end_date (following specified duration from start)
    durated_returns = returns.loc[start_date:closest_date]

    # 3. Calcualte key metrics on durated returns
    durated_metrics = construct_key_metrics(durated_returns)

    return durated_metrics

# Example usage of the function
if __name__ == "__main__":
    try:
        # Retrieve returns data
        df_returns = retrieve_historical_returns("ETF_data_info.json")
        print(df_returns.head())  # Display first few rows of the returns DataFrame

        key_metrics = construct_key_metrics(df_returns)
        for dataframe in key_metrics[2:]:
            print(dataframe.head()) #Display first few rows of the key metrics DataFrames
        
        backtest_key_metrics = backtest_resample(df_returns, duration_months=5)
        for dataframe in key_metrics:
            print(dataframe.head()) #Display first few rows of the key metrics DataFrames, along with the durated returns

    except Exception as e:
        print(f"Error: {e}")
