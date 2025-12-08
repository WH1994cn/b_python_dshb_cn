import pandas as pd
import numpy as np

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    original input data are extracted, converted into operable format.
    Output: return dataframe incl. small cap and large cap monthly returns. 
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME.csv",
                       header=7, index_col=0, nrows=1192, parse_dates=True, na_values=-99.99,engine='python')
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets.apply(lambda s: pd.to_numeric(s, errors='coerce'))
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    URL: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#Research
    """
    ind = pd.read_csv("data/30_Industry_Portfolios.csv", header=7, index_col=0, skipfooter=1, 
                      parse_dates= True, na_values=-99.99)
    ind=ind.apply(lambda s: pd.to_numeric(s, errors='coerce'))
    ind = ind/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def drawdown(return_series: pd.Series):
    """Input: Takes a time series of asset returns.
       Output: returns a DataFrame with columns for
       the wealth index: how capital with start cap of 1000 changes with time, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})



def semideviation(r):
    """
    Input: Takes a time series of asset returns r.
    Returns the semideviation aka negative semideviation of returns r.
    r must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
    

def var_historic(r, level=5):
    """
    Calculates the historic Value at Risk (VaR) at a specified confidence level.
    
    Returns the return threshold such that 'level' percent of returns fall below it,
    representing the worst-case loss at the given confidence level based on historical data.
    
    Parameters
    ----------
    r : pd.Series or pd.DataFrame, time series of periodic returns
    level : int or float, optional (default=5)
        Percentile level for VaR calculation (e.g., 5 for 5th percentile)
        Represents the probability of losses exceeding the VaR threshold
    
    Returns
    -------
    float or pd.Series
        Historic VaR as a positive number (e.g., 0.05 means 5% potential loss)
        Returns a Series if input is a DataFrame with multiple columns
    
    Notes
    -----
    - Uses numpy.percentile to find the historical return at the specified percentile
    - Returns the absolute value (negative sign removed) for easier interpretation
    - VaR at 5% means there's a 5% probability of losses exceeding this value
    - This is a non-parametric approach that doesn't assume any distribution
    """
    try: 
        if isinstance(r, pd.DataFrame):
            return r.aggregate(var_historic, level=level)
        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Expected retrun data to be a Series or DataFrame")
    except TypeError as e:
        raise TypeError(f"Invalid input type: {type(r).__name__}. Expected pandas Series or DataFrame") from e
    except Exception as e:
        raise ValueError(f"Error calculating VaR: {str(e)}") from e
        

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns using compound growth formula.
    
    Converts periodic returns (e.g., monthly, daily) to an equivalent annualized return
    by calculating the compound growth over the period and scaling it to annual terms.
    
    Parameters
    ----------
    r : pd.Series or pd.DataFrame. Time series of periodic returns (e.g., monthly returns as decimals)
    periods_per_year : int, Number of periods in a year (e.g., 12 for monthly, 252 for daily trading days)
    
    Returns
    -------
    float or pd.Series
        Annualized return as a decimal (e.g., 0.15 for 15% annual return)
        Returns a Series if input is a DataFrame with multiple columns
    
    Notes
    -----
    Formula: (1 + r)^(periods_per_year/n) - 1
    where compounded_growth = (1 + r).prod() and n = number of observations
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the volatility (standard deviation) of a set of returns.
    
    Converts periodic volatility to annualized volatility using the square root of time rule,
    which assumes returns are independent and identically distributed (i.i.d.).
    
    Parameters
    ----------
    r : pd.Series or pd.DataFrame, time series of periodic returns (e.g., monthly returns as decimals)
    periods_per_year : int, number of periods in a year (12 for monthly, 252 for daily trading days)
    
    Returns
    -------
    float or pd.Series
        Annualized volatility as a decimal (e.g., 0.20 for 20% annualized volatility)
        Returns a Series if input is a DataFrame with multiple columns
    
    Notes
    -----
    Formula: σ_annual = σ_periodic × √(periods_per_year)
    This is based on the property that variance scales linearly with time,
    so standard deviation scales with the square root of time.
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized Sharpe ratio of a set of returns.
    
    The Sharpe ratio measures risk-adjusted return by comparing excess returns
    (returns above the risk-free rate) to the volatility of those returns.
    Higher values indicate better risk-adjusted performance.
    
    Parameters
    ----------
    r : pd.Series or pd.DataFrame, time series of periodic returns
    riskfree_rate : float, annual risk-free rate (as decimal, e.g., 0.03 for 3%)
    periods_per_year : int, number of periods in a year (e.g., 12 for monthly, 252 for daily trading days)
    
    Returns
    -------
    float or pd.Series
        Annualized Sharpe ratio (unitless)
        Returns a Series if input is a DataFrame with multiple columns
    
    Notes
    -----
    Formula: Sharpe Ratio = (Annualized Excess Return) / (Annualized Volatility)
    - Excess return = portfolio return - risk-free rate
    - Risk-free rate is converted from annual to per-period rate
    - Values > 1 are generally considered good, > 2 are very good, > 3 are excellent
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


def portfolio_return(weights, returns):
    """
    Computes the weighted return of a portfolio.
    
    Calculates the total portfolio return by multiplying each asset's weight
    by its return and summing the results.
    
    Parameters
    ----------
    weights : np.ndarray or Nx1 matrix, portfolio weights for each asset (must sum to 1.0)
    returns : np.ndarray or Nx1 matrix, returns for each asset (as decimals)
    
    Returns
    -------
    float
        Portfolio return (as decimal)
    
    Notes
    -----
    Formula: r_portfolio = w^T × r
    where w is the weight vector and r is the returns vector
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the volatility (standard deviation) of a portfolio.
    
    Calculates portfolio volatility using the covariance matrix of asset returns
    and portfolio weights, accounting for correlations between assets.
    
    Parameters
    ----------
    weights : np.ndarray or Nx1 matrix, portfolio weights for each asset (must sum to 1.0)
    covmat : np.ndarray or NxN matrix, covariance matrix of asset returns
    
    Returns
    -------
    float
        Portfolio volatility/standard deviation (as decimal)
    
    Notes
    -----
    Formula: σ_portfolio = √(w^T × Σ × w)
    where w is the weight vector and Σ is the covariance matrix
    This accounts for the variance of individual assets and their correlations
    """
    return (weights.T @ covmat @ weights)**0.5




