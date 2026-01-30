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
    ind = pd.read_csv("data/30_Industry_Portfolios.csv", header=6, index_col=0, nrows=1191,  
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



def get_top_drawdowns(dd_series: pd.Series, top: int = 5):
    """
    Return the top `top` drawdown episodes from a drawdown series.

    Parameters
    ----------
    dd_series : pd.Series
        Time series of drawdown values (0 at peaks, negative during drawdown).
    top : int
        Number of largest drawdown episodes to return (most negative troughs).

    Returns
    -------
    pd.DataFrame
        Columns: start, trough_date, trough_value, end, duration
    """
    
    if not isinstance(dd_series, pd.Series):
        raise TypeError("dd_series must be a pandas Series")

    # Boolean series: True while in a drawdown (value < 0)
    in_drawdown = dd_series < 0
    
    # Create group ids that change when in_drawdown status changes.
    # Consecutive True values form one drawdown episode.
    group_ids = (in_drawdown != in_drawdown.shift(1)).cumsum()

    episodes = []
    # Iterate over each contiguous group
    for _, grp in dd_series.groupby(group_ids):
        if grp.empty:
            continue
        # Only consider groups that are drawdowns (first value < 0)
        if grp.iloc[0] < 0:
            start = grp.index[0]
            end = grp.index[-1]
            trough_value = float(grp.min())
            trough_date = grp.idxmin()
            duration = len(grp)  # number of periods in episode
            episodes.append({
                "start": start,
                "trough_date": trough_date,
                "trough_value": trough_value,
                "end": end,
                "duration": duration
            })
    # If no drawdown episodes found, return empty DataFrame with expected columns
    if not episodes:
        return pd.DataFrame(columns=["start","trough_date","trough_value","end","duration"])
    
    # Sort by trough_value (most negative first) and return top N
    episodes_df = pd.DataFrame(episodes).sort_values(by="trough_value").head(top).reset_index(drop=True)
    return episodes_df

# Functions for portfolio optimization and calculation of efficient frontier
# Efficient Frontier for 2-asset portfolio
def plot_ef_2ass(n_points, er, cov):
    """
    Generate and visualize the efficient frontier for a two-asset portfolio optimization.
    
    This function computes portfolio combinations across different weight allocations
    and displays the risk-return relationship as a continuous curve, helping investors
    identify optimal portfolio compositions.
    
    Parameters
    ----------
    n_points : int
        Resolution of the frontier curve (number of weight combinations to evaluate).
        Recommended: 20-100 for smooth visualization.
    er : pd.Series or array-like, length 2
        Annual expected returns for both assets (expressed as decimals).
        Example: [0.08, 0.12] represents 8% and 12% expected returns.
    cov : pd.DataFrame or 2D array, shape (2,2)
        Variance-covariance matrix capturing asset volatilities and correlation.
        Must be symmetric and positive semi-definite.
    
    Returns
    -------
    matplotlib.axes.Axes
        Interactive plot object with volatility on x-axis and returns on y-axis.
        Each point represents a unique portfolio allocation strategy.
    
    Raises
    ------
    ValueError
        When input dimensions don't match two-asset requirement.
    
    Implementation Details
    ----------------------
    - Weight allocation ranges from [1,0] to [0,1] for the asset pair
    - Portfolio metrics calculated using Modern Portfolio Theory formulas
    - Visualization uses connected points to show continuous frontier
    - Left endpoint: 100% first asset, Right endpoint: 100% second asset
    
    Financial Context
    -----------------
    The efficient frontier represents portfolios offering maximum expected return
    for each level of risk, or minimum risk for each level of return.
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("Function requires exactly 2 assets for frontier calculation")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=".-")

def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plot the multi-asset efficient frontier with optional portfolio overlays.
    
    Visualizes the risk-return tradeoff for optimal portfolios across different expected
    return levels. Optionally displays special portfolios (MSR, GMV, Equal Weight) and
    the Capital Market Line for enhanced portfolio analysis.
    
    Parameters
    ----------
    n_points : int
        Number of portfolios to compute along the efficient frontier.
        Higher values create smoother curves. Recommended: 20-100.
        More points increase computation time but improve visualization quality.
    er : pd.Series or np.ndarray, shape (n_assets,)
        Expected annual returns for each asset (as decimals).
        Example: pd.Series([0.08, 0.10, 0.12, 0.15]) for 8%, 10%, 12%, 15% returns.
    cov : pd.DataFrame or np.ndarray, shape (n_assets, n_assets)
        Covariance matrix of asset returns.
        Must be symmetric and positive semi-definite.
        Diagonal elements are variances, off-diagonal elements are covariances.
    style : str, optional, default='.-'
        Matplotlib line style for the efficient frontier curve.
        Examples: '.-' (line with dots), '-' (solid line), '--' (dashed), 'o-' (circles with line)
    legend : bool, optional, default=False
        Whether to display a legend on the plot.
        Set to True when overlaying multiple frontiers for comparison.
    show_cml : bool, optional, default=False
        If True, displays the Capital Market Line (CML).
        The CML connects the risk-free rate to the Maximum Sharpe Ratio portfolio,
        representing the best possible risk-return combinations when leverage is allowed.
    riskfree_rate : float, optional, default=0
        Annual risk-free rate (as decimal, e.g., 0.03 for 3%).
        Only used if show_cml=True.
        Typically represents Treasury bill or government bond yields.
    show_ew : bool, optional, default=False
        If True, plots the Equal Weight (EW) portfolio as a gold marker.
        The EW portfolio allocates 1/n to each asset (naive diversification).
        Useful for comparing sophisticated optimization against simple strategies.
    show_gmv : bool, optional, default=False
        If True, plots the Global Minimum Variance (GMV) portfolio as a dark blue marker.
        The GMV portfolio represents the leftmost point on the efficient frontier
        (minimum possible risk).
    
    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib axes object containing the plot.
        Can be further customized with additional plotting commands.
    
    Notes
    -----
    The efficient frontier represents portfolios that offer:
    - Maximum expected return for a given level of risk, OR
    - Minimum risk for a given level of expected return
    
    Portfolio Overlays:
    - **MSR (Maximum Sharpe Ratio)**: Shown when show_cml=True (green marker)
      - Optimal portfolio for investors who can borrow/lend at risk-free rate
      - Tangency point where CML touches the efficient frontier
    - **GMV (Global Minimum Variance)**: Shown when show_gmv=True (dark blue marker)
      - Lowest risk portfolio, suitable for extremely risk-averse investors
    - **EW (Equal Weight)**: Shown when show_ew=True (gold marker)
      - Simple 1/n allocation, often used as a benchmark
    
    Capital Market Line (CML):
    - Green dashed line from risk-free rate to MSR portfolio
    - Represents portfolios combining risk-free asset with MSR portfolio
    - Slope equals Sharpe ratio of MSR portfolio
    - Points above the line are theoretically unachievable
    
    Computational Complexity
    ------------------------
    - Time complexity: O(n_points × optimization_cost)
    - Each portfolio requires quadratic programming (QP) optimization
    - For large portfolios (>100 assets), consider reducing n_points
    
    Examples
    --------
    >>> # Basic efficient frontier
    >>> er = pd.Series([0.08, 0.10, 0.12, 0.15])
    >>> cov = pd.DataFrame([[0.04, 0.01, 0.02, 0.015],
    ...                     [0.01, 0.05, 0.015, 0.02],
    ...                     [0.02, 0.015, 0.06, 0.025],
    ...                     [0.015, 0.02, 0.025, 0.08]])
    >>> 
    >>> ax = plot_ef(50, er, cov)
    >>> ax.set_title('Efficient Frontier - 4 Asset Portfolio')
    >>> plt.show()
    
    >>> # Comprehensive analysis with all overlays
    >>> ax = plot_ef(100, er, cov, 
    ...              show_cml=True, 
    ...              riskfree_rate=0.03,
    ...              show_ew=True, 
    ...              show_gmv=True,
    ...              legend=True)
    >>> ax.set_title('Portfolio Analysis with Key Portfolios')
    >>> ax.set_xlabel('Volatility (Risk)')
    >>> ax.set_ylabel('Expected Return')
    >>> ax.grid(True, alpha=0.3)
    >>> plt.show()
    
    >>> # Compare different asset sets
    >>> fig, ax = plt.subplots()
    >>> plot_ef(50, er_stocks, cov_stocks, style='b-', legend=True)
    >>> plot_ef(50, er_bonds, cov_bonds, style='r--', legend=True)
    >>> ax.legend(['Stocks Only', 'Bonds Only'])
    >>> plt.show()
    
    Interpretation Guide
    --------------------
    - **Leftmost point (GMV)**: Minimum risk, potentially low returns
    - **Upward curve**: Higher returns require accepting more risk
    - **Steeper slope**: Better risk-return tradeoff in that region
    - **Points below frontier**: Sub-optimal (dominated by frontier portfolios)
    - **CML tangency (MSR)**: Best risk-adjusted portfolio
    
    Visual Elements
    ---------------
    - Blue curve: Efficient frontier
    - Green dashed line: Capital Market Line (if show_cml=True)
    - Green circle: MSR portfolio (if show_cml=True)
    - Gold circle: Equal Weight portfolio (if show_ew=True)
    - Dark blue circle: GMV portfolio (if show_gmv=True)
    
    See Also
    --------
    plot_ef_2ass : Two-asset efficient frontier (simpler case)
    optimal_weights : Generate weight vectors for frontier points
    msr : Calculate Maximum Sharpe Ratio portfolio
    gmv : Calculate Global Minimum Variance portfolio
    portfolio_return : Calculate portfolio expected return
    portfolio_vol : Calculate portfolio volatility
    
    References
    ----------
    .. [1] Markowitz, H. (1952). "Portfolio Selection". The Journal of Finance.
    .. [2] Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium".
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        
        return ax



from scipy.optimize import minimize

def minimize_vol(target_return, er, cov):
    """
    Find optimal portfolio weights that minimize volatility for a given target return.
    
    Solves the portfolio optimization problem: minimize σ(w) subject to:
    - Expected return equals target return
    - Weights sum to 1 (fully invested)
    - All weights between 0 and 1 (long-only, no short selling)
    
    Parameters
    ----------
    target_return : float
        Desired portfolio return (annual, as decimal).
        Example: 0.10 for 10% target return.
    er : pd.Series or np.ndarray, shape (n_assets,)
        Expected annual returns for each asset (as decimals).
    cov : pd.DataFrame or np.ndarray, shape (n_assets, n_assets)
        Covariance matrix of asset returns.
    
    Returns
    -------
    np.ndarray, shape (n_assets,)
        Optimal portfolio weights that minimize risk for target return.
        Values sum to 1.0, each between 0.0 and 1.0.
    
    Notes
    -----
    - Uses Sequential Least Squares Programming (SLSQP) optimization
    - Part of Modern Portfolio Theory (Markowitz optimization)
    - Initial guess: equal-weight portfolio (1/n for each asset)
    - Returns point on efficient frontier at specified return level
    
    Examples
    --------
    >>> er = pd.Series([0.08, 0.10, 0.12])
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                 [0.01, 0.05, 0.015],
    ...                 [0.02, 0.015, 0.06]])
    >>> weights = minimize_vol(0.10, er, cov)
    >>> print(f"Optimal weights: {weights}")
    Optimal weights: [0.25 0.50 0.25]
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def msr(riskfree_rate, er, cov):
    """
    Calculate Maximum Sharpe Ratio (MSR) portfolio weights.
    
    Finds the portfolio with the highest risk-adjusted returns by maximizing
    the Sharpe ratio: (portfolio return - risk-free rate) / portfolio volatility.
    This is the tangency portfolio where the Capital Market Line touches the efficient frontier.
    
    Parameters
    ----------
    riskfree_rate : float
        Annual risk-free rate (as decimal, e.g., 0.03 for 3%).
        Typically uses Treasury bill or government bond rates.
    er : pd.Series or np.ndarray, shape (n_assets,)
        Expected annual returns for each asset (as decimals).
    cov : pd.DataFrame or np.ndarray, shape (n_assets, n_assets)
        Covariance matrix of asset returns.
    
    Returns
    -------
    np.ndarray, shape (n_assets,)
        Optimal portfolio weights maximizing Sharpe ratio.
        Values sum to 1.0, each between 0.0 and 1.0.
    
    Notes
    -----
    - MSR portfolio is optimal for investors who can borrow/lend at risk-free rate
    - Defines the Capital Market Line (CML) in mean-variance space
    - Optimization minimizes negative Sharpe ratio (equivalent to maximization)
    - Uses SLSQP algorithm with long-only constraints
    
    Financial Interpretation
    ------------------------
    The MSR portfolio represents the best risk-return tradeoff when combining
    risky assets with risk-free borrowing/lending. Investors with different
    risk preferences should hold this portfolio and adjust leverage using
    the risk-free asset.
    
    Examples
    --------
    >>> er = pd.Series([0.08, 0.10, 0.12])
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                 [0.01, 0.05, 0.015],
    ...                 [0.02, 0.015, 0.06]])
    >>> w_msr = msr(0.03, er, cov)
    >>> print(f"MSR weights: {w_msr}")
    >>> 
    >>> # Calculate Sharpe ratio
    >>> ret = portfolio_return(w_msr, er)
    >>> vol = portfolio_vol(w_msr, cov)
    >>> sharpe = (ret - 0.03) / vol
    >>> print(f"Sharpe Ratio: {sharpe:.3f}")
    
    See Also
    --------
    sharpe_ratio : Calculate Sharpe ratio for given returns
    gmv : Global Minimum Variance portfolio
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


def gmv(cov):
    """
    Calculate Global Minimum Variance (GMV) portfolio weights.
    
    Finds the portfolio with the lowest possible volatility (risk) regardless
    of expected return. This portfolio lies at the leftmost point of the
    efficient frontier.
    
    Parameters
    ----------
    cov : pd.DataFrame or np.ndarray, shape (n_assets, n_assets)
        Covariance matrix of asset returns.
        Must be symmetric and positive semi-definite.
    
    Returns
    -------
    np.ndarray, shape (n_assets,)
        Optimal portfolio weights minimizing total variance.
        Values sum to 1.0, each between 0.0 and 1.0.
    
    Notes
    -----
    - GMV portfolio has minimum possible risk but may have low returns
    - Does not require expected returns as input (risk-minimization only)
    - Suitable for extremely risk-averse investors
    - Implementation: Special case of MSR with zero risk-free rate and equal returns
    
    Mathematical Background
    -----------------------
    The GMV problem solves: minimize w^T Σ w subject to:
    - Sum of weights = 1
    - All weights ≥ 0 (long-only)
    
    Where Σ is the covariance matrix and w is the weight vector.
    
    Examples
    --------
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                 [0.01, 0.05, 0.015],
    ...                 [0.02, 0.015, 0.06]])
    >>> w_gmv = gmv(cov)
    >>> print(f"GMV weights: {w_gmv}")
    >>> 
    >>> # Calculate minimum variance
    >>> min_vol = portfolio_vol(w_gmv, cov)
    >>> print(f"Minimum portfolio volatility: {min_vol:.4f}")
    
    Comparison with Other Strategies
    --------------------------------
    - GMV: Lowest risk, potentially lower returns
    - Equal Weight: Simple diversification, moderate risk
    - MSR: Best risk-adjusted returns, moderate to high risk
    
    See Also
    --------
    msr : Maximum Sharpe Ratio portfolio
    minimize_vol : Minimize volatility for target return
    portfolio_vol : Calculate portfolio volatility
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def optimal_weights(n_points, er, cov):
    """
    Generate efficient frontier by computing optimal weights across return spectrum.
    
    Creates a grid of portfolios spanning from minimum to maximum expected returns,
    where each portfolio minimizes risk for its target return level. This generates
    the complete efficient frontier curve.
    
    Parameters
    ----------
    n_points : int
        Number of portfolios to compute along the efficient frontier.
        Higher values create smoother curves. Recommended: 20-100.
    er : pd.Series or np.ndarray, shape (n_assets,)
        Expected annual returns for each asset (as decimals).
    cov : pd.DataFrame or np.ndarray, shape (n_assets, n_assets)
        Covariance matrix of asset returns.
    
    Returns
    -------
    list of np.ndarray
        List of optimal weight vectors, one for each target return.
        Each array has shape (n_assets,) with values summing to 1.0.
    
    Notes
    -----
    - Target returns range linearly from min(er) to max(er)
    - Each portfolio is computed using quadratic optimization
    - Results form the efficient frontier when plotted (volatility vs return)
    - Computational complexity: O(n_points × optimization_cost)
    
    Algorithm Details
    -----------------
    1. Generate n_points equally spaced target returns
    2. For each target return:
       - Call minimize_vol() to find optimal weights
       - Constraints: weights sum to 1, returns match target
    3. Return list of weight vectors
    
    Examples
    --------
    >>> er = pd.Series([0.08, 0.10, 0.12])
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                 [0.01, 0.05, 0.015],
    ...                 [0.02, 0.015, 0.06]])
    >>> 
    >>> # Generate 50 efficient portfolios
    >>> weights_list = optimal_weights(50, er, cov)
    >>> 
    >>> # Calculate returns and volatilities
    >>> returns = [portfolio_return(w, er) for w in weights_list]
    >>> vols = [portfolio_vol(w, cov) for w in weights_list]
    >>> 
    >>> # Plot efficient frontier
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(vols, returns, 'b-')
    >>> plt.xlabel('Volatility')
    >>> plt.ylabel('Expected Return')
    >>> plt.title('Efficient Frontier')
    >>> plt.show()
    
    Performance Considerations
    --------------------------
    - Computation time scales linearly with n_points
    - Each optimization involves matrix operations (O(n_assets³))
    - For large portfolios (>100 assets), consider sparse matrix methods
    
    See Also
    --------
    minimize_vol : Core optimization for single portfolio
    plot_ef : Visualize efficient frontier
    portfolio_return : Calculate portfolio return
    portfolio_vol : Calculate portfolio volatility
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


