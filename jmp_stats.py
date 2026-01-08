"""
JMP-Style Statistical Analysis Library for Python
==================================================

A comprehensive library that replicates JMP's statistical analysis capabilities
using single function calls. Leverages scipy, statsmodels, pandas, numpy, and
matplotlib/seaborn for visualization.

Installation Requirements:
    pip install numpy pandas scipy statsmodels matplotlib seaborn

NEW IN v1.1:
    - read_csv() / read_excel(): Import data with automatic type detection
    - hat_matrix(): Full influence diagnostics (leverage, Cook's D, DFFITS, DFBETAS)
    - subset_regression(): All possible subsets with multiple criteria
    - stepwise_regression_enhanced(): Multiple stopping rules and directions
    - compare_stepwise_criteria(): Compare all stopping rules side-by-side

Usage Examples:

    # Import data
    from jmp_stats import read_csv, describe, linear_regression, stepwise_regression_enhanced
    
    df = read_csv('data.csv')
    
    # Descriptive statistics
    stats = describe(df['price'])
    print(stats)
    
    # Linear regression with influence diagnostics
    results = linear_regression(df['y'], df[['x1', 'x2']])
    print(results)
    
    hat_results = hat_matrix(df['y'], df[['x1', 'x2']])
    print(hat_results)
    
    # Stepwise regression with different criteria
    # Using BIC (default)
    step_bic = stepwise_regression_enhanced(df['y'], df[predictors], 
                                            direction='both', criterion='bic')
    
    # Using cross-validation
    step_cv = stepwise_regression_enhanced(df['y'], df[predictors],
                                           direction='forward', criterion='cv', cv_folds=10)
    
    # Using p-value rules
    step_pval = stepwise_regression_enhanced(df['y'], df[predictors],
                                             direction='backward', criterion='pvalue',
                                             alpha_enter=0.05, alpha_remove=0.10)
    
    # All subsets regression
    subsets = subset_regression(df['y'], df[predictors], max_vars=10)
    print(subsets.best_overall)

Stopping Criteria for Stepwise:
    - 'pvalue': P-value threshold (add if p < alpha_enter, remove if p > alpha_remove)
    - 'aic': Akaike Information Criterion (minimize)
    - 'bic': Bayesian Information Criterion (minimize)  
    - 'adj_rsq': Adjusted R-squared (maximize)
    - 'rmse': Root Mean Square Error (minimize)
    - 'cp' / 'mallows_cp': Mallows' Cp (minimize)
    - 'cv': Cross-validated RMSE (minimize)

Directions for Stepwise:
    - 'forward': Start with no variables, add one at a time
    - 'backward': Start with all variables, remove one at a time
    - 'both' / 'mixed': Both forward and backward steps at each iteration
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, normaltest, jarque_bera, levene, bartlett
from scipy.stats import f_oneway, kruskal, pearsonr, spearmanr, kendalltau
from scipy.stats import t as t_dist, norm, chi2
import warnings
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

# Optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# =============================================================================
# DATA CLASSES FOR STRUCTURED OUTPUT
# =============================================================================

@dataclass
class DescriptiveStats:
    """Container for descriptive statistics results (JMP Distribution style)."""
    n: int
    mean: float
    std: float
    std_err: float
    variance: float
    cv: float
    min: float
    max: float
    range: float
    median: float
    mode: Any
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    quantiles: Dict[str, float]
    missing: int
    
    def __str__(self):
        return f"""
Descriptive Statistics
======================
N                    {self.n}
Missing              {self.missing}
Mean                 {self.mean:.6f}
Std Dev              {self.std:.6f}
Std Err Mean         {self.std_err:.6f}
Variance             {self.variance:.6f}
CV (%)               {self.cv:.2f}

Minimum              {self.min:.6f}
Maximum              {self.max:.6f}
Range                {self.range:.6f}

Quantiles:
  100% (Max)         {self.quantiles['100%']:.6f}
  99.5%              {self.quantiles['99.5%']:.6f}
  97.5%              {self.quantiles['97.5%']:.6f}
  90%                {self.quantiles['90%']:.6f}
  75% (Q3)           {self.q3:.6f}
  50% (Median)       {self.median:.6f}
  25% (Q1)           {self.q1:.6f}
  10%                {self.quantiles['10%']:.6f}
  2.5%               {self.quantiles['2.5%']:.6f}
  0.5%               {self.quantiles['0.5%']:.6f}
  0% (Min)           {self.quantiles['0%']:.6f}

IQR                  {self.iqr:.6f}
Skewness             {self.skewness:.6f}
Kurtosis             {self.kurtosis:.6f}
"""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy export."""
        data = {
            'Statistic': ['N', 'Missing', 'Mean', 'Std Dev', 'Std Err Mean', 
                         'Variance', 'CV (%)', 'Min', 'Max', 'Range', 'Median',
                         'Q1', 'Q3', 'IQR', 'Skewness', 'Kurtosis'],
            'Value': [self.n, self.missing, self.mean, self.std, self.std_err,
                     self.variance, self.cv, self.min, self.max, self.range,
                     self.median, self.q1, self.q3, self.iqr, self.skewness,
                     self.kurtosis]
        }
        return pd.DataFrame(data)


@dataclass
class NormalityTest:
    """Container for normality test results."""
    shapiro_stat: float
    shapiro_pvalue: float
    dagostino_stat: float
    dagostino_pvalue: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    is_normal_shapiro: bool
    is_normal_dagostino: bool
    is_normal_jb: bool
    alpha: float = 0.05
    
    def __str__(self):
        return f"""
Normality Tests (alpha = {self.alpha})
======================================
Test                 Statistic    P-Value      Normal?
------------------------------------------------------
Shapiro-Wilk         {self.shapiro_stat:10.6f}   {self.shapiro_pvalue:.6f}   {'Yes' if self.is_normal_shapiro else 'No'}
D'Agostino-Pearson   {self.dagostino_stat:10.6f}   {self.dagostino_pvalue:.6f}   {'Yes' if self.is_normal_dagostino else 'No'}
Jarque-Bera          {self.jarque_bera_stat:10.6f}   {self.jarque_bera_pvalue:.6f}   {'Yes' if self.is_normal_jb else 'No'}
"""


@dataclass
class RegressionResults:
    """Container for regression analysis results (JMP Fit Model style)."""
    r_squared: float
    adj_r_squared: float
    rmse: float
    mae: float
    mape: float
    intercept: float
    intercept_se: float
    intercept_tstat: float
    intercept_pvalue: float
    slopes: Dict[str, float]
    slope_se: Dict[str, float]
    slope_tstat: Dict[str, float]
    slope_pvalue: Dict[str, float]
    ss_regression: float
    ss_residual: float
    ss_total: float
    df_regression: int
    df_residual: int
    df_total: int
    ms_regression: float
    ms_residual: float
    f_statistic: float
    f_pvalue: float
    durbin_watson: float
    n_obs: int
    vif: Optional[Dict[str, float]] = None
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    predicted: np.ndarray = field(default_factory=lambda: np.array([]))
    conf_int: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def __str__(self):
        coef_lines = []
        for var in self.slopes:
            ci = self.conf_int.get(var, (np.nan, np.nan))
            coef_lines.append(
                f"{var:20s} {self.slopes[var]:12.6f} {self.slope_se[var]:10.6f} "
                f"{self.slope_tstat[var]:10.4f} {self.slope_pvalue[var]:10.6f}  "
                f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            )
        
        ci_int = self.conf_int.get('Intercept', (np.nan, np.nan))
        
        return f"""
Linear Regression Results
=========================
Summary of Fit:
  R-Squared            {self.r_squared:.6f}
  Adj R-Squared        {self.adj_r_squared:.6f}
  Root Mean Sq Error   {self.rmse:.6f}
  Mean Absolute Error  {self.mae:.6f}
  Observations         {self.n_obs}

Analysis of Variance:
Source          DF      Sum of Squares     Mean Square      F Ratio    Prob > F
--------------------------------------------------------------------------------
Model           {self.df_regression:3d}     {self.ss_regression:16.4f}  {self.ms_regression:14.4f}   {self.f_statistic:10.4f}   {self.f_pvalue:.6f}
Error           {self.df_residual:3d}     {self.ss_residual:16.4f}  {self.ms_residual:14.4f}
Total           {self.df_total:3d}     {self.ss_total:16.4f}

Parameter Estimates:
Term                      Estimate     Std Error      t Ratio    Prob>|t|     95% CI
-------------------------------------------------------------------------------------
Intercept            {self.intercept:12.6f} {self.intercept_se:10.6f} {self.intercept_tstat:10.4f} {self.intercept_pvalue:10.6f}  [{ci_int[0]:.4f}, {ci_int[1]:.4f}]
{chr(10).join(coef_lines)}

Durbin-Watson        {self.durbin_watson:.6f}
"""

    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """Return DataFrames for coefficients, ANOVA, and fit stats."""
        coef_data = {
            'Term': ['Intercept'] + list(self.slopes.keys()),
            'Estimate': [self.intercept] + list(self.slopes.values()),
            'Std Error': [self.intercept_se] + list(self.slope_se.values()),
            't Ratio': [self.intercept_tstat] + list(self.slope_tstat.values()),
            'Prob>|t|': [self.intercept_pvalue] + list(self.slope_pvalue.values()),
        }
        anova_data = {
            'Source': ['Model', 'Error', 'Total'],
            'DF': [self.df_regression, self.df_residual, self.df_total],
            'Sum of Squares': [self.ss_regression, self.ss_residual, self.ss_total],
            'Mean Square': [self.ms_regression, self.ms_residual, np.nan],
            'F Ratio': [self.f_statistic, np.nan, np.nan],
            'Prob > F': [self.f_pvalue, np.nan, np.nan]
        }
        fit_data = {
            'Statistic': ['R-Squared', 'Adj R-Squared', 'RMSE', 'MAE', 'N'],
            'Value': [self.r_squared, self.adj_r_squared, self.rmse, self.mae, self.n_obs]
        }
        return {
            'coefficients': pd.DataFrame(coef_data),
            'anova': pd.DataFrame(anova_data),
            'fit_statistics': pd.DataFrame(fit_data)
        }


@dataclass 
class ResidualDiagnostics:
    """Container for residual diagnostic results."""
    bp_stat: float
    bp_pvalue: float
    white_stat: float
    white_pvalue: float
    shapiro_stat: float
    shapiro_pvalue: float
    durbin_watson: float
    leverage_points: List[int]
    high_influence_points: List[int]
    outlier_residuals: List[int]
    homoscedastic: bool
    residuals_normal: bool
    independent: bool
    
    def __str__(self):
        dw_interpretation = "Independent" if 1.5 < self.durbin_watson < 2.5 else "Possible autocorrelation"
        return f"""
Residual Diagnostics
====================
Heteroscedasticity Tests:
  Breusch-Pagan:     χ² = {self.bp_stat:.4f}, p = {self.bp_pvalue:.6f}  {'✓ Homoscedastic' if self.homoscedastic else '✗ Heteroscedastic'}
  White Test:        χ² = {self.white_stat:.4f}, p = {self.white_pvalue:.6f}

Normality of Residuals:
  Shapiro-Wilk:      W = {self.shapiro_stat:.4f}, p = {self.shapiro_pvalue:.6f}  {'✓ Normal' if self.residuals_normal else '✗ Non-normal'}

Independence:
  Durbin-Watson:     {self.durbin_watson:.4f}  ({dw_interpretation})

Outlier Detection:
  High Leverage Points:     {len(self.leverage_points)} observations
  High Influence (Cook's D): {len(self.high_influence_points)} observations  
  Outlier Residuals (|t|>2): {len(self.outlier_residuals)} observations
"""


@dataclass
class CorrelationResults:
    """Container for correlation analysis results."""
    pearson_r: float
    pearson_pvalue: float
    spearman_r: float
    spearman_pvalue: float
    kendall_tau: float
    kendall_pvalue: float
    n: int
    
    def __str__(self):
        return f"""
Correlation Analysis (n = {self.n})
===================================
Method              Correlation    P-Value
------------------------------------------
Pearson r           {self.pearson_r:10.6f}   {self.pearson_pvalue:.6f}
Spearman ρ          {self.spearman_r:10.6f}   {self.spearman_pvalue:.6f}
Kendall τ           {self.kendall_tau:10.6f}   {self.kendall_pvalue:.6f}
"""


@dataclass
class ANOVAResults:
    """Container for ANOVA results."""
    f_statistic: float
    f_pvalue: float
    df_between: int
    df_within: int
    ss_between: float
    ss_within: float
    ss_total: float
    ms_between: float
    ms_within: float
    group_means: Dict[str, float]
    group_stds: Dict[str, float]
    group_ns: Dict[str, int]
    eta_squared: float
    omega_squared: float
    tukey_results: Optional[Any] = None
    
    def __str__(self):
        group_lines = []
        for g in self.group_means:
            group_lines.append(
                f"  {g:20s}  n={self.group_ns[g]:5d}  "
                f"Mean={self.group_means[g]:12.4f}  SD={self.group_stds[g]:10.4f}"
            )
        return f"""
One-Way ANOVA
=============
Analysis of Variance:
Source              DF      Sum of Squares     Mean Square      F Ratio    Prob > F
-----------------------------------------------------------------------------------
Between Groups      {self.df_between:3d}     {self.ss_between:16.4f}  {self.ms_between:14.4f}   {self.f_statistic:10.4f}   {self.f_pvalue:.6f}
Within Groups       {self.df_within:3d}     {self.ss_within:16.4f}  {self.ms_within:14.4f}
Total               {self.df_between + self.df_within:3d}     {self.ss_total:16.4f}

Effect Sizes:
  η² (Eta-squared)    {self.eta_squared:.6f}
  ω² (Omega-squared)  {self.omega_squared:.6f}

Group Statistics:
{chr(10).join(group_lines)}
"""


@dataclass
class TTestResults:
    """Container for t-test results."""
    test_type: str
    t_statistic: float
    p_value_two_tailed: float
    p_value_one_tailed: float
    df: float
    mean_diff: float
    std_err: float
    conf_int_low: float
    conf_int_high: float
    effect_size_d: float
    sample_means: List[float]
    sample_stds: List[float]
    sample_ns: List[int]
    
    def __str__(self):
        return f"""
T-Test Results ({self.test_type})
{'=' * (len(self.test_type) + 18)}
Test Statistic:      t = {self.t_statistic:.6f}
Degrees of Freedom:  df = {self.df:.2f}
P-Value (two-tail):  {self.p_value_two_tailed:.6f}
P-Value (one-tail):  {self.p_value_one_tailed:.6f}

Mean Difference:     {self.mean_diff:.6f}
Std Error:           {self.std_err:.6f}
95% CI:              [{self.conf_int_low:.6f}, {self.conf_int_high:.6f}]

Effect Size:
  Cohen's d          {self.effect_size_d:.4f}

Sample Statistics:
  Sample 1: n={self.sample_ns[0]}, Mean={self.sample_means[0]:.4f}, SD={self.sample_stds[0]:.4f}
""" + (f"  Sample 2: n={self.sample_ns[1]}, Mean={self.sample_means[1]:.4f}, SD={self.sample_stds[1]:.4f}" 
       if len(self.sample_ns) > 1 else "")


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def describe(data: Union[pd.Series, np.ndarray, List], dropna: bool = True) -> DescriptiveStats:
    """
    Generate comprehensive descriptive statistics (JMP Distribution style).
    
    Parameters
    ----------
    data : array-like
        Numeric data to analyze
    dropna : bool
        Whether to drop NA values before analysis
        
    Returns
    -------
    DescriptiveStats
        Object containing all descriptive statistics
        
    Example
    -------
    >>> stats = describe(df['price'])
    >>> print(stats)
    >>> stats.to_dataframe()
    """
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(data, pd.Series):
        arr = data.values
    else:
        arr = np.asarray(data)
    
    missing = int(np.sum(np.isnan(arr))) if arr.dtype.kind == 'f' else 0
    if dropna:
        arr = arr[~np.isnan(arr)] if arr.dtype.kind == 'f' else arr
    
    n = len(arr)
    if n == 0:
        raise ValueError("No valid data points after removing NA values")
    
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))
    variance = float(np.var(arr, ddof=1))
    
    quantiles = {
        '100%': float(np.percentile(arr, 100)),
        '99.5%': float(np.percentile(arr, 99.5)),
        '97.5%': float(np.percentile(arr, 97.5)),
        '90%': float(np.percentile(arr, 90)),
        '75%': float(np.percentile(arr, 75)),
        '50%': float(np.percentile(arr, 50)),
        '25%': float(np.percentile(arr, 25)),
        '10%': float(np.percentile(arr, 10)),
        '2.5%': float(np.percentile(arr, 2.5)),
        '0.5%': float(np.percentile(arr, 0.5)),
        '0%': float(np.percentile(arr, 0)),
    }
    
    try:
        mode_result = stats.mode(arr, keepdims=True)
        mode_val = mode_result.mode[0]
    except:
        mode_val = np.nan
    
    return DescriptiveStats(
        n=n,
        mean=mean_val,
        std=std_val,
        std_err=std_val / np.sqrt(n),
        variance=variance,
        cv=(std_val / mean_val * 100) if mean_val != 0 else np.inf,
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        range=float(np.max(arr) - np.min(arr)),
        median=float(np.median(arr)),
        mode=mode_val,
        q1=float(np.percentile(arr, 25)),
        q3=float(np.percentile(arr, 75)),
        iqr=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        skewness=float(stats.skew(arr)),
        kurtosis=float(stats.kurtosis(arr)),
        quantiles=quantiles,
        missing=missing
    )


def test_normality(data: Union[pd.Series, np.ndarray, List], alpha: float = 0.05) -> NormalityTest:
    """
    Perform comprehensive normality tests.
    
    Parameters
    ----------
    data : array-like
        Numeric data to test
    alpha : float
        Significance level for hypothesis tests
        
    Returns
    -------
    NormalityTest
        Object containing test statistics and conclusions
    """
    arr = np.asarray(data)
    arr = arr[~np.isnan(arr)]
    
    if len(arr) <= 5000:
        shap_stat, shap_p = shapiro(arr)
    else:
        sample = np.random.choice(arr, 5000, replace=False)
        shap_stat, shap_p = shapiro(sample)
    
    if len(arr) >= 20:
        dag_stat, dag_p = normaltest(arr)
    else:
        dag_stat, dag_p = np.nan, np.nan
    
    jb_stat, jb_p = jarque_bera(arr)
    
    return NormalityTest(
        shapiro_stat=float(shap_stat),
        shapiro_pvalue=float(shap_p),
        dagostino_stat=float(dag_stat) if not np.isnan(dag_stat) else np.nan,
        dagostino_pvalue=float(dag_p) if not np.isnan(dag_p) else np.nan,
        jarque_bera_stat=float(jb_stat),
        jarque_bera_pvalue=float(jb_p),
        is_normal_shapiro=shap_p > alpha,
        is_normal_dagostino=dag_p > alpha if not np.isnan(dag_p) else None,
        is_normal_jb=jb_p > alpha,
        alpha=alpha
    )


def linear_regression(y: Union[pd.Series, np.ndarray],
                      X: Union[pd.DataFrame, np.ndarray, pd.Series],
                      add_constant: bool = True,
                      alpha: float = 0.05) -> RegressionResults:
    """
    Perform linear regression with comprehensive output (JMP Fit Model style).
    
    Parameters
    ----------
    y : array-like
        Dependent variable (response)
    X : array-like or DataFrame
        Independent variable(s) (predictors)
    add_constant : bool
        Whether to add intercept term
    alpha : float
        Significance level for confidence intervals
        
    Returns
    -------
    RegressionResults
        Comprehensive regression results object
        
    Example
    -------
    >>> results = linear_regression(df['PRSM'], df[['FICO', 'Repayment_Pct']])
    >>> print(results)
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required. Install: pip install statsmodels")
    
    y = np.asarray(y).flatten()
    
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(X, pd.DataFrame):
        X_names = list(X.columns)
        X = X.values
    else:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_names = [f'X{i+1}' for i in range(X.shape[1])]
    
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y = y[mask]
    X = X[mask]
    
    n = len(y)
    k = X.shape[1]
    
    if add_constant:
        X_with_const = sm.add_constant(X)
    else:
        X_with_const = X
    
    model = sm.OLS(y, X_with_const).fit()
    
    predictions = model.predict(X_with_const)
    residuals = y - predictions
    
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / y)) * 100) if np.all(y != 0) else np.nan
    
    if add_constant:
        intercept = float(model.params[0])
        intercept_se = float(model.bse[0])
        intercept_t = float(model.tvalues[0])
        intercept_p = float(model.pvalues[0])
        slopes = {name: float(model.params[i+1]) for i, name in enumerate(X_names)}
        slope_se = {name: float(model.bse[i+1]) for i, name in enumerate(X_names)}
        slope_t = {name: float(model.tvalues[i+1]) for i, name in enumerate(X_names)}
        slope_p = {name: float(model.pvalues[i+1]) for i, name in enumerate(X_names)}
        conf_int = {'Intercept': (float(model.conf_int(alpha)[0, 0]), float(model.conf_int(alpha)[0, 1]))}
        for i, name in enumerate(X_names):
            conf_int[name] = (float(model.conf_int(alpha)[i+1, 0]), float(model.conf_int(alpha)[i+1, 1]))
    else:
        intercept = 0.0
        intercept_se = 0.0
        intercept_t = np.nan
        intercept_p = np.nan
        slopes = {name: float(model.params[i]) for i, name in enumerate(X_names)}
        slope_se = {name: float(model.bse[i]) for i, name in enumerate(X_names)}
        slope_t = {name: float(model.tvalues[i]) for i, name in enumerate(X_names)}
        slope_p = {name: float(model.pvalues[i]) for i, name in enumerate(X_names)}
        conf_int = {}
        for i, name in enumerate(X_names):
            conf_int[name] = (float(model.conf_int(alpha)[i, 0]), float(model.conf_int(alpha)[i, 1]))
    
    vif = None
    if k > 1 and add_constant:
        try:
            vif = {}
            for i, name in enumerate(X_names):
                vif[name] = float(variance_inflation_factor(X_with_const, i+1))
        except:
            vif = None
    
    dw = float(durbin_watson(residuals))
    
    return RegressionResults(
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        rmse=rmse,
        mae=mae,
        mape=mape,
        intercept=intercept,
        intercept_se=intercept_se,
        intercept_tstat=intercept_t,
        intercept_pvalue=intercept_p,
        slopes=slopes,
        slope_se=slope_se,
        slope_tstat=slope_t,
        slope_pvalue=slope_p,
        ss_regression=float(model.ess),
        ss_residual=float(model.ssr),
        ss_total=float(model.ess + model.ssr),
        df_regression=int(model.df_model),
        df_residual=int(model.df_resid),
        df_total=int(model.df_model + model.df_resid),
        ms_regression=float(model.mse_model),
        ms_residual=float(model.mse_resid),
        f_statistic=float(model.fvalue),
        f_pvalue=float(model.f_pvalue),
        durbin_watson=dw,
        n_obs=n,
        vif=vif,
        residuals=residuals,
        predicted=predictions,
        conf_int=conf_int
    )


def residual_diagnostics(y: Union[pd.Series, np.ndarray],
                         X: Union[pd.DataFrame, np.ndarray, pd.Series],
                         alpha: float = 0.05) -> ResidualDiagnostics:
    """
    Perform comprehensive residual diagnostics for regression.
    
    Checks: Heteroscedasticity, Normality, Independence, Outliers/Influence
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required")
    
    y = np.asarray(y).flatten()
    
    if isinstance(X, pd.Series):
        X = X.values.reshape(-1, 1)
    elif isinstance(X, pd.DataFrame):
        X = X.values
    else:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
    
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y = y[mask]
    X = X[mask]
    
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    residuals = model.resid
    
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_const)
    try:
        white_stat, white_p, _, _ = het_white(residuals, X_const)
    except:
        white_stat, white_p = np.nan, np.nan
    
    if len(residuals) <= 5000:
        shap_stat, shap_p = shapiro(residuals)
    else:
        shap_stat, shap_p = shapiro(np.random.choice(residuals, 5000, replace=False))
    
    dw = durbin_watson(residuals)
    
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    leverage_threshold = 2 * (X.shape[1] + 1) / len(y)
    leverage_points = list(np.where(leverage > leverage_threshold)[0])
    
    cooks_d = influence.cooks_distance[0]
    cooks_threshold = 4 / len(y)
    high_influence = list(np.where(cooks_d > cooks_threshold)[0])
    
    stud_resid = influence.resid_studentized_external
    outliers = list(np.where(np.abs(stud_resid) > 2)[0])
    
    return ResidualDiagnostics(
        bp_stat=float(bp_stat),
        bp_pvalue=float(bp_p),
        white_stat=float(white_stat) if not np.isnan(white_stat) else np.nan,
        white_pvalue=float(white_p) if not np.isnan(white_p) else np.nan,
        shapiro_stat=float(shap_stat),
        shapiro_pvalue=float(shap_p),
        durbin_watson=float(dw),
        leverage_points=leverage_points,
        high_influence_points=high_influence,
        outlier_residuals=outliers,
        homoscedastic=bp_p > alpha,
        residuals_normal=shap_p > alpha,
        independent=1.5 < dw < 2.5
    )


def correlation(x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray]) -> CorrelationResults:
    """Calculate correlation coefficients with significance tests."""
    x = np.asarray(x)
    y = np.asarray(y)
    
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    kendall_t, kendall_p = kendalltau(x, y)
    
    return CorrelationResults(
        pearson_r=float(pearson_r),
        pearson_pvalue=float(pearson_p),
        spearman_r=float(spearman_r),
        spearman_pvalue=float(spearman_p),
        kendall_tau=float(kendall_t),
        kendall_pvalue=float(kendall_p),
        n=len(x)
    )


def correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate correlation matrix with p-values."""
    numeric_df = df.select_dtypes(include=[np.number])
    n = len(numeric_df.columns)
    corr_mat = np.zeros((n, n))
    pval_mat = np.zeros((n, n))
    
    for i, col1 in enumerate(numeric_df.columns):
        for j, col2 in enumerate(numeric_df.columns):
            x = numeric_df[col1].dropna()
            y = numeric_df[col2].dropna()
            common = x.index.intersection(y.index)
            x = x.loc[common]
            y = y.loc[common]
            
            if method == 'pearson':
                r, p = pearsonr(x, y)
            elif method == 'spearman':
                r, p = spearmanr(x, y)
            elif method == 'kendall':
                r, p = kendalltau(x, y)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            corr_mat[i, j] = r
            pval_mat[i, j] = p
    
    return (pd.DataFrame(corr_mat, index=numeric_df.columns, columns=numeric_df.columns),
            pd.DataFrame(pval_mat, index=numeric_df.columns, columns=numeric_df.columns))


def oneway_anova(data: pd.DataFrame, response: str, factor: str, 
                 alpha: float = 0.05, posthoc: bool = True) -> ANOVAResults:
    """Perform one-way ANOVA with effect sizes and optional post-hoc tests."""
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required")
    
    df_clean = data[[response, factor]].dropna()
    
    groups = df_clean.groupby(factor)[response]
    group_means = groups.mean().to_dict()
    group_stds = groups.std().to_dict()
    group_ns = groups.count().to_dict()
    
    group_data = [group[response].values for name, group in df_clean.groupby(factor)]
    
    f_stat, p_val = f_oneway(*group_data)
    
    grand_mean = df_clean[response].mean()
    n_total = len(df_clean)
    k = len(group_means)
    
    ss_total = np.sum((df_clean[response] - grand_mean)**2)
    ss_between = sum(n * (m - grand_mean)**2 for m, n in zip(group_means.values(), group_ns.values()))
    ss_within = ss_total - ss_between
    
    df_between = k - 1
    df_within = n_total - k
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    eta_sq = ss_between / ss_total
    omega_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within)
    
    tukey = None
    if posthoc and p_val < alpha:
        tukey = pairwise_tukeyhsd(df_clean[response], df_clean[factor], alpha=alpha)
    
    return ANOVAResults(
        f_statistic=float(f_stat),
        f_pvalue=float(p_val),
        df_between=df_between,
        df_within=df_within,
        ss_between=float(ss_between),
        ss_within=float(ss_within),
        ss_total=float(ss_total),
        ms_between=float(ms_between),
        ms_within=float(ms_within),
        group_means={k: float(v) for k, v in group_means.items()},
        group_stds={k: float(v) for k, v in group_stds.items()},
        group_ns={k: int(v) for k, v in group_ns.items()},
        eta_squared=float(eta_sq),
        omega_squared=float(omega_sq),
        tukey_results=tukey
    )


def ttest_1sample(data: Union[pd.Series, np.ndarray], popmean: float, alpha: float = 0.05) -> TTestResults:
    """One-sample t-test."""
    arr = np.asarray(data)
    arr = arr[~np.isnan(arr)]
    
    t_stat, p_two = stats.ttest_1samp(arr, popmean)
    
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    se = std / np.sqrt(n)
    df = n - 1
    
    t_crit = t_dist.ppf(1 - alpha/2, df)
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    
    d = (mean - popmean) / std
    
    return TTestResults(
        test_type='one-sample', t_statistic=float(t_stat),
        p_value_two_tailed=float(p_two), p_value_one_tailed=float(p_two/2),
        df=float(df), mean_diff=float(mean - popmean), std_err=float(se),
        conf_int_low=float(ci_low), conf_int_high=float(ci_high),
        effect_size_d=float(d), sample_means=[float(mean)],
        sample_stds=[float(std)], sample_ns=[n]
    )


def ttest_2sample(group1: Union[pd.Series, np.ndarray], group2: Union[pd.Series, np.ndarray],
                  equal_var: bool = True, alpha: float = 0.05) -> TTestResults:
    """Two-sample t-test (independent samples)."""
    a1 = np.asarray(group1)
    a2 = np.asarray(group2)
    a1 = a1[~np.isnan(a1)]
    a2 = a2[~np.isnan(a2)]
    
    t_stat, p_two = stats.ttest_ind(a1, a2, equal_var=equal_var)
    
    n1, n2 = len(a1), len(a2)
    m1, m2 = np.mean(a1), np.mean(a2)
    s1, s2 = np.std(a1, ddof=1), np.std(a2, ddof=1)
    
    mean_diff = m1 - m2
    
    if equal_var:
        sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        se = sp * np.sqrt(1/n1 + 1/n2)
        df = n1 + n2 - 2
        d = mean_diff / sp
    else:
        se = np.sqrt(s1**2/n1 + s2**2/n2)
        df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
        d = mean_diff / np.sqrt((s1**2 + s2**2) / 2)
    
    t_crit = t_dist.ppf(1 - alpha/2, df)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se
    
    return TTestResults(
        test_type='two-sample (equal var)' if equal_var else 'two-sample (Welch)',
        t_statistic=float(t_stat), p_value_two_tailed=float(p_two),
        p_value_one_tailed=float(p_two/2), df=float(df),
        mean_diff=float(mean_diff), std_err=float(se),
        conf_int_low=float(ci_low), conf_int_high=float(ci_high),
        effect_size_d=float(d), sample_means=[float(m1), float(m2)],
        sample_stds=[float(s1), float(s2)], sample_ns=[n1, n2]
    )


def ttest_paired(before: Union[pd.Series, np.ndarray], after: Union[pd.Series, np.ndarray],
                 alpha: float = 0.05) -> TTestResults:
    """Paired t-test (dependent samples)."""
    a1 = np.asarray(before)
    a2 = np.asarray(after)
    
    mask = ~(np.isnan(a1) | np.isnan(a2))
    a1 = a1[mask]
    a2 = a2[mask]
    
    t_stat, p_two = stats.ttest_rel(a1, a2)
    
    diff = a1 - a2
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se = std_diff / np.sqrt(n)
    df = n - 1
    
    t_crit = t_dist.ppf(1 - alpha/2, df)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se
    
    d = mean_diff / std_diff
    
    return TTestResults(
        test_type='paired', t_statistic=float(t_stat),
        p_value_two_tailed=float(p_two), p_value_one_tailed=float(p_two/2),
        df=float(df), mean_diff=float(mean_diff), std_err=float(se),
        conf_int_low=float(ci_low), conf_int_high=float(ci_high),
        effect_size_d=float(d), sample_means=[float(np.mean(a1)), float(np.mean(a2))],
        sample_stds=[float(np.std(a1, ddof=1)), float(np.std(a2, ddof=1))],
        sample_ns=[n, n]
    )


def normal_probability(mean: float, std: float, x: Optional[float] = None,
                       prob: Optional[float] = None, tail: str = 'lower') -> Dict[str, float]:
    """
    Normal probability calculator (like JMP's Normal Probability Calculator).
    
    Example
    -------
    >>> # What's the probability of repaying at least $60,000?
    >>> result = normal_probability(mean=64000, std=16000, x=60000)
    >>> print(f"P(X >= 60000) = {result['prob_upper']:.4f}")
    """
    if x is not None:
        z = (x - mean) / std
        p_lower = norm.cdf(z)
        p_upper = 1 - p_lower
        return {'x': x, 'z_score': z, 'prob_lower': p_lower, 'prob_upper': p_upper}
    elif prob is not None:
        if tail == 'lower':
            z = norm.ppf(prob)
        else:
            z = norm.ppf(1 - prob)
        x_val = mean + z * std
        return {'x': x_val, 'z_score': z, 
                'prob_lower': prob if tail == 'lower' else 1-prob,
                'prob_upper': 1-prob if tail == 'lower' else prob}
    else:
        raise ValueError("Must provide either x or prob")


def stepwise_regression(y: Union[pd.Series, np.ndarray], X: pd.DataFrame,
                        method: str = 'both', criterion: str = 'bic',
                        alpha_enter: float = 0.05, alpha_remove: float = 0.10,
                        max_iter: int = 100, verbose: bool = False) -> Dict[str, Any]:
    """
    Stepwise regression for variable selection (like JMP's Stepwise platform).
    
    Parameters
    ----------
    y : array-like - Response variable
    X : DataFrame - Candidate predictors
    method : str - 'forward', 'backward', or 'both' (stepwise)
    criterion : str - 'bic', 'aic', or 'pvalue'
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required")
    
    y = np.asarray(y)
    mask = ~np.isnan(y)
    y = y[mask]
    X = X.iloc[mask].copy()
    X = X.dropna(axis=1, how='any')
    
    all_features = list(X.columns)
    
    def get_criterion(model, criterion):
        if criterion == 'bic':
            return model.bic
        elif criterion == 'aic':
            return model.aic
        return None
    
    def fit_model(features):
        if len(features) == 0:
            return None
        X_sub = sm.add_constant(X[features])
        return sm.OLS(y, X_sub).fit()
    
    selected = [] if method != 'backward' else all_features.copy()
    step_history = []
    
    for iteration in range(max_iter):
        changed = False
        
        if method in ['forward', 'both']:
            remaining = [f for f in all_features if f not in selected]
            best_new_score = None
            best_new_feature = None
            
            for feature in remaining:
                test_features = selected + [feature]
                model = fit_model(test_features)
                if model is None:
                    continue
                
                if criterion == 'pvalue':
                    idx = list(test_features).index(feature) + 1
                    pval = model.pvalues[idx]
                    if pval < alpha_enter:
                        if best_new_score is None or pval < best_new_score:
                            best_new_score = pval
                            best_new_feature = feature
                else:
                    score = get_criterion(model, criterion)
                    if best_new_score is None or score < best_new_score:
                        best_new_score = score
                        best_new_feature = feature
            
            if best_new_feature is not None:
                if len(selected) > 0:
                    current_model = fit_model(selected)
                    current_score = get_criterion(current_model, criterion)
                    if criterion == 'pvalue' or best_new_score < current_score:
                        selected.append(best_new_feature)
                        step_history.append((iteration+1, 'enter', best_new_feature, best_new_score))
                        changed = True
                        if verbose:
                            print(f"Step {iteration+1}: Enter {best_new_feature}")
                else:
                    selected.append(best_new_feature)
                    step_history.append((iteration+1, 'enter', best_new_feature, best_new_score))
                    changed = True
                    if verbose:
                        print(f"Step {iteration+1}: Enter {best_new_feature}")
        
        if method in ['backward', 'both'] and len(selected) > 0:
            current_model = fit_model(selected)
            current_bic = get_criterion(current_model, criterion)
            worst_score = None
            worst_feature = None
            
            for feature in selected:
                test_features = [f for f in selected if f != feature]
                if len(test_features) == 0:
                    continue
                model = fit_model(test_features)
                
                if criterion == 'pvalue':
                    idx = list(selected).index(feature) + 1
                    pval = current_model.pvalues[idx]
                    if pval > alpha_remove:
                        if worst_score is None or pval > worst_score:
                            worst_score = pval
                            worst_feature = feature
                else:
                    score = get_criterion(model, criterion)
                    if score < current_bic:
                        if worst_score is None or score < worst_score:
                            worst_score = score
                            worst_feature = feature
            
            if worst_feature is not None:
                selected.remove(worst_feature)
                step_history.append((iteration+1, 'remove', worst_feature, worst_score))
                changed = True
                if verbose:
                    print(f"Step {iteration+1}: Remove {worst_feature}")
        
        if not changed:
            break
    
    final_model = linear_regression(y, X[selected]) if len(selected) > 0 else None
    
    return {
        'selected_features': selected,
        'model': final_model,
        'step_history': step_history,
        'n_iterations': iteration + 1
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_distribution(data: Union[pd.Series, np.ndarray], title: str = "Distribution",
                      bins: int = 30, show_stats: bool = True,
                      figsize: Tuple[int, int] = (12, 8)) -> Optional[Any]:
    """Create JMP-style distribution plot with histogram, box plot, and normal overlay."""
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return None
    
    arr = np.asarray(data)
    arr = arr[~np.isnan(arr)]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    ax1 = axes[0, 0]
    n, bins_arr, patches = ax1.hist(arr, bins=bins, density=True, alpha=0.7, edgecolor='black')
    mu, sigma = norm.fit(arr)
    x = np.linspace(arr.min(), arr.max(), 100)
    ax1.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{title} - Histogram')
    ax1.legend()
    
    ax2 = axes[0, 1]
    bp = ax2.boxplot(arr, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax2.set_ylabel('Value')
    ax2.set_title(f'{title} - Box Plot')
    
    ax3 = axes[1, 0]
    stats.probplot(arr, dist="norm", plot=ax3)
    ax3.set_title(f'{title} - Normal Q-Q Plot')
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if show_stats:
        desc = describe(arr)
        stats_text = f"""
Summary Statistics
==================
N:          {desc.n}
Mean:       {desc.mean:.4f}
Std Dev:    {desc.std:.4f}
Min:        {desc.min:.4f}
Q1:         {desc.q1:.4f}
Median:     {desc.median:.4f}
Q3:         {desc.q3:.4f}
Max:        {desc.max:.4f}
Skewness:   {desc.skewness:.4f}
Kurtosis:   {desc.kurtosis:.4f}
"""
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def plot_regression_diagnostics(y: Union[pd.Series, np.ndarray],
                                X: Union[pd.DataFrame, np.ndarray, pd.Series],
                                figsize: Tuple[int, int] = (14, 10)) -> Optional[Any]:
    """Create JMP-style regression diagnostic plots."""
    if not HAS_MATPLOTLIB or not HAS_STATSMODELS:
        print("matplotlib and statsmodels required")
        return None
    
    y = np.asarray(y).flatten()
    
    if isinstance(X, pd.Series):
        X = X.values.reshape(-1, 1)
    elif isinstance(X, pd.DataFrame):
        X = X.values
    else:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
    
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y = y[mask]
    X = X[mask]
    
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    predicted = model.predict(X_const)
    residuals = model.resid
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    ax1 = axes[0, 0]
    ax1.scatter(predicted, y, alpha=0.5, edgecolors='black', linewidth=0.5)
    min_val = min(predicted.min(), y.min())
    max_val = max(predicted.max(), y.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect fit')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Actual by Predicted Plot')
    ax1.legend()
    
    ax2 = axes[0, 1]
    ax2.scatter(predicted, residuals, alpha=0.5, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Residual')
    ax2.set_title('Residual by Predicted Plot')
    
    ax3 = axes[1, 0]
    ax3.scatter(range(len(residuals)), residuals, alpha=0.5, edgecolors='black', linewidth=0.5)
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Row Number')
    ax3.set_ylabel('Residual')
    ax3.set_title('Residual by Row Plot')
    
    ax4 = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Residual Normal Quantile Plot')
    
    plt.tight_layout()
    return fig


def plot_scatter_with_regression(x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray],
                                 x_label: str = 'X', y_label: str = 'Y',
                                 title: str = 'Bivariate Fit', show_ci: bool = True,
                                 figsize: Tuple[int, int] = (10, 8)) -> Optional[Any]:
    """Create scatter plot with regression line and confidence bands."""
    if not HAS_MATPLOTLIB:
        return None
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(x, y, alpha=0.5, edgecolors='black', linewidth=0.5, label='Data')
    
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', lw=2, 
            label=f'Fit: y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}')
    
    if show_ci and HAS_STATSMODELS:
        X_const = sm.add_constant(x)
        model = sm.OLS(y, X_const).fit()
        X_pred = sm.add_constant(x_line)
        pred = model.get_prediction(X_pred)
        ci = pred.conf_int(alpha=0.05)
        ax.fill_between(x_line, ci[:, 0], ci[:, 1], alpha=0.2, color='red', label='95% CI')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10),
                            annot: bool = True) -> Optional[Any]:
    """Create correlation matrix heatmap."""
    if not HAS_MATPLOTLIB:
        return None
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if HAS_SEABORN:
        sns.heatmap(corr_matrix, annot=annot, cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, ax=ax, fmt='.2f')
    else:
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        if annot:
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center')
    
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    return fig


def plot_control_chart(data: Union[pd.Series, np.ndarray], chart_type: str = 'individuals',
                       subgroup_size: int = 1, title: str = 'Control Chart',
                       figsize: Tuple[int, int] = (14, 6)) -> Optional[Any]:
    """Create statistical process control chart."""
    if not HAS_MATPLOTLIB:
        return None
    
    arr = np.asarray(data)
    arr = arr[~np.isnan(arr)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mean = np.mean(arr)
    mr = np.abs(np.diff(arr))
    mr_bar = np.mean(mr)
    
    ucl = mean + 3 * mr_bar / 1.128
    lcl = mean - 3 * mr_bar / 1.128
    
    x = range(len(arr))
    ax.plot(x, arr, 'b-o', markersize=4, label='Individual Values')
    ax.axhline(y=mean, color='green', linestyle='-', lw=2, label=f'Mean = {mean:.4f}')
    ax.axhline(y=ucl, color='red', linestyle='--', lw=2, label=f'UCL = {ucl:.4f}')
    ax.axhline(y=lcl, color='red', linestyle='--', lw=2, label=f'LCL = {lcl:.4f}')
    
    ooc = (arr > ucl) | (arr < lcl)
    if np.any(ooc):
        ax.scatter(np.where(ooc)[0], arr[ooc], color='red', s=100, zorder=5, label='Out of Control')
    
    ax.set_xlabel('Observation')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return fig


# =============================================================================
# CONVENIENCE FUNCTIONS (JMP-style workflows)
# =============================================================================

def fit_y_by_x(data: pd.DataFrame, y_col: str, x_col: str, plot: bool = True) -> Dict[str, Any]:
    """
    JMP-style Fit Y by X analysis - combines regression with diagnostics and plots.
    
    Example
    -------
    >>> results = fit_y_by_x(df, y_col='Amount_Repaid', x_col='Total_Amount')
    >>> print(results['regression'])
    """
    df_clean = data[[y_col, x_col]].dropna()
    y = df_clean[y_col]
    X = df_clean[[x_col]]
    
    reg = linear_regression(y, X)
    diag = residual_diagnostics(y, X)
    corr = correlation(X[x_col], y)
    
    result = {'regression': reg, 'diagnostics': diag, 'correlation': corr, 'figure': None}
    
    if plot and HAS_MATPLOTLIB:
        result['figure'] = plot_scatter_with_regression(X[x_col], y, x_label=x_col, y_label=y_col)
    
    return result


def fit_model(data: pd.DataFrame, response: str, predictors: List[str],
              interactions: Optional[List[Tuple[str, str]]] = None,
              plot_diagnostics: bool = True) -> Dict[str, Any]:
    """
    JMP-style Fit Model analysis - multiple regression with full diagnostics.
    
    Example
    -------
    >>> results = fit_model(df, 'PRSM', ['FICO', 'Repayment_Pct'],
    ...                     interactions=[('FICO', 'Repayment_Pct')])
    """
    df_clean = data[[response] + predictors].dropna()
    
    X = df_clean[predictors].copy()
    
    if interactions:
        for var1, var2 in interactions:
            interaction_name = f'{var1}*{var2}'
            X[interaction_name] = df_clean[var1] * df_clean[var2]
    
    y = df_clean[response]
    
    reg = linear_regression(y, X)
    diag = residual_diagnostics(y, X)
    
    effects_data = {
        'Source': list(reg.slopes.keys()),
        'DF': [1] * len(reg.slopes),
        't Ratio': list(reg.slope_tstat.values()),
        'Prob>|t|': list(reg.slope_pvalue.values()),
        'Significant': [p < 0.05 for p in reg.slope_pvalue.values()]
    }
    effects_df = pd.DataFrame(effects_data)
    
    result = {'regression': reg, 'diagnostics': diag, 'effects_tests': effects_df, 'figure': None}
    
    if plot_diagnostics and HAS_MATPLOTLIB:
        result['figure'] = plot_regression_diagnostics(y, X)
    
    return result


def distribution_analysis(data: Union[pd.Series, np.ndarray], name: str = "Variable",
                          plot: bool = True) -> Dict[str, Any]:
    """JMP-style Distribution analysis - descriptive stats, normality tests, and plots."""
    desc = describe(data)
    norm_test = test_normality(data)
    
    result = {'descriptive': desc, 'normality': norm_test, 'figure': None}
    
    if plot and HAS_MATPLOTLIB:
        result['figure'] = plot_distribution(data, title=name)
    
    return result


def multivariate_analysis(data: pd.DataFrame, variables: Optional[List[str]] = None,
                          plot: bool = True) -> Dict[str, Any]:
    """JMP-style Multivariate analysis - correlation matrix with scatterplot matrix."""
    if variables:
        df_subset = data[variables]
    else:
        df_subset = data.select_dtypes(include=[np.number])
    
    corr_mat, pval_mat = correlation_matrix(df_subset)
    
    result = {'correlation_matrix': corr_mat, 'pvalue_matrix': pval_mat, 'figure': None}
    
    if plot and HAS_MATPLOTLIB:
        result['figure'] = plot_correlation_matrix(df_subset)
    
    return result


# =============================================================================
# DATA UTILITIES
# =============================================================================

def detect_outliers(data: Union[pd.Series, np.ndarray], method: str = 'iqr',
                    threshold: float = 1.5) -> Dict[str, Any]:
    """Detect outliers using IQR or Z-score method."""
    arr = np.asarray(data)
    arr = arr[~np.isnan(arr)]
    
    if method == 'iqr':
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
    elif method == 'zscore':
        mean = np.mean(arr)
        std = np.std(arr)
        lower = mean - threshold * std
        upper = mean + threshold * std
    else:
        raise ValueError(f"Unknown method: {method}")
    
    outlier_mask = (arr < lower) | (arr > upper)
    
    return {
        'outlier_indices': np.where(outlier_mask)[0],
        'outlier_values': arr[outlier_mask],
        'lower_bound': lower,
        'upper_bound': upper,
        'n_outliers': int(np.sum(outlier_mask))
    }


def recode(data: pd.Series, mapping: Dict[Any, Any], default: Any = None) -> pd.Series:
    """Recode values in a series (like JMP's Recode function)."""
    if default is None:
        return data.map(lambda x: mapping.get(x, x))
    else:
        return data.map(lambda x: mapping.get(x, default))


def log_transform(data: Union[pd.Series, np.ndarray], add_constant: float = 1.0) -> np.ndarray:
    """Apply log transformation with optional constant for zeros."""
    arr = np.asarray(data)
    return np.log(arr + add_constant)


# =============================================================================
# DATA IMPORT FUNCTIONS
# =============================================================================

def read_csv(filepath: str, 
             sep: str = ',',
             header: Union[int, List[int], str] = 'infer',
             index_col: Optional[Union[int, str, List]] = None,
             usecols: Optional[List] = None,
             dtype: Optional[Dict] = None,
             na_values: Optional[List] = None,
             parse_dates: Union[bool, List] = False,
             encoding: str = 'utf-8',
             skip_rows: Optional[int] = None,
             n_rows: Optional[int] = None,
             low_memory: bool = True,
             verbose: bool = True) -> pd.DataFrame:
    """
    Import CSV file and create a DataFrame ready for analysis.
    
    Automatically detects and handles common data issues:
    - Identifies numeric vs categorical columns
    - Handles missing values
    - Provides summary of imported data
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    sep : str
        Delimiter (default ',')
    header : int, list, or 'infer'
        Row number(s) for column headers
    index_col : int, str, or list, optional
        Column(s) to use as row index
    usecols : list, optional
        Subset of columns to import
    dtype : dict, optional
        Data types for columns {'col_name': dtype}
    na_values : list, optional
        Additional strings to recognize as NA/NaN
    parse_dates : bool or list
        Columns to parse as dates
    encoding : str
        File encoding (default 'utf-8')
    skip_rows : int, optional
        Number of rows to skip at start
    n_rows : int, optional
        Number of rows to read
    low_memory : bool
        Process file in chunks for lower memory usage
    verbose : bool
        Print summary of imported data
        
    Returns
    -------
    pd.DataFrame
        DataFrame ready for use with all jmp_stats functions
        
    Example
    -------
    >>> df = read_csv('data.csv')
    >>> df = read_csv('data.csv', na_values=['NA', 'N/A', '.', ''])
    >>> df = read_csv('data.csv', usecols=['FICO', 'PRSM', 'Loan_Amount'])
    """
    # Default NA values that JMP recognizes
    default_na = ['', 'NA', 'N/A', 'NaN', 'nan', '.', 'NULL', 'null', 'None', 'none', '#N/A', '#VALUE!']
    if na_values:
        default_na.extend(na_values)
    
    try:
        df = pd.read_csv(
            filepath,
            sep=sep,
            header=header,
            index_col=index_col,
            usecols=usecols,
            dtype=dtype,
            na_values=default_na,
            parse_dates=parse_dates,
            encoding=encoding,
            skiprows=skip_rows,
            nrows=n_rows,
            low_memory=low_memory
        )
    except UnicodeDecodeError:
        # Try alternative encodings
        for enc in ['latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, sep=sep, header=header, index_col=index_col,
                                usecols=usecols, dtype=dtype, na_values=default_na,
                                parse_dates=parse_dates, encoding=enc, skiprows=skip_rows,
                                nrows=n_rows, low_memory=low_memory)
                if verbose:
                    print(f"Successfully read file using {enc} encoding")
                break
            except:
                continue
        else:
            raise ValueError(f"Could not read file with any common encoding")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Data Import Summary")
        print(f"{'='*60}")
        print(f"File: {filepath}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(f"\nColumn Types:")
        print(f"  Numeric:     {len(df.select_dtypes(include=[np.number]).columns)}")
        print(f"  Categorical: {len(df.select_dtypes(include=['object', 'category']).columns)}")
        print(f"  DateTime:    {len(df.select_dtypes(include=['datetime64']).columns)}")
        
        # Missing values summary
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing Values:")
            for col in missing[missing > 0].index:
                pct = missing[col] / len(df) * 100
                print(f"  {col}: {missing[col]:,} ({pct:.1f}%)")
        else:
            print(f"\nNo missing values detected")
        
        print(f"\nColumn Names:")
        for i, col in enumerate(df.columns):
            dtype_str = str(df[col].dtype)
            print(f"  {i+1}. {col} ({dtype_str})")
        print(f"{'='*60}\n")
    
    return df


def read_excel(filepath: str,
               sheet_name: Union[str, int] = 0,
               header: Union[int, List[int]] = 0,
               index_col: Optional[Union[int, str]] = None,
               usecols: Optional[List] = None,
               dtype: Optional[Dict] = None,
               na_values: Optional[List] = None,
               verbose: bool = True) -> pd.DataFrame:
    """
    Import Excel file (.xlsx, .xls) and create a DataFrame ready for analysis.
    
    Parameters
    ----------
    filepath : str
        Path to the Excel file
    sheet_name : str or int
        Sheet name or index (default: first sheet)
    header : int or list
        Row number(s) for column headers
    index_col : int or str, optional
        Column to use as row index
    usecols : list, optional
        Subset of columns to import
    dtype : dict, optional
        Data types for columns
    na_values : list, optional
        Additional strings to recognize as NA/NaN
    verbose : bool
        Print summary of imported data
        
    Returns
    -------
    pd.DataFrame
        DataFrame ready for use with all jmp_stats functions
    """
    default_na = ['', 'NA', 'N/A', 'NaN', 'nan', '.', 'NULL', 'null', 'None', 'none']
    if na_values:
        default_na.extend(na_values)
    
    df = pd.read_excel(
        filepath,
        sheet_name=sheet_name,
        header=header,
        index_col=index_col,
        usecols=usecols,
        dtype=dtype,
        na_values=default_na
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Data Import Summary")
        print(f"{'='*60}")
        print(f"File: {filepath}")
        print(f"Sheet: {sheet_name}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nMissing Values:")
            for col in missing[missing > 0].index[:10]:  # Show first 10
                pct = missing[col] / len(df) * 100
                print(f"  {col}: {missing[col]:,} ({pct:.1f}%)")
        print(f"{'='*60}\n")
    
    return df


# =============================================================================
# HAT MATRIX AND INFLUENCE DIAGNOSTICS
# =============================================================================

@dataclass
class HatMatrixResults:
    """Container for hat matrix and influence diagnostics."""
    hat_diag: np.ndarray  # Diagonal of hat matrix (leverage values)
    leverage_threshold: float
    high_leverage_indices: List[int]
    cooks_distance: np.ndarray
    cooks_threshold: float
    high_influence_indices: List[int]
    dffits: np.ndarray
    dffits_threshold: float
    dfbetas: np.ndarray
    dfbetas_threshold: float
    studentized_residuals: np.ndarray
    outlier_indices: List[int]  # |studentized residual| > 2
    
    def __str__(self):
        return f"""
Hat Matrix and Influence Diagnostics
=====================================

Leverage (Hat Values):
  Threshold (2p/n):           {self.leverage_threshold:.6f}
  High Leverage Points:       {len(self.high_leverage_indices)} observations
  Indices:                    {self.high_leverage_indices[:10]}{'...' if len(self.high_leverage_indices) > 10 else ''}

Cook's Distance:
  Threshold (4/n):            {self.cooks_threshold:.6f}
  High Influence Points:      {len(self.high_influence_indices)} observations
  Indices:                    {self.high_influence_indices[:10]}{'...' if len(self.high_influence_indices) > 10 else ''}

DFFITS:
  Threshold (2√(p/n)):        {self.dffits_threshold:.6f}
  
Studentized Residuals:
  Outliers (|t| > 2):         {len(self.outlier_indices)} observations
  Indices:                    {self.outlier_indices[:10]}{'...' if len(self.outlier_indices) > 10 else ''}
"""
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return DataFrame with all diagnostic values for each observation."""
        n = len(self.hat_diag)
        return pd.DataFrame({
            'Observation': range(n),
            'Leverage': self.hat_diag,
            'High_Leverage': self.hat_diag > self.leverage_threshold,
            'Cooks_D': self.cooks_distance,
            'High_Influence': self.cooks_distance > self.cooks_threshold,
            'DFFITS': self.dffits,
            'Studentized_Resid': self.studentized_residuals,
            'Outlier': np.abs(self.studentized_residuals) > 2
        })


def hat_matrix(y: Union[pd.Series, np.ndarray],
               X: Union[pd.DataFrame, np.ndarray, pd.Series],
               return_full_matrix: bool = False) -> Union[HatMatrixResults, Tuple[HatMatrixResults, np.ndarray]]:
    """
    Calculate hat matrix and comprehensive influence diagnostics.
    
    The hat matrix H = X(X'X)^(-1)X' maps observed y to predicted y-hat.
    Diagonal elements h_ii measure leverage of each observation.
    
    Parameters
    ----------
    y : array-like
        Response variable
    X : array-like
        Predictor variable(s)
    return_full_matrix : bool
        If True, also return the full n×n hat matrix (memory intensive for large n)
        
    Returns
    -------
    HatMatrixResults
        Comprehensive influence diagnostics
    np.ndarray (optional)
        Full hat matrix if return_full_matrix=True
        
    Example
    -------
    >>> hat_results = hat_matrix(df['y'], df[['x1', 'x2']])
    >>> print(hat_results)
    >>> diagnostics_df = hat_results.to_dataframe()
    
    Notes
    -----
    Key thresholds:
    - Leverage: h_ii > 2p/n indicates high leverage
    - Cook's D: D_i > 4/n indicates high influence
    - DFFITS: |DFFITS| > 2√(p/n) indicates influence on fitted values
    - Studentized residuals: |t_i| > 2 suggests outlier
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required")
    
    y = np.asarray(y).flatten()
    
    if isinstance(X, pd.Series):
        X = X.values.reshape(-1, 1)
    elif isinstance(X, pd.DataFrame):
        X = X.values
    else:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
    
    # Remove NaN
    mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    y = y[mask]
    X = X[mask]
    
    n = len(y)
    
    # Add constant
    X_const = sm.add_constant(X)
    p = X_const.shape[1]  # Number of parameters including intercept
    
    # Fit model
    model = sm.OLS(y, X_const).fit()
    
    # Get influence measures
    influence = model.get_influence()
    
    # Hat matrix diagonal (leverage)
    hat_diag = influence.hat_matrix_diag
    leverage_threshold = 2 * p / n
    high_leverage = list(np.where(hat_diag > leverage_threshold)[0])
    
    # Cook's Distance
    cooks_d = influence.cooks_distance[0]
    cooks_threshold = 4 / n
    high_influence = list(np.where(cooks_d > cooks_threshold)[0])
    
    # DFFITS
    dffits = influence.dffits[0]
    dffits_threshold = 2 * np.sqrt(p / n)
    
    # DFBETAS
    dfbetas = influence.dfbetas
    dfbetas_threshold = 2 / np.sqrt(n)
    
    # Studentized residuals (external)
    stud_resid = influence.resid_studentized_external
    outliers = list(np.where(np.abs(stud_resid) > 2)[0])
    
    results = HatMatrixResults(
        hat_diag=hat_diag,
        leverage_threshold=leverage_threshold,
        high_leverage_indices=high_leverage,
        cooks_distance=cooks_d,
        cooks_threshold=cooks_threshold,
        high_influence_indices=high_influence,
        dffits=dffits,
        dffits_threshold=dffits_threshold,
        dfbetas=dfbetas,
        dfbetas_threshold=dfbetas_threshold,
        studentized_residuals=stud_resid,
        outlier_indices=outliers
    )
    
    if return_full_matrix:
        # Calculate full hat matrix H = X(X'X)^(-1)X'
        XtX_inv = np.linalg.inv(X_const.T @ X_const)
        H = X_const @ XtX_inv @ X_const.T
        return results, H
    
    return results


# =============================================================================
# SUBSET REGRESSION (ALL POSSIBLE REGRESSIONS)
# =============================================================================

@dataclass
class SubsetRegressionResults:
    """Container for all subsets regression results."""
    results_df: pd.DataFrame  # DataFrame with all subset results
    best_by_rsq: Dict[int, Dict]  # Best model for each size by R²
    best_by_adj_rsq: Dict[int, Dict]  # Best by Adjusted R²
    best_by_aic: Dict[int, Dict]  # Best by AIC
    best_by_bic: Dict[int, Dict]  # Best by BIC
    best_by_cp: Dict[int, Dict]  # Best by Mallows' Cp
    best_by_rmse: Dict[int, Dict]  # Best by RMSE
    best_overall: Dict[str, Dict]  # Best model by each criterion
    n_models: int
    
    def __str__(self):
        lines = [
            "\nAll Subsets Regression Summary",
            "=" * 50,
            f"Total models evaluated: {self.n_models}",
            "\nBest Models by Criterion:",
            "-" * 50,
        ]
        
        for criterion, model in self.best_overall.items():
            vars_str = ', '.join(model['variables'][:5])
            if len(model['variables']) > 5:
                vars_str += '...'
            lines.append(f"\n{criterion}:")
            lines.append(f"  Variables ({model['n_vars']}): {vars_str}")
            lines.append(f"  R² = {model['r_squared']:.6f}")
            lines.append(f"  Adj R² = {model['adj_r_squared']:.6f}")
            lines.append(f"  RMSE = {model['rmse']:.6f}")
            lines.append(f"  AIC = {model['aic']:.2f}")
            lines.append(f"  BIC = {model['bic']:.2f}")
            lines.append(f"  Cp = {model['mallows_cp']:.4f}")
        
        return '\n'.join(lines)


def subset_regression(y: Union[pd.Series, np.ndarray],
                      X: pd.DataFrame,
                      max_vars: Optional[int] = None,
                      criterion: str = 'all',
                      cv_folds: int = 5,
                      verbose: bool = True) -> SubsetRegressionResults:
    """
    Perform all subsets regression to find best variable combinations.
    
    Evaluates all possible combinations of predictors and ranks them by
    multiple criteria: R², Adjusted R², AIC, BIC, Mallows' Cp, RMSE, and CV.
    
    Parameters
    ----------
    y : array-like
        Response variable
    X : DataFrame
        Candidate predictor variables
    max_vars : int, optional
        Maximum number of variables to consider (default: all)
    criterion : str
        'all' to compute all criteria, or specific: 'rsq', 'adj_rsq', 'aic', 'bic', 'cp', 'rmse', 'cv'
    cv_folds : int
        Number of folds for cross-validation
    verbose : bool
        Print progress
        
    Returns
    -------
    SubsetRegressionResults
        Results containing best models by each criterion
        
    Example
    -------
    >>> results = subset_regression(df['y'], df[['x1', 'x2', 'x3', 'x4']])
    >>> print(results)
    >>> results.results_df  # Full DataFrame of all models
    >>> results.best_overall['BIC']  # Best model by BIC
    
    Notes
    -----
    Warning: Number of models grows as 2^p. For p > 15 variables,
    consider using stepwise regression instead.
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required")
    
    from itertools import combinations
    
    y = np.asarray(y).flatten()
    
    # Clean data
    all_cols = list(X.columns)
    df_combined = pd.concat([pd.Series(y, name='_y_'), X], axis=1).dropna()
    y_clean = df_combined['_y_'].values
    X_clean = df_combined[all_cols]
    
    n = len(y_clean)
    p_total = len(all_cols)
    
    if max_vars is None:
        max_vars = p_total
    
    if p_total > 20:
        warnings.warn(f"Warning: {p_total} variables will generate {2**p_total - 1:,} models. "
                     f"Consider using max_vars parameter or stepwise regression.")
    
    # Calculate MSE for full model (needed for Mallows' Cp)
    X_full = sm.add_constant(X_clean.values)
    model_full = sm.OLS(y_clean, X_full).fit()
    mse_full = model_full.mse_resid
    
    results = []
    total_models = sum(len(list(combinations(all_cols, k))) for k in range(1, max_vars + 1))
    
    if verbose:
        print(f"Evaluating {total_models:,} models...")
    
    model_count = 0
    
    for k in range(1, max_vars + 1):
        for var_combo in combinations(all_cols, k):
            model_count += 1
            
            if verbose and model_count % 100 == 0:
                print(f"  Progress: {model_count}/{total_models} models ({100*model_count/total_models:.1f}%)")
            
            var_list = list(var_combo)
            X_subset = sm.add_constant(X_clean[var_list].values)
            
            try:
                model = sm.OLS(y_clean, X_subset).fit()
                
                # Calculate metrics
                p = k + 1  # Number of parameters (including intercept)
                
                # Mallows' Cp = SSE_p/MSE_full - n + 2p
                sse = model.ssr
                mallows_cp = sse / mse_full - n + 2 * p
                
                # RMSE
                rmse = np.sqrt(model.mse_resid)
                
                # Cross-validation (optional, slower)
                cv_rmse = np.nan
                cv_adj_rsq = np.nan
                if criterion in ['all', 'cv']:
                    cv_rmse, cv_adj_rsq = _cross_validate(y_clean, X_clean[var_list].values, cv_folds)
                
                results.append({
                    'n_vars': k,
                    'variables': var_list,
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'rmse': rmse,
                    'aic': model.aic,
                    'bic': model.bic,
                    'mallows_cp': mallows_cp,
                    'cv_rmse': cv_rmse,
                    'cv_adj_rsq': cv_adj_rsq,
                    'f_statistic': model.fvalue,
                    'f_pvalue': model.f_pvalue
                })
            except:
                continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best models by each criterion for each model size
    best_by_rsq = {}
    best_by_adj_rsq = {}
    best_by_aic = {}
    best_by_bic = {}
    best_by_cp = {}
    best_by_rmse = {}
    
    for k in range(1, max_vars + 1):
        subset_k = results_df[results_df['n_vars'] == k]
        if len(subset_k) > 0:
            best_by_rsq[k] = subset_k.loc[subset_k['r_squared'].idxmax()].to_dict()
            best_by_adj_rsq[k] = subset_k.loc[subset_k['adj_r_squared'].idxmax()].to_dict()
            best_by_aic[k] = subset_k.loc[subset_k['aic'].idxmin()].to_dict()
            best_by_bic[k] = subset_k.loc[subset_k['bic'].idxmin()].to_dict()
            best_by_cp[k] = subset_k.loc[subset_k['mallows_cp'].idxmin()].to_dict()
            best_by_rmse[k] = subset_k.loc[subset_k['rmse'].idxmin()].to_dict()
    
    # Find overall best by each criterion
    best_overall = {
        'R²': results_df.loc[results_df['r_squared'].idxmax()].to_dict(),
        'Adjusted R²': results_df.loc[results_df['adj_r_squared'].idxmax()].to_dict(),
        'AIC': results_df.loc[results_df['aic'].idxmin()].to_dict(),
        'BIC': results_df.loc[results_df['bic'].idxmin()].to_dict(),
        'Mallows Cp': results_df.loc[results_df['mallows_cp'].idxmin()].to_dict(),
        'RMSE': results_df.loc[results_df['rmse'].idxmin()].to_dict(),
    }
    
    if criterion in ['all', 'cv'] and not results_df['cv_rmse'].isna().all():
        best_overall['CV RMSE'] = results_df.loc[results_df['cv_rmse'].idxmin()].to_dict()
        best_overall['CV Adj R²'] = results_df.loc[results_df['cv_adj_rsq'].idxmax()].to_dict()
    
    if verbose:
        print(f"Completed evaluation of {model_count} models")
    
    return SubsetRegressionResults(
        results_df=results_df,
        best_by_rsq=best_by_rsq,
        best_by_adj_rsq=best_by_adj_rsq,
        best_by_aic=best_by_aic,
        best_by_bic=best_by_bic,
        best_by_cp=best_by_cp,
        best_by_rmse=best_by_rmse,
        best_overall=best_overall,
        n_models=model_count
    )


def _cross_validate(y: np.ndarray, X: np.ndarray, n_folds: int = 5) -> Tuple[float, float]:
    """Helper function for k-fold cross-validation."""
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    
    cv_errors = []
    cv_rsq = []
    
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        
        X_train = sm.add_constant(X[train_idx])
        X_test = sm.add_constant(X[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        try:
            model = sm.OLS(y_train, X_train).fit()
            predictions = model.predict(X_test)
            mse = np.mean((y_test - predictions) ** 2)
            cv_errors.append(mse)
            
            # Calculate R² on test set
            ss_res = np.sum((y_test - predictions) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # Adjusted R² 
            n_test = len(y_test)
            p = X.shape[1]
            adj_r2 = 1 - (1 - r2) * (n_test - 1) / (n_test - p - 1)
            cv_rsq.append(adj_r2)
        except:
            continue
    
    if len(cv_errors) > 0:
        return np.sqrt(np.mean(cv_errors)), np.mean(cv_rsq)
    return np.nan, np.nan


# =============================================================================
# ENHANCED STEPWISE REGRESSION
# =============================================================================

@dataclass
class StepwiseResults:
    """Container for enhanced stepwise regression results."""
    selected_features: List[str]
    model: Optional[RegressionResults]
    step_history: List[Dict]
    stopping_criterion: str
    direction: str
    final_metrics: Dict[str, float]
    comparison_df: pd.DataFrame  # Comparison of stopping criteria
    
    def __str__(self):
        if not self.selected_features:
            return "No variables selected"
        
        history_str = ""
        for step in self.step_history[:10]:
            action = step.get('action', 'unknown')
            var = step.get('variable', 'unknown')
            criterion_val = step.get('criterion_value', np.nan)
            history_str += f"  Step {step.get('step', '?')}: {action.upper():6s} {var} "
            history_str += f"({self.stopping_criterion}={criterion_val:.4f})\n"
        
        if len(self.step_history) > 10:
            history_str += f"  ... and {len(self.step_history) - 10} more steps\n"
        
        vars_str = ', '.join(self.selected_features[:8])
        if len(self.selected_features) > 8:
            vars_str += '...'
        
        return f"""
Stepwise Regression Results
===========================
Direction:           {self.direction}
Stopping Criterion:  {self.stopping_criterion}
Variables Selected:  {len(self.selected_features)}
Variables:           {vars_str}

Final Model Metrics:
  R²:                {self.final_metrics.get('r_squared', np.nan):.6f}
  Adjusted R²:       {self.final_metrics.get('adj_r_squared', np.nan):.6f}
  RMSE:              {self.final_metrics.get('rmse', np.nan):.6f}
  AIC:               {self.final_metrics.get('aic', np.nan):.2f}
  BIC:               {self.final_metrics.get('bic', np.nan):.2f}
  Mallows' Cp:       {self.final_metrics.get('mallows_cp', np.nan):.4f}

Step History:
{history_str}
"""


def stepwise_regression_enhanced(
    y: Union[pd.Series, np.ndarray],
    X: pd.DataFrame,
    direction: str = 'both',
    criterion: str = 'bic',
    alpha_enter: float = 0.05,
    alpha_remove: float = 0.10,
    max_iter: int = 100,
    cv_folds: int = 5,
    verbose: bool = False
) -> StepwiseResults:
    """
    Enhanced stepwise regression with multiple stopping criteria and directions.
    
    Parameters
    ----------
    y : array-like
        Response variable
    X : DataFrame
        Candidate predictor variables
    direction : str
        Selection direction:
        - 'forward': Start empty, add variables
        - 'backward': Start full, remove variables
        - 'both' or 'mixed': Forward and backward at each step (stepwise)
    criterion : str
        Stopping criterion:
        - 'pvalue': Add lowest p-value < alpha_enter, remove highest p-value > alpha_remove
        - 'aic': Minimize Akaike Information Criterion
        - 'bic': Minimize Bayesian Information Criterion
        - 'adj_rsq' or 'adjusted_r2': Maximize Adjusted R²
        - 'rmse': Minimize Root Mean Square Error
        - 'mallows_cp' or 'cp': Minimize Mallows' Cp
        - 'cv' or 'crossvalidation': Minimize Cross-Validated RMSE
    alpha_enter : float
        P-value threshold to enter (for pvalue criterion)
    alpha_remove : float
        P-value threshold to remove (for pvalue criterion)
    max_iter : int
        Maximum number of iterations
    cv_folds : int
        Number of folds for cross-validation (if criterion='cv')
    verbose : bool
        Print step-by-step progress
        
    Returns
    -------
    StepwiseResults
        Comprehensive stepwise results with model and history
        
    Example
    -------
    >>> # Forward selection using AIC
    >>> results = stepwise_regression_enhanced(df['y'], df[predictors], 
    ...                                        direction='forward', criterion='aic')
    
    >>> # Backward elimination using p-values
    >>> results = stepwise_regression_enhanced(df['y'], df[predictors],
    ...                                        direction='backward', criterion='pvalue',
    ...                                        alpha_remove=0.10)
    
    >>> # Mixed stepwise using cross-validation
    >>> results = stepwise_regression_enhanced(df['y'], df[predictors],
    ...                                        direction='both', criterion='cv', cv_folds=10)
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required")
    
    # Normalize criterion names
    criterion = criterion.lower()
    criterion_map = {
        'adjusted_r2': 'adj_rsq',
        'adjusted_rsq': 'adj_rsq',
        'crossvalidation': 'cv',
        'cross_validation': 'cv',
        'mallows_cp': 'cp',
        'mallows': 'cp'
    }
    criterion = criterion_map.get(criterion, criterion)
    
    # Validate inputs
    valid_directions = ['forward', 'backward', 'both', 'mixed']
    valid_criteria = ['pvalue', 'aic', 'bic', 'adj_rsq', 'rmse', 'cp', 'cv']
    
    if direction not in valid_directions:
        raise ValueError(f"direction must be one of {valid_directions}")
    if criterion not in valid_criteria:
        raise ValueError(f"criterion must be one of {valid_criteria}")
    
    if direction == 'mixed':
        direction = 'both'
    
    y = np.asarray(y).flatten()
    
    # Clean data
    all_cols = list(X.columns)
    df_combined = pd.concat([pd.Series(y, name='_y_'), X], axis=1).dropna()
    y_clean = df_combined['_y_'].values
    X_clean = df_combined[all_cols]
    
    n = len(y_clean)
    
    # Calculate MSE for full model (needed for Mallows' Cp)
    X_full = sm.add_constant(X_clean.values)
    model_full = sm.OLS(y_clean, X_full).fit()
    mse_full = model_full.mse_resid
    
    def fit_and_evaluate(features):
        """Fit model and return evaluation metrics."""
        if len(features) == 0:
            return None
        
        X_sub = sm.add_constant(X_clean[features].values)
        model = sm.OLS(y_clean, X_sub).fit()
        
        p = len(features) + 1
        mallows_cp = model.ssr / mse_full - n + 2 * p
        
        metrics = {
            'model': model,
            'r_squared': model.rsquared,
            'adj_rsq': model.rsquared_adj,
            'aic': model.aic,
            'bic': model.bic,
            'rmse': np.sqrt(model.mse_resid),
            'cp': mallows_cp,
            'pvalues': dict(zip(['Intercept'] + features, model.pvalues))
        }
        
        # Cross-validation if needed
        if criterion == 'cv':
            cv_rmse, _ = _cross_validate(y_clean, X_clean[features].values, cv_folds)
            metrics['cv'] = cv_rmse
        
        return metrics
    
    def get_criterion_value(metrics, crit):
        """Get the value for comparison based on criterion."""
        if metrics is None:
            return np.inf if crit in ['aic', 'bic', 'rmse', 'cp', 'cv'] else -np.inf
        return metrics.get(crit, np.inf)
    
    def is_better(new_val, old_val, crit):
        """Check if new value is better than old based on criterion."""
        if crit in ['aic', 'bic', 'rmse', 'cp', 'cv']:
            return new_val < old_val  # Minimize
        else:  # adj_rsq
            return new_val > old_val  # Maximize
    
    # Initialize based on direction
    if direction == 'backward':
        selected = all_cols.copy()
    else:
        selected = []
    
    step_history = []
    comparison_data = []
    
    for iteration in range(max_iter):
        changed = False
        current_metrics = fit_and_evaluate(selected) if selected else None
        current_criterion = get_criterion_value(current_metrics, criterion)
        
        # Save comparison data
        if current_metrics:
            comparison_data.append({
                'step': iteration,
                'n_vars': len(selected),
                'variables': ', '.join(selected),
                'r_squared': current_metrics['r_squared'],
                'adj_rsq': current_metrics['adj_rsq'],
                'rmse': current_metrics['rmse'],
                'aic': current_metrics['aic'],
                'bic': current_metrics['bic'],
                'cp': current_metrics['cp']
            })
        
        # FORWARD STEP
        if direction in ['forward', 'both']:
            remaining = [f for f in all_cols if f not in selected]
            best_to_add = None
            best_add_criterion = current_criterion
            best_add_pvalue = 1.0
            
            for feature in remaining:
                test_features = selected + [feature]
                test_metrics = fit_and_evaluate(test_features)
                
                if test_metrics is None:
                    continue
                
                if criterion == 'pvalue':
                    # Get p-value of the new feature
                    feature_pvalue = test_metrics['pvalues'].get(feature, 1.0)
                    if feature_pvalue < alpha_enter and feature_pvalue < best_add_pvalue:
                        best_add_pvalue = feature_pvalue
                        best_to_add = feature
                        best_add_criterion = feature_pvalue
                else:
                    test_criterion = get_criterion_value(test_metrics, criterion)
                    if is_better(test_criterion, best_add_criterion, criterion):
                        best_add_criterion = test_criterion
                        best_to_add = feature
            
            # Add the best feature if it improves the model
            if best_to_add is not None:
                if criterion == 'pvalue' or is_better(best_add_criterion, current_criterion, criterion):
                    selected.append(best_to_add)
                    step_history.append({
                        'step': iteration + 1,
                        'action': 'enter',
                        'variable': best_to_add,
                        'criterion': criterion,
                        'criterion_value': best_add_criterion
                    })
                    changed = True
                    if verbose:
                        print(f"Step {iteration+1}: ENTER {best_to_add} ({criterion}={best_add_criterion:.6f})")
        
        # BACKWARD STEP
        if direction in ['backward', 'both'] and len(selected) > 0:
            # Re-fit current model after potential addition
            current_metrics = fit_and_evaluate(selected)
            current_criterion = get_criterion_value(current_metrics, criterion)
            
            best_to_remove = None
            best_remove_criterion = current_criterion
            worst_pvalue = 0.0
            
            for feature in selected:
                test_features = [f for f in selected if f != feature]
                if len(test_features) == 0:
                    continue
                    
                test_metrics = fit_and_evaluate(test_features)
                
                if criterion == 'pvalue':
                    # Get p-value of feature in current model
                    feature_pvalue = current_metrics['pvalues'].get(feature, 0.0)
                    if feature_pvalue > alpha_remove and feature_pvalue > worst_pvalue:
                        worst_pvalue = feature_pvalue
                        best_to_remove = feature
                        best_remove_criterion = feature_pvalue
                else:
                    test_criterion = get_criterion_value(test_metrics, criterion)
                    if is_better(test_criterion, best_remove_criterion, criterion):
                        best_remove_criterion = test_criterion
                        best_to_remove = feature
            
            # Remove the worst feature if it improves the model
            if best_to_remove is not None:
                if criterion == 'pvalue' or is_better(best_remove_criterion, current_criterion, criterion):
                    selected.remove(best_to_remove)
                    step_history.append({
                        'step': iteration + 1,
                        'action': 'remove',
                        'variable': best_to_remove,
                        'criterion': criterion,
                        'criterion_value': best_remove_criterion
                    })
                    changed = True
                    if verbose:
                        print(f"Step {iteration+1}: REMOVE {best_to_remove} ({criterion}={best_remove_criterion:.6f})")
        
        if not changed:
            break
    
    # Final model
    final_metrics = {}
    final_model = None
    
    if len(selected) > 0:
        final_eval = fit_and_evaluate(selected)
        if final_eval:
            final_metrics = {
                'r_squared': final_eval['r_squared'],
                'adj_r_squared': final_eval['adj_rsq'],
                'rmse': final_eval['rmse'],
                'aic': final_eval['aic'],
                'bic': final_eval['bic'],
                'mallows_cp': final_eval['cp']
            }
            final_model = linear_regression(y_clean, X_clean[selected])
    
    comparison_df = pd.DataFrame(comparison_data) if comparison_data else pd.DataFrame()
    
    return StepwiseResults(
        selected_features=selected,
        model=final_model,
        step_history=step_history,
        stopping_criterion=criterion,
        direction=direction,
        final_metrics=final_metrics,
        comparison_df=comparison_df
    )


def compare_stepwise_criteria(
    y: Union[pd.Series, np.ndarray],
    X: pd.DataFrame,
    direction: str = 'both',
    cv_folds: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run stepwise regression with all stopping criteria and compare results.
    
    Parameters
    ----------
    y : array-like
        Response variable
    X : DataFrame
        Candidate predictors
    direction : str
        'forward', 'backward', or 'both'
    cv_folds : int
        Folds for cross-validation
    verbose : bool
        Print progress
        
    Returns
    -------
    DataFrame
        Comparison of results across all criteria
        
    Example
    -------
    >>> comparison = compare_stepwise_criteria(df['y'], df[predictors])
    >>> print(comparison)
    """
    criteria = ['pvalue', 'aic', 'bic', 'adj_rsq', 'rmse', 'cp', 'cv']
    results = []
    
    for crit in criteria:
        if verbose:
            print(f"Running stepwise with {crit}...")
        
        try:
            res = stepwise_regression_enhanced(y, X, direction=direction, 
                                               criterion=crit, cv_folds=cv_folds)
            results.append({
                'Criterion': crit.upper(),
                'N_Variables': len(res.selected_features),
                'Variables': ', '.join(res.selected_features[:5]) + ('...' if len(res.selected_features) > 5 else ''),
                'R²': res.final_metrics.get('r_squared', np.nan),
                'Adj R²': res.final_metrics.get('adj_r_squared', np.nan),
                'RMSE': res.final_metrics.get('rmse', np.nan),
                'AIC': res.final_metrics.get('aic', np.nan),
                'BIC': res.final_metrics.get('bic', np.nan),
                'Mallows_Cp': res.final_metrics.get('mallows_cp', np.nan)
            })
        except Exception as e:
            if verbose:
                print(f"  Error with {crit}: {e}")
    
    return pd.DataFrame(results)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__version__ = "1.1.0"
__all__ = [
    # Data classes
    'DescriptiveStats', 'NormalityTest', 'RegressionResults', 'ResidualDiagnostics',
    'CorrelationResults', 'ANOVAResults', 'TTestResults', 
    'HatMatrixResults', 'SubsetRegressionResults', 'StepwiseResults',
    
    # Data import
    'read_csv', 'read_excel',
    
    # Core analysis
    'describe', 'test_normality', 'linear_regression', 'residual_diagnostics',
    'correlation', 'correlation_matrix', 'oneway_anova',
    'ttest_1sample', 'ttest_2sample', 'ttest_paired',
    'normal_probability',
    
    # Hat matrix and influence
    'hat_matrix',
    
    # Model selection
    'subset_regression', 'stepwise_regression', 
    'stepwise_regression_enhanced', 'compare_stepwise_criteria',
    
    # Convenience functions
    'fit_y_by_x', 'fit_model', 'distribution_analysis', 'multivariate_analysis',
    
    # Visualization
    'plot_distribution', 'plot_regression_diagnostics', 'plot_scatter_with_regression',
    'plot_correlation_matrix', 'plot_control_chart',
    
    # Utilities
    'detect_outliers', 'recode', 'log_transform',
]
