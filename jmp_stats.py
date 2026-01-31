"""
JMP-Style Statistical Analysis Library for Python
==================================================

A comprehensive library that replicates JMP's statistical analysis capabilities
using single function calls. Leverages scipy, statsmodels, pandas, numpy, and
matplotlib/seaborn for visualization.

Installation Requirements:
    pip install numpy pandas scipy statsmodels matplotlib seaborn scikit-learn

NEW IN v2.5.0:
    - K NEAREST NEIGHBORS (JMP's Analyze > Predictive Modeling > K Nearest Neighbors):
        - k_nearest_neighbors(): Auto-detects classification/regression
        - knn_classification(): Convenience wrapper for classification
        - knn_regression(): Convenience wrapper for regression
        - Automatic K selection based on validation error
        - Confusion matrix and misclassification rates
        - plot_knn_results(): JMP-style error rate plots
        - save_knn_predictions(): Like JMP's Save Predicteds
        - knn_lift_curve(): Lift curves for classification
        - Support for validation_column (JMP-style train/validation/test)
    
    - BOOTSTRAP METHODS (JMP's Bootstrap command):
        - bootstrap(): Generic bootstrap for any statistic
        - bootstrap_rmse(): Bootstrap confidence interval for RMSE
        - plot_bootstrap_distribution(): Visualize bootstrap distribution
        - Supports percentile, basic, and BCa confidence intervals

NEW IN v2.3:
    - ENHANCED LINEAR REGRESSION:
        - linear_regression_formula(): Formula-based regression with automatic
          categorical encoding (e.g., 'y ~ x1 + C(category)')
        - poly_degree parameter: Automatically generate polynomial terms
        - log_y parameter: Log-transform response variable
        - log_X parameter: Log-transform specified predictors
    
NEW IN v2.2:
    - PREDICTION INTERVALS:
        - prediction_interval(): Calculate prediction intervals for new observations
        - Returns both CI (for mean) and PI (for individual predictions)
        - Based on ISLP pg 110: "Prediction intervals will always be wider than
          confidence intervals because they account for the uncertainty 
          associated with ε, the irreducible error."

NEW IN v2.1:
    - CATEGORICAL VARIABLE ENCODING:
        - encode_categorical(): Create dummy variables for regression
        - encode_effect(): Effect coding (-1, 0, 1) like JMP's default
        - get_categorical_columns(): Identify categorical vs numeric columns
        - EncodedDataFrame class with encoding metadata and helper methods

NEW IN v2.0:
    - INTERACTIVE VISUALIZATIONS:
        - plot_leverage_interactive(): JMP-style interactive leverage plots
        - plot_influence_dashboard(): Comprehensive influence diagnostic dashboard
    - PREDICTION PROFILER:
        - prediction_profiler(): JMP-style interactive prediction profiler
        - PredictionProfiler class for saved profiler objects
    - DESIGN OF EXPERIMENTS (DoE):
        - fractional_factorial_design(): Create fractional factorial designs
        - response_surface_design(): Central Composite & Box-Behnken designs
        - optimal_design(): D-optimal and I-optimal designs
        - analyze_factorial(): Analyze factorial experiments with effects
        - effect_screening(): Screening designs with Pareto of effects
    - TIME SERIES ANALYSIS:
        - arima(): ARIMA modeling with automatic order selection
        - exponential_smoothing(): Simple, double, and Holt-Winters smoothing
        - autocorrelation_analysis(): ACF/PACF with significance bounds
        - seasonal_decomposition(): Decompose into trend, seasonal, residual
        - time_series_forecast(): Unified forecasting interface
        - plot_time_series_diagnostics(): Comprehensive TS diagnostic plots

NEW IN v1.3:
    - train_test_split(): Split data into training and test sets
    - validate_model(): Train on training data, evaluate on both train/test
    - compare_models(): Compare multiple feature sets on train/test
    - compare_all_criteria(): Run all stepwise criteria and compare on test data
    - plot_train_test_comparison(): Visualize train vs test performance
    - plot_model_comparison(): Visualize model comparisons

NEW IN v1.2:
    - covariate_combinations(): Generate all interaction and polynomial terms
    - full_factorial_design(): Create full factorial design matrices  
    - polynomial_features(): sklearn-style polynomial feature generation

NEW IN v1.1:
    - read_csv() / read_excel(): Import data with automatic type detection
    - hat_matrix(): Full influence diagnostics (leverage, Cook's D, DFFITS, DFBETAS)
    - subset_regression(): All possible subsets with multiple criteria
    - stepwise_regression_enhanced(): Multiple stopping rules and directions
    - compare_stepwise_criteria(): Compare all stopping rules side-by-side

FUTURE DEVELOPMENT (TODO):
    Priority additions for upcoming versions:
    
    1. MIXED EFFECTS MODELS:
        - random_effects(): Support for random intercepts and slopes
        - mixed_model(): Linear mixed effects regression (like JMP's Mixed Model)
    
    2. LOGISTIC REGRESSION:
        - logistic_regression(): Binary and multinomial logistic regression
        - ROC curves and AUC calculation
        - Classification metrics (confusion matrix, precision, recall)
    
    3. ENHANCED MODEL SELECTION:
        - cross_validation(): K-fold CV with multiple metrics
        - regularized_regression(): Ridge, Lasso, Elastic Net
        - model_averaging(): Combine predictions from multiple models
    
    4. ADDITIONAL DIAGNOSTICS:
        - vif(): Standalone Variance Inflation Factors function
        - partial_regression_plots(): Added variable plots
        - component_residual_plots(): Partial residual plots

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
    
    # =====================
    # TRAIN/TEST VALIDATION
    # =====================
    
    from jmp_stats import train_test_split, validate_model, compare_models
    
    # Split data
    split = jmp.train_test_split(df['y'], df[predictors], test_size=0.2, random_state=42)
    print(split)  # Shows train/test sizes
    
    # Validate a single model
    results = jmp.validate_model(split.y_train, split.X_train,
                                 split.y_test, split.X_test,
                                 model_name="Full Model")
    print(results)  # Shows train vs test metrics
    
    # Compare multiple models
    models = {
        'Full Model': list(predictors),
        'Stepwise BIC': step_bic.selected_features,
        'Stepwise AIC': step_aic.selected_features,
    }
    comparison = jmp.compare_models(split.y_train, split.X_train,
                                    split.y_test, split.X_test, models)
    print(comparison)
    
    # One-liner: compare all stepwise criteria
    comparison = jmp.compare_all_criteria(df['y'], df[predictors], 
                                          test_size=0.2, random_state=42)
    jmp.plot_model_comparison(comparison)
    
    # =====================
    # STEPWISE REGRESSION
    # =====================
    
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

    # =====================
    # COVARIATE COMBINATIONS
    # =====================
    
    from jmp_stats import covariate_combinations
    
    # Create all pairwise interactions
    result = covariate_combinations(df, ['x1', 'x2', 'x3'])
    print(result)  # Shows x1, x2, x3, x1:x2, x1:x3, x2:x3
    
    # Add polynomial terms (squared)
    result = covariate_combinations(df, ['x1', 'x2'], 
                                    polynomial_degree=2)
    # Creates: x1, x2, x1:x2, x1^2, x2^2
    
    # Three-way interactions with cubic polynomials
    result = covariate_combinations(df, ['a', 'b', 'c'],
                                    max_interaction_order=3,
                                    polynomial_degree=3)
    
    # Use expanded features with stepwise regression
    expanded = covariate_combinations(df, predictors, max_interaction_order=2)
    step = stepwise_regression_enhanced(df['y'], expanded.expanded_df, 
                                        criterion='bic')

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

# sklearn imports for KNN
try:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split as sklearn_train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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
class PredictionIntervalResults:
    """Container for prediction interval results."""
    y_pred: np.ndarray
    lower_pi: np.ndarray
    upper_pi: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    se_pred: np.ndarray
    se_fit: np.ndarray
    alpha: float
    
    def __str__(self):
        n = len(self.y_pred)
        lines = [
            "Prediction Interval Results",
            "=" * 40,
            f"Alpha: {self.alpha} ({(1-self.alpha)*100:.0f}% intervals)",
            f"Number of predictions: {n}",
            "",
            "Predictions:",
            f"{'Obs':>5} {'Predicted':>12} {'PI Lower':>12} {'PI Upper':>12} {'CI Lower':>12} {'CI Upper':>12}"
        ]
        for i in range(min(n, 10)):
            lines.append(f"{i+1:>5} {self.y_pred[i]:>12.2f} {self.lower_pi[i]:>12.2f} {self.upper_pi[i]:>12.2f} {self.lower_ci[i]:>12.2f} {self.upper_ci[i]:>12.2f}")
        if n > 10:
            lines.append(f"... ({n - 10} more rows)")
        return "\n".join(lines)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return results as a DataFrame."""
        return pd.DataFrame({
            'Predicted': self.y_pred,
            'SE_Pred': self.se_pred,
            'SE_Fit': self.se_fit,
            'PI_Lower': self.lower_pi,
            'PI_Upper': self.upper_pi,
            'CI_Lower': self.lower_ci,
            'CI_Upper': self.upper_ci
        })


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
                      alpha: float = 0.05,
                      poly_degree: int = 1,
                      log_y: bool = False,
                      log_X: Optional[List[str]] = None) -> RegressionResults:
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
    poly_degree : int, default 1
        Polynomial degree for predictors. If > 1, automatically generates
        polynomial terms (e.g., poly_degree=2 adds X^2 terms)
    log_y : bool, default False
        If True, log-transform the response variable
    log_X : list of str, optional
        List of column names to log-transform. Only works when X is a DataFrame.
        
    Returns
    -------
    RegressionResults
        Comprehensive regression results object
        
    Example
    -------
    >>> # Basic regression
    >>> results = linear_regression(df['PRSM'], df[['FICO', 'Repayment_Pct']])
    >>> print(results)
    
    >>> # Quadratic regression (auto-generates squared terms)
    >>> results = linear_regression(df['Sales'], df[['Period']], poly_degree=2)
    
    >>> # Log-transformed regression
    >>> results = linear_regression(df['Revenue'], df[['Commission', 'Selling']], 
    ...                             log_y=True, log_X=['Commission', 'Selling'])
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required. Install: pip install statsmodels")
    
    # Handle log transformation of y
    y_orig = y
    y = np.asarray(y).flatten()
    if log_y:
        if np.any(y <= 0):
            raise ValueError("Cannot log-transform y: contains non-positive values")
        y = np.log(y)
    
    # Convert X to DataFrame for easier manipulation
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
        X_names = list(X.columns)
    else:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_names = [f'X{i+1}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=X_names)
    
    # Handle log transformation of specified X columns
    if log_X is not None:
        for col in log_X:
            if col in X_df.columns:
                if np.any(X_df[col] <= 0):
                    # Add small constant to handle zeros
                    X_df[col] = np.log(X_df[col].replace(0, 0.001))
                else:
                    X_df[col] = np.log(X_df[col])
                # Rename column to indicate log transform
                X_df = X_df.rename(columns={col: f'Log_{col}'})
        X_names = list(X_df.columns)
    
    # Handle polynomial terms
    if poly_degree > 1:
        new_cols = {}
        for col in list(X_df.columns):
            for p in range(2, poly_degree + 1):
                new_col_name = f'{col}^{p}' if not col.startswith('Log_') else f'{col}^{p}'
                new_cols[new_col_name] = X_df[col] ** p
        for col_name, col_data in new_cols.items():
            X_df[col_name] = col_data
        X_names = list(X_df.columns)
    
    X = X_df.values
    
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


def linear_regression_formula(formula: str,
                               data: pd.DataFrame,
                               alpha: float = 0.05) -> RegressionResults:
    """
    Perform linear regression using R-style formula with automatic categorical encoding.
    
    This function provides a convenient interface for regression with categorical
    variables, similar to JMP's automatic handling of nominal variables.
    
    Parameters
    ----------
    formula : str
        R-style formula (e.g., 'y ~ x1 + x2 + C(category)')
        Use C(varname) to explicitly treat a variable as categorical.
        Categorical variables are automatically dummy-coded.
    data : DataFrame
        DataFrame containing all variables referenced in the formula
    alpha : float
        Significance level for confidence intervals
        
    Returns
    -------
    RegressionResults
        Comprehensive regression results object
        
    Example
    -------
    >>> # Basic regression with categorical variable
    >>> results = linear_regression_formula(
    ...     'Sales ~ Price + C(Region) + Advertising',
    ...     data=df
    ... )
    
    >>> # With interaction terms
    >>> results = linear_regression_formula(
    ...     'Revenue ~ C(Territory) + C(Gender) + Age + Score',
    ...     data=hiring
    ... )
    
    >>> # Log transform in formula
    >>> results = linear_regression_formula(
    ...     'np.log(Revenue) ~ np.log(Commission) + C(Status)',
    ...     data=df
    ... )
    
    Notes
    -----
    - Categorical variables are automatically dummy-coded with the first level as reference
    - Use Q("Variable Name") for variable names with spaces or special characters
    - Supports numpy functions in formula (e.g., np.log, np.sqrt)
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required. Install: pip install statsmodels")
    
    import statsmodels.formula.api as smf
    
    # Fit model using formula API
    model = smf.ols(formula, data=data).fit()
    
    # Extract results
    n = int(model.nobs)
    predictions = model.predict(data)
    residuals = model.resid
    
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    y_values = model.model.endog
    mape = float(np.mean(np.abs(residuals / y_values)) * 100) if np.all(y_values != 0) else np.nan
    
    # Parse parameter names (excluding Intercept)
    param_names = [p for p in model.params.index if p != 'Intercept']
    
    intercept = float(model.params.get('Intercept', 0))
    intercept_se = float(model.bse.get('Intercept', 0))
    intercept_t = float(model.tvalues.get('Intercept', np.nan))
    intercept_p = float(model.pvalues.get('Intercept', np.nan))
    
    slopes = {name: float(model.params[name]) for name in param_names}
    slope_se = {name: float(model.bse[name]) for name in param_names}
    slope_t = {name: float(model.tvalues[name]) for name in param_names}
    slope_p = {name: float(model.pvalues[name]) for name in param_names}
    
    ci = model.conf_int(alpha)
    conf_int = {'Intercept': (float(ci.loc['Intercept', 0]), float(ci.loc['Intercept', 1]))} if 'Intercept' in ci.index else {}
    for name in param_names:
        conf_int[name] = (float(ci.loc[name, 0]), float(ci.loc[name, 1]))
    
    # VIF calculation for formula models is complex due to categorical expansion
    # Skip for now
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
        residuals=np.array(residuals),
        predicted=np.array(predictions),
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


def prediction_interval(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray, pd.Series],
    X_new: Union[pd.DataFrame, np.ndarray, pd.Series, None] = None,
    alpha: float = 0.05,
    add_constant: bool = True
) -> PredictionIntervalResults:
    """
    Calculate prediction intervals for new observations.
    
    Returns both confidence intervals (for the mean response) and prediction
    intervals (for individual observations). Prediction intervals are always
    wider because they account for irreducible error.
    
    Based on ISLP pg 110: "Prediction intervals will always be wider than
    confidence intervals because they account for the uncertainty associated
    with epsilon, the irreducible error."
    
    Parameters
    ----------
    y : array-like
        Response variable (training data)
    X : DataFrame or array-like
        Predictor variables (training data)
    X_new : DataFrame or array-like, optional
        New X values for prediction. If None, uses training X.
    alpha : float, default 0.05
        Significance level (0.05 = 95% intervals)
    add_constant : bool, default True
        Whether to add intercept term
        
    Returns
    -------
    PredictionIntervalResults
        Contains y_pred, lower_pi, upper_pi, lower_ci, upper_ci, se_pred, se_fit
    
    Example
    -------
    >>> results = prediction_interval(y_train, X_train, X_new)
    >>> print(results.to_dataframe())
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required. Install: pip install statsmodels")
    
    # Convert inputs to numpy arrays
    y = np.asarray(y).flatten()
    
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(X, pd.DataFrame):
        X_names = list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        X_names = [f'X{i+1}' for i in range(X_arr.shape[1])]
    
    # Handle missing values
    mask = ~(np.isnan(y) | np.any(np.isnan(X_arr), axis=1))
    y_clean = y[mask]
    X_clean = X_arr[mask]
    
    # Add constant if requested
    if add_constant:
        X_design = sm.add_constant(X_clean)
    else:
        X_design = X_clean
    
    # Fit the model
    model = sm.OLS(y_clean, X_design).fit()
    
    n = len(y_clean)
    p = X_design.shape[1]  # number of parameters (including intercept)
    df_resid = n - p
    mse = model.mse_resid
    
    # Prepare X_new
    if X_new is None:
        X_new_arr = X_clean.copy()
    else:
        if isinstance(X_new, pd.Series):
            X_new = X_new.to_frame()
        if isinstance(X_new, pd.DataFrame):
            X_new_arr = X_new.values
        else:
            X_new_arr = np.asarray(X_new)
            if X_new_arr.ndim == 1:
                X_new_arr = X_new_arr.reshape(-1, 1)
    
    # Add constant to X_new if needed
    if add_constant:
        X_new_design = sm.add_constant(X_new_arr, has_constant='add')
    else:
        X_new_design = X_new_arr
    
    # Predictions
    y_pred = model.predict(X_new_design)
    
    # Calculate leverage (h) for new observations
    # h = x_new' (X'X)^-1 x_new
    XtX_inv = np.linalg.inv(X_design.T @ X_design)
    h = np.array([x @ XtX_inv @ x.T for x in X_new_design])
    
    # Standard errors
    # SE for fitted value (confidence interval): sqrt(MSE * h)
    # SE for prediction (prediction interval): sqrt(MSE * (1 + h))
    se_fit = np.sqrt(mse * h)
    se_pred = np.sqrt(mse * (1 + h))  # The "1" adds irreducible error variance
    
    # t critical value
    t_crit = stats.t.ppf(1 - alpha/2, df_resid)
    
    # Confidence intervals (for mean response)
    lower_ci = y_pred - t_crit * se_fit
    upper_ci = y_pred + t_crit * se_fit
    
    # Prediction intervals (for individual observations)
    lower_pi = y_pred - t_crit * se_pred
    upper_pi = y_pred + t_crit * se_pred
    
    return PredictionIntervalResults(
        y_pred=y_pred,
        lower_pi=lower_pi,
        upper_pi=upper_pi,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        se_pred=se_pred,
        se_fit=se_fit,
        alpha=alpha
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
# CATEGORICAL VARIABLE ENCODING
# =============================================================================

@dataclass
class EncodedDataFrame:
    """Container for encoded categorical variable results."""
    data: pd.DataFrame  # The encoded DataFrame
    encoding_info: Dict[str, Dict[str, Any]]  # Info about each encoded column
    original_columns: List[str]  # Original column names
    encoded_columns: List[str]  # All columns after encoding
    dummy_columns: List[str]  # Just the new dummy columns
    numeric_columns: List[str]  # Columns that were already numeric
    reference_levels: Dict[str, str]  # Reference level for each categorical
    
    def __str__(self):
        cat_info = []
        for col, info in self.encoding_info.items():
            cat_info.append(f"  {col}: {info['n_levels']} levels -> {info['n_dummies']} dummies (ref: '{info['reference']}')")
        cat_str = '\n'.join(cat_info) if cat_info else '  (none)'
        
        return f"""
Encoded DataFrame Summary
=========================
Original columns:     {len(self.original_columns)}
Encoded columns:      {len(self.encoded_columns)}
Numeric (unchanged):  {len(self.numeric_columns)}
Dummy columns added:  {len(self.dummy_columns)}

Categorical Encodings:
{cat_str}

Use .data to access the encoded DataFrame.
Use .get_dummies_for(col) to get dummy column names for a specific variable.
"""
    
    def get_dummies_for(self, column: str) -> List[str]:
        """Get the dummy column names for a specific categorical variable."""
        if column in self.encoding_info:
            return self.encoding_info[column]['dummy_columns']
        return []
    
    def get_groups(self) -> Dict[str, List[str]]:
        """Get all categorical variable groups (for grouped stepwise selection)."""
        return {col: info['dummy_columns'] for col, info in self.encoding_info.items()}


def encode_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    drop_first: bool = True,
    reference_levels: Optional[Dict[str, str]] = None,
    prefix_sep: str = '_',
    dummy_na: bool = False,
    dtype: type = float,
    verbose: bool = True
) -> EncodedDataFrame:
    """
    Encode categorical variables as dummy variables for regression analysis.
    
    Creates dummy/indicator variables from categorical columns, suitable for
    use in linear regression, stepwise regression, and other modeling functions.
    This follows JMP's approach to handling categorical predictors.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing categorical and/or numeric columns
    columns : list of str, optional
        Specific columns to encode. If None, automatically detects and encodes
        all object and category dtype columns.
    drop_first : bool, default True
        Whether to drop the first (reference) level to avoid multicollinearity.
        Set to False for full dummy encoding (useful for some applications).
    reference_levels : dict, optional
        Specify reference level for specific columns.
        Example: {'Color': 'Red', 'Size': 'Small'}
        Columns not in this dict use the first level alphabetically.
    prefix_sep : str, default '_'
        Separator between column name and level name in dummy column names.
        Example: 'Color_Blue', 'Color_Green'
    dummy_na : bool, default False
        Add a dummy column for missing values. If False, rows with NA in
        categorical columns will have 0 in all dummy columns for that variable.
    dtype : type, default float
        Data type for dummy columns (float or int)
    verbose : bool, default True
        Print summary of encoding
        
    Returns
    -------
    EncodedDataFrame
        Object containing:
        - data: The encoded DataFrame
        - encoding_info: Details about each encoded column
        - Methods for accessing dummy column groups
        
    Examples
    --------
    >>> # Basic usage - auto-detect categorical columns
    >>> encoded = encode_categorical(df)
    >>> print(encoded)  # See summary
    >>> encoded.data  # Access the encoded DataFrame
    
    >>> # Specify which columns to encode
    >>> encoded = encode_categorical(df, columns=['Color', 'Size'])
    
    >>> # Set specific reference levels
    >>> encoded = encode_categorical(df, reference_levels={'Color': 'Red'})
    
    >>> # Use with stepwise regression
    >>> encoded = encode_categorical(df[predictors])
    >>> results = stepwise_regression_enhanced(df['y'], encoded.data)
    
    >>> # Get dummy columns for a specific variable (for grouped selection)
    >>> color_dummies = encoded.get_dummies_for('Color')
    
    Notes
    -----
    - JMP uses effect coding by default, but dummy (indicator) coding is more
      common in Python. This function uses dummy coding.
    - For effect coding, you would need to manually recode (-1, 0, 1 scheme).
    - The reference level (dropped category) represents the baseline.
    - Coefficients for dummy variables represent the difference from the reference.
    
    See Also
    --------
    stepwise_regression_enhanced : Stepwise regression that can use encoded data
    linear_regression : Linear regression with encoded predictors
    """
    df = df.copy()
    
    # Identify columns to encode
    if columns is None:
        # Auto-detect categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        categorical_cols = [c for c in columns if c in df.columns]
    
    # Identify numeric columns (pass through unchanged)
    numeric_cols = [c for c in df.columns if c not in categorical_cols]
    
    if not categorical_cols:
        if verbose:
            print("No categorical columns to encode.")
        return EncodedDataFrame(
            data=df,
            encoding_info={},
            original_columns=list(df.columns),
            encoded_columns=list(df.columns),
            dummy_columns=[],
            numeric_columns=list(df.columns),
            reference_levels={}
        )
    
    reference_levels = reference_levels or {}
    encoding_info = {}
    all_dummy_cols = []
    final_ref_levels = {}
    
    # Start with numeric columns
    result_df = df[numeric_cols].copy()
    
    # Process each categorical column
    for col in categorical_cols:
        # Get unique levels (excluding NaN unless dummy_na)
        levels = df[col].dropna().unique()
        levels = sorted([str(l) for l in levels])  # Sort alphabetically
        
        # Determine reference level
        if col in reference_levels:
            ref_level = str(reference_levels[col])
            if ref_level not in levels:
                warnings.warn(f"Reference level '{ref_level}' not found in column '{col}'. "
                            f"Using first level '{levels[0]}' instead.")
                ref_level = levels[0]
        else:
            ref_level = levels[0]  # First alphabetically
        
        final_ref_levels[col] = ref_level
        
        # Reorder levels so reference is first (will be dropped)
        levels_ordered = [ref_level] + [l for l in levels if l != ref_level]
        
        # Convert column to categorical with ordered levels
        df[col] = pd.Categorical(df[col].astype(str).replace('nan', np.nan), 
                                  categories=levels_ordered)
        
        # Create dummies
        dummies = pd.get_dummies(
            df[col], 
            prefix=col, 
            prefix_sep=prefix_sep,
            drop_first=drop_first,
            dummy_na=dummy_na,
            dtype=dtype
        )
        
        dummy_col_names = list(dummies.columns)
        all_dummy_cols.extend(dummy_col_names)
        
        # Store encoding info
        encoding_info[col] = {
            'levels': levels_ordered,
            'n_levels': len(levels_ordered),
            'reference': ref_level,
            'drop_first': drop_first,
            'dummy_columns': dummy_col_names,
            'n_dummies': len(dummy_col_names)
        }
        
        # Add dummies to result
        result_df = pd.concat([result_df, dummies], axis=1)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Categorical Encoding Summary")
        print(f"{'='*60}")
        print(f"Original columns: {len(df.columns)}")
        print(f"Categorical columns encoded: {len(categorical_cols)}")
        print(f"Numeric columns (unchanged): {len(numeric_cols)}")
        print(f"Total columns after encoding: {len(result_df.columns)}")
        print(f"\nEncoding Details:")
        for col, info in encoding_info.items():
            print(f"  {col}:")
            print(f"    Levels: {info['levels']}")
            print(f"    Reference (dropped): '{info['reference']}'")
            print(f"    Dummy columns: {info['dummy_columns']}")
        print(f"{'='*60}\n")
    
    return EncodedDataFrame(
        data=result_df,
        encoding_info=encoding_info,
        original_columns=list(df.columns),
        encoded_columns=list(result_df.columns),
        dummy_columns=all_dummy_cols,
        numeric_columns=numeric_cols,
        reference_levels=final_ref_levels
    )


def encode_effect(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    reference_levels: Optional[Dict[str, str]] = None,
    prefix_sep: str = '_',
    verbose: bool = True
) -> EncodedDataFrame:
    """
    Encode categorical variables using effect coding (JMP's default).
    
    This is JMP's default parameterization for categorical variables in Fit Model.
    Effect coding uses -1, 0, 1 values where:
    - Each non-reference level gets 1 in its column, 0 in others
    - The reference (last) level gets -1 in ALL columns
    
    This coding centers effects around the grand mean, making coefficients
    represent deviations from the overall average rather than differences
    from a baseline group.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing categorical columns
    columns : list of str, optional
        Specific columns to encode. If None, auto-detects categorical columns.
    reference_levels : dict, optional
        Specify reference level for specific columns. By default, the last
        level (alphabetically) is used as reference, matching JMP's behavior.
    prefix_sep : str, default '_'
        Separator between column name and level name
    verbose : bool, default True
        Print summary
        
    Returns
    -------
    EncodedDataFrame
        Object containing the effect-coded DataFrame and encoding information
        
    Notes
    -----
    JMP Effect Coding Example (3-level factor: A, B, C):
        Level A: (1, 0)   - coded as 1 in first column, 0 in second
        Level B: (0, 1)   - coded as 0 in first column, 1 in second
        Level C: (-1, -1) - reference level, coded as -1 in ALL columns
    
    Coefficient Interpretation:
    - Coefficients represent deviation from the grand mean
    - Sum of effects across all levels = 0
    - Reference level effect = -(sum of other level effects)
    - Intercept represents the grand mean (not reference group mean)
    
    Compare to dummy/indicator coding (encode_categorical with drop_first=True):
    - Coefficients represent difference from reference level
    - Reference level is the baseline (coded as 0 in all columns)
    - Intercept represents the reference group mean
    
    When to use effect coding (JMP default):
    - ANOVA-style analysis where you want effects relative to grand mean
    - Balanced experimental designs
    - When you want coefficients to sum to zero
    
    When to use dummy coding:
    - When you have a meaningful reference/control group
    - When you want to interpret coefficients as differences from baseline
    - More common in epidemiology, social sciences
    
    Examples
    --------
    >>> # JMP-style effect coding for ANOVA
    >>> encoded = encode_effect(df, columns=['Treatment'])
    >>> results = linear_regression(df['y'], encoded.data)
    
    >>> # Specify which level should be the reference
    >>> encoded = encode_effect(df, reference_levels={'Treatment': 'Control'})
    
    See Also
    --------
    encode_categorical : Dummy/indicator coding (0, 1)
    encode_jmp : Alias for encode_effect (JMP's default coding)
    """
    df = df.copy()
    
    # Identify columns to encode
    if columns is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        categorical_cols = [c for c in columns if c in df.columns]
    
    numeric_cols = [c for c in df.columns if c not in categorical_cols]
    
    if not categorical_cols:
        if verbose:
            print("No categorical columns to encode.")
        return EncodedDataFrame(
            data=df,
            encoding_info={},
            original_columns=list(df.columns),
            encoded_columns=list(df.columns),
            dummy_columns=[],
            numeric_columns=list(df.columns),
            reference_levels={}
        )
    
    reference_levels = reference_levels or {}
    encoding_info = {}
    all_effect_cols = []
    final_ref_levels = {}
    
    result_df = df[numeric_cols].copy()
    
    for col in categorical_cols:
        levels = sorted(df[col].dropna().unique().astype(str).tolist())
        
        if col in reference_levels:
            ref_level = str(reference_levels[col])
            if ref_level not in levels:
                ref_level = levels[-1]  # Last level as reference for effect coding
        else:
            ref_level = levels[-1]  # Last level as reference (common convention)
        
        final_ref_levels[col] = ref_level
        non_ref_levels = [l for l in levels if l != ref_level]
        
        effect_col_names = []
        for level in non_ref_levels:
            effect_col = f"{col}{prefix_sep}{level}"
            effect_col_names.append(effect_col)
            
            # Create effect coded column
            result_df[effect_col] = df[col].apply(
                lambda x: 1.0 if str(x) == level else (-1.0 if str(x) == ref_level else 0.0)
            )
        
        all_effect_cols.extend(effect_col_names)
        
        encoding_info[col] = {
            'levels': levels,
            'n_levels': len(levels),
            'reference': ref_level,
            'coding': 'effect',
            'dummy_columns': effect_col_names,
            'n_dummies': len(effect_col_names)
        }
    
    if verbose:
        print(f"\n{'='*60}")
        print("Effect Coding Summary")
        print(f"{'='*60}")
        print(f"Categorical columns encoded: {len(categorical_cols)}")
        print(f"Total columns after encoding: {len(result_df.columns)}")
        print(f"\nEncoding Details (Effect Coding: -1, 0, 1):")
        for col, info in encoding_info.items():
            print(f"  {col}:")
            print(f"    Levels: {info['levels']}")
            print(f"    Reference (coded as -1): '{info['reference']}'")
            print(f"    Effect columns: {info['dummy_columns']}")
        print(f"{'='*60}\n")
    
    return EncodedDataFrame(
        data=result_df,
        encoding_info=encoding_info,
        original_columns=list(df.columns),
        encoded_columns=list(result_df.columns),
        dummy_columns=all_effect_cols,
        numeric_columns=numeric_cols,
        reference_levels=final_ref_levels
    )


def get_categorical_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify categorical vs numeric columns in a DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame
        
    Returns
    -------
    dict
        Dictionary with keys 'categorical' and 'numeric', each containing
        a list of column names
        
    Examples
    --------
    >>> col_types = get_categorical_columns(df)
    >>> print(f"Categorical: {col_types['categorical']}")
    >>> print(f"Numeric: {col_types['numeric']}")
    """
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    other = [c for c in df.columns if c not in categorical and c not in numeric]
    
    return {
        'categorical': categorical,
        'numeric': numeric,
        'other': other
    }


# Alias for JMP's default categorical encoding
encode_jmp = encode_effect
"""Alias for encode_effect() - JMP's default categorical variable parameterization."""


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
# COVARIATE COMBINATIONS FOR REGRESSION MODELING
# =============================================================================

@dataclass
class CovariateCombinations:
    """Container for covariate combination results."""
    original_columns: List[str]
    interaction_columns: List[str]
    polynomial_columns: List[str]
    all_new_columns: List[str]
    expanded_df: pd.DataFrame
    n_original: int
    n_interactions: int
    n_polynomial: int
    n_total: int
    
    def __str__(self):
        return f"""
Covariate Combinations
======================
Original Variables:      {self.n_original}
Interaction Terms:       {self.n_interactions}
Polynomial Terms:        {self.n_polynomial}
Total New Columns:       {len(self.all_new_columns)}
Total Columns:           {self.n_total}

Original: {', '.join(self.original_columns[:5])}{'...' if len(self.original_columns) > 5 else ''}
Interactions: {', '.join(self.interaction_columns[:5])}{'...' if len(self.interaction_columns) > 5 else ''}
Polynomial: {', '.join(self.polynomial_columns[:5])}{'...' if len(self.polynomial_columns) > 5 else ''}
"""

    def get_terms_up_to_order(self, order: int) -> List[str]:
        """Get all term names up to specified interaction order."""
        terms = list(self.original_columns)
        for col in self.interaction_columns:
            n_vars = col.count(':') + 1
            if n_vars <= order:
                terms.append(col)
        return terms


def covariate_combinations(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_interaction_order: int = 2,
    polynomial_degree: int = 1,
    include_interactions: bool = True,
    include_polynomials: bool = True,
    interaction_sep: str = ':',
    polynomial_sep: str = '^',
    exclude_self_interactions: bool = True,
    keep_original: bool = True,
    numeric_only: bool = True
) -> CovariateCombinations:
    """
    Generate all covariate combinations for regression modeling.
    
    Creates interaction terms (x1*x2, x1*x2*x3, etc.) and polynomial terms
    (x^2, x^3, etc.) from the specified columns, similar to JMP's 
    "Construct Model Effects" feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the predictor variables.
    columns : list of str, optional
        Columns to use for combinations. If None, uses all numeric columns.
    max_interaction_order : int, default=2
        Maximum order of interactions to generate.
        - 2: pairwise interactions (x1*x2)
        - 3: three-way interactions (x1*x2*x3)
        - etc.
    polynomial_degree : int, default=1
        Maximum polynomial degree. 
        - 1: no polynomial terms (just original)
        - 2: squared terms (x^2)
        - 3: squared and cubic (x^2, x^3)
    include_interactions : bool, default=True
        Whether to generate interaction terms.
    include_polynomials : bool, default=True
        Whether to generate polynomial terms (degree > 1).
    interaction_sep : str, default=':'
        Separator for interaction term names (e.g., 'x1:x2').
    polynomial_sep : str, default='^'
        Separator for polynomial term names (e.g., 'x^2').
    exclude_self_interactions : bool, default=True
        If True, excludes interactions of a variable with itself (x1:x1).
        Polynomial terms handle powers separately.
    keep_original : bool, default=True
        Whether to include original columns in the output DataFrame.
    numeric_only : bool, default=True
        If True and columns is None, only uses numeric columns.
        
    Returns
    -------
    CovariateCombinations
        Dataclass containing:
        - original_columns: List of original column names
        - interaction_columns: List of generated interaction column names
        - polynomial_columns: List of generated polynomial column names
        - all_new_columns: List of all newly created column names
        - expanded_df: DataFrame with all original and new columns
        - n_original, n_interactions, n_polynomial, n_total: Counts
        
    Examples
    --------
    >>> # Basic usage - pairwise interactions
    >>> result = covariate_combinations(df, ['x1', 'x2', 'x3'])
    >>> print(result.expanded_df.columns.tolist())
    ['x1', 'x2', 'x3', 'x1:x2', 'x1:x3', 'x2:x3']
    
    >>> # With polynomial terms
    >>> result = covariate_combinations(df, ['x1', 'x2'], polynomial_degree=2)
    >>> print(result.expanded_df.columns.tolist())
    ['x1', 'x2', 'x1:x2', 'x1^2', 'x2^2']
    
    >>> # Three-way interactions with cubic polynomials
    >>> result = covariate_combinations(df, ['a', 'b', 'c'], 
    ...                                  max_interaction_order=3, 
    ...                                  polynomial_degree=3)
    
    >>> # Use with linear regression
    >>> result = covariate_combinations(df, ['x1', 'x2', 'x3'])
    >>> model = linear_regression(df['y'], result.expanded_df)
    
    >>> # Use with stepwise regression
    >>> result = covariate_combinations(df, predictors, max_interaction_order=2)
    >>> step = stepwise_regression_enhanced(df['y'], result.expanded_df)
    
    Notes
    -----
    - For k variables with max_interaction_order=2: generates k*(k-1)/2 interactions
    - For k variables with polynomial_degree=d: generates k*(d-1) polynomial terms
    - Warning: High orders with many variables can create very large feature spaces
    - Consider using subset_regression or stepwise_regression to select terms
    
    See Also
    --------
    linear_regression : Fit linear regression model
    stepwise_regression_enhanced : Variable selection with various criteria
    subset_regression : All possible subsets regression
    """
    from itertools import combinations
    
    # Determine columns to use
    if columns is None:
        if numeric_only:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            columns = df.columns.tolist()
    
    # Validate columns exist
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    # Validate parameters
    if max_interaction_order < 1:
        raise ValueError("max_interaction_order must be >= 1")
    if polynomial_degree < 1:
        raise ValueError("polynomial_degree must be >= 1")
    
    # Start with original columns
    expanded_data = {}
    if keep_original:
        for col in columns:
            expanded_data[col] = df[col].values.copy()
    
    interaction_columns = []
    polynomial_columns = []
    
    # Generate interaction terms
    if include_interactions and max_interaction_order >= 2:
        for order in range(2, max_interaction_order + 1):
            for combo in combinations(columns, order):
                # Skip if exclude_self_interactions and any column repeats
                # (though combinations already handles this)
                if exclude_self_interactions and len(set(combo)) < len(combo):
                    continue
                    
                # Create interaction term name
                term_name = interaction_sep.join(combo)
                
                # Calculate interaction value (product of all columns)
                interaction_value = np.ones(len(df))
                for col in combo:
                    interaction_value = interaction_value * df[col].values
                
                expanded_data[term_name] = interaction_value
                interaction_columns.append(term_name)
    
    # Generate polynomial terms
    if include_polynomials and polynomial_degree >= 2:
        for col in columns:
            for degree in range(2, polynomial_degree + 1):
                term_name = f"{col}{polynomial_sep}{degree}"
                expanded_data[term_name] = df[col].values ** degree
                polynomial_columns.append(term_name)
    
    # Create expanded DataFrame
    expanded_df = pd.DataFrame(expanded_data, index=df.index)
    
    # Compile all new columns
    all_new_columns = interaction_columns + polynomial_columns
    
    # Warn if feature space is very large
    if len(all_new_columns) > 100:
        warnings.warn(
            f"Generated {len(all_new_columns)} new columns. Consider using "
            "stepwise_regression or subset_regression for variable selection.",
            UserWarning
        )
    
    return CovariateCombinations(
        original_columns=columns,
        interaction_columns=interaction_columns,
        polynomial_columns=polynomial_columns,
        all_new_columns=all_new_columns,
        expanded_df=expanded_df,
        n_original=len(columns),
        n_interactions=len(interaction_columns),
        n_polynomial=len(polynomial_columns),
        n_total=len(expanded_df.columns)
    )


def full_factorial_design(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    include_intercept: bool = False
) -> pd.DataFrame:
    """
    Generate a full factorial design matrix with all interaction terms.
    
    This is a convenience wrapper around covariate_combinations that generates
    all possible interactions up to the maximum order (k-way for k variables).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Columns to include. If None, uses all numeric columns.
    include_intercept : bool, default=False
        Whether to include an intercept column of ones.
        
    Returns
    -------
    pd.DataFrame
        Design matrix with all main effects and interactions.
        
    Example
    -------
    >>> design = full_factorial_design(df, ['A', 'B', 'C'])
    >>> # Returns: A, B, C, A:B, A:C, B:C, A:B:C
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    result = covariate_combinations(
        df,
        columns=columns,
        max_interaction_order=len(columns),
        polynomial_degree=1,
        include_polynomials=False
    )
    
    design_df = result.expanded_df.copy()
    
    if include_intercept:
        design_df.insert(0, 'Intercept', 1)
    
    return design_df


def polynomial_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    degree: int = 2,
    include_interactions: bool = True,
    include_bias: bool = False
) -> pd.DataFrame:
    """
    Generate polynomial features similar to sklearn's PolynomialFeatures.
    
    Creates polynomial terms and optionally interaction terms for regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Columns to transform. If None, uses all numeric columns.
    degree : int, default=2
        Maximum polynomial degree.
    include_interactions : bool, default=True
        Whether to include interaction terms.
    include_bias : bool, default=False
        Whether to include a bias/intercept column of ones.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with polynomial and interaction features.
        
    Example
    -------
    >>> poly_df = polynomial_features(df, ['x1', 'x2'], degree=2)
    >>> # Returns: x1, x2, x1^2, x2^2, x1:x2
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    result = covariate_combinations(
        df,
        columns=columns,
        max_interaction_order=degree if include_interactions else 1,
        polynomial_degree=degree,
        include_interactions=include_interactions,
        include_polynomials=True
    )
    
    poly_df = result.expanded_df.copy()
    
    if include_bias:
        poly_df.insert(0, 'bias', 1)
    
    return poly_df


# =============================================================================
# TRAIN/TEST SPLIT AND MODEL VALIDATION
# =============================================================================

@dataclass
class TrainTestSplit:
    """Container for train/test split data."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_size: int
    test_size: int
    test_ratio: float
    
    def __str__(self):
        return f"""
Train/Test Split
================
Total Observations:  {self.train_size + self.test_size}
Training Set:        {self.train_size} ({100*(1-self.test_ratio):.1f}%)
Test Set:            {self.test_size} ({100*self.test_ratio:.1f}%)
Features:            {len(self.X_train.columns)}
"""


@dataclass
class ModelValidationResults:
    """
    Container for model validation results comparing train vs test performance.
    
    Matches JMP's validation metrics terminology:
    - RSquare Training: R² on training data
    - RSquare Validation: R² on validation/test data (can be negative if model overfits)
    - RASE: Root Average Squared Error (same as RMSE)
    
    Note: Per JMP documentation, "It is possible for RSquare Validation to be negative."
    This occurs when the model predictions are worse than simply predicting the mean,
    indicating severe overfitting.
    """
    model_name: str
    selected_features: List[str]
    
    # Training metrics (JMP: "Training" column)
    train_r_squared: float
    train_adj_r_squared: float
    train_rmse: float  # JMP calls this RASE (Root Average Squared Error)
    train_mae: float
    train_mape: float
    
    # Validation/Test metrics (JMP: "Validation" column)
    test_r_squared: float  # JMP: "RSquare Validation" - CAN BE NEGATIVE
    test_adj_r_squared: float
    test_rmse: float  # JMP: "RASE Validation"
    test_mae: float
    test_mape: float
    
    # Comparison metrics
    r_squared_diff: float
    rmse_diff: float
    rmse_pct_increase: float
    overfit_warning: bool
    
    # Stored data for plotting
    train_actual: np.ndarray = field(default_factory=lambda: np.array([]))
    train_predicted: np.ndarray = field(default_factory=lambda: np.array([]))
    test_actual: np.ndarray = field(default_factory=lambda: np.array([]))
    test_predicted: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Full model object
    train_model: Optional[Any] = None
    
    # JMP-style aliases
    @property
    def rsquare_training(self) -> float:
        """JMP terminology: RSquare Training"""
        return self.train_r_squared
    
    @property
    def rsquare_validation(self) -> float:
        """JMP terminology: RSquare Validation (can be negative if overfit)"""
        return self.test_r_squared
    
    @property
    def rase_training(self) -> float:
        """JMP terminology: RASE Training (Root Average Squared Error)"""
        return self.train_rmse
    
    @property
    def rase_validation(self) -> float:
        """JMP terminology: RASE Validation"""
        return self.test_rmse
    
    def __str__(self):
        # Determine overfit severity
        if self.test_r_squared < 0:
            overfit_msg = "⚠️  SEVERE OVERFITTING (Validation R² < 0: model worse than mean)"
        elif self.overfit_warning:
            overfit_msg = "⚠️  POSSIBLE OVERFITTING"
        else:
            overfit_msg = "✓ Model generalizes well"
        
        return f"""
Model Validation Results: {self.model_name}
{'=' * (27 + len(self.model_name))}
Features: {len(self.selected_features)}
{', '.join(self.selected_features[:5])}{'...' if len(self.selected_features) > 5 else ''}

                    Training        Validation    Difference
                    --------        ----------    ----------
RSquare             {self.train_r_squared:8.4f}        {self.test_r_squared:8.4f}        {self.r_squared_diff:+8.4f}
RSquare Adj         {self.train_adj_r_squared:8.4f}        {self.test_adj_r_squared:8.4f}
RASE                {self.train_rmse:8.6f}      {self.test_rmse:8.6f}      {self.rmse_diff:+8.6f}
MAE                 {self.train_mae:8.6f}      {self.test_mae:8.6f}

{overfit_msg}

Note: Negative Validation RSquare indicates the model performs worse than
predicting the mean - a sign of overfitting. Consider reducing model complexity.
"""

    def to_dataframe(self) -> pd.DataFrame:
        """Return metrics as a DataFrame (JMP-style column names)."""
        return pd.DataFrame({
            'Metric': ['RSquare', 'RSquare Adj', 'RASE', 'MAE'],
            'Training': [self.train_r_squared, self.train_adj_r_squared, 
                        self.train_rmse, self.train_mae],
            'Validation': [self.test_r_squared, self.test_adj_r_squared,
                          self.test_rmse, self.test_mae],
            'Difference': [self.r_squared_diff, np.nan, self.rmse_diff, np.nan]
        })


@dataclass
class ModelComparisonResults:
    """Container for comparing multiple models on train/validation data (JMP-style)."""
    model_results: List[ModelValidationResults]
    comparison_df: pd.DataFrame
    best_model_by_test_rmse: str  # JMP: best by RASE Validation
    best_model_by_test_r2: str    # JMP: best by RSquare Validation
    
    def __str__(self):
        return f"""
Model Comparison Summary (JMP-Style)
====================================
Models Compared: {len(self.model_results)}
Best by RASE Validation: {self.best_model_by_test_rmse}
Best by RSquare Validation: {self.best_model_by_test_r2}

Note: Negative RSquare Validation indicates overfitting (model worse than mean).
Rule of thumb: Need 10-20 observations per feature to avoid overfitting.

{self.comparison_df.to_string()}
"""


def train_test_split(
    y: Union[pd.Series, np.ndarray],
    X: pd.DataFrame,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: bool = False
) -> TrainTestSplit:
    """
    Split data into training and test sets.
    
    Parameters
    ----------
    y : array-like
        Response/target variable.
    X : DataFrame
        Predictor variables.
    test_size : float, default=0.2
        Proportion of data to use for testing (0.0 to 1.0).
    random_state : int, optional
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    stratify : bool, default=False
        If True, preserves the distribution of y in both sets.
        Only works for categorical/discrete y.
        
    Returns
    -------
    TrainTestSplit
        Dataclass containing X_train, X_test, y_train, y_test and metadata.
        
    Example
    -------
    >>> split = train_test_split(df['y'], df[predictors], test_size=0.2, random_state=42)
    >>> print(split)
    >>> model = linear_regression(split.y_train, split.X_train)
    """
    # Convert to proper types
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    # Handle missing values
    valid_idx = ~(y.isna() | X.isna().any(axis=1))
    y_clean = y[valid_idx].reset_index(drop=True)
    X_clean = X[valid_idx].reset_index(drop=True)
    
    n = len(y_clean)
    n_test = int(n * test_size)
    n_train = n - n_test
    
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(n)
    
    if shuffle:
        if stratify and y_clean.nunique() < 20:  # Only stratify for discrete y
            # Stratified split
            test_indices = []
            for val in y_clean.unique():
                val_indices = indices[y_clean == val]
                n_val_test = max(1, int(len(val_indices) * test_size))
                test_indices.extend(np.random.choice(val_indices, n_val_test, replace=False))
            test_indices = np.array(test_indices)
            train_indices = np.setdiff1d(indices, test_indices)
        else:
            np.random.shuffle(indices)
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
    else:
        # No shuffle - use last portion as test (common for time series)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
    
    return TrainTestSplit(
        X_train=X_clean.iloc[train_indices].reset_index(drop=True),
        X_test=X_clean.iloc[test_indices].reset_index(drop=True),
        y_train=y_clean.iloc[train_indices].reset_index(drop=True),
        y_test=y_clean.iloc[test_indices].reset_index(drop=True),
        train_indices=train_indices,
        test_indices=test_indices,
        train_size=len(train_indices),
        test_size=len(test_indices),
        test_ratio=test_size
    )


def make_validation_column(
    df: pd.DataFrame,
    validation_portion: float = 0.2,
    test_portion: float = 0.0,
    random_state: Optional[int] = None,
    method: str = 'random'
) -> pd.DataFrame:
    """
    Create a validation column for the DataFrame (JMP-style).
    
    This mimics JMP's "Make Validation Column" functionality, which creates
    a column that can be used to split data into Training, Validation, and 
    optionally Test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    validation_portion : float, default=0.2
        Proportion of data for validation set (0.0 to 1.0).
    test_portion : float, default=0.0
        Proportion of data for test set (0.0 to 1.0).
        If 0, only Training and Validation sets are created.
    random_state : int, optional
        Random seed for reproducibility.
    method : str, default='random'
        Method for creating validation column:
        - 'random': Random assignment
        - 'sequential': First rows = Training, last rows = Validation/Test
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added 'Validation' column containing:
        - 0 = Training
        - 1 = Validation  
        - 2 = Test (if test_portion > 0)
        
    Example
    -------
    >>> # JMP-style: 70% Train, 20% Validation, 10% Test
    >>> df = make_validation_column(df, validation_portion=0.2, test_portion=0.1)
    >>> 
    >>> # Use with stepwise
    >>> train_df = df[df['Validation'] == 0]
    >>> valid_df = df[df['Validation'] == 1]
    
    Notes
    -----
    JMP's validation column uses:
    - Smallest value = Training set
    - Middle value = Validation set  
    - Largest value = Test set
    
    This function follows that convention with 0, 1, 2.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(df)
    train_portion = 1.0 - validation_portion - test_portion
    
    if train_portion <= 0:
        raise ValueError("Training portion must be > 0. Reduce validation_portion or test_portion.")
    
    result_df = df.copy()
    
    if method == 'random':
        # Random assignment
        rand_vals = np.random.random(n)
        validation_col = np.zeros(n, dtype=int)
        validation_col[rand_vals >= train_portion] = 1
        if test_portion > 0:
            validation_col[rand_vals >= (train_portion + validation_portion)] = 2
    else:
        # Sequential assignment
        validation_col = np.zeros(n, dtype=int)
        n_train = int(n * train_portion)
        n_valid = int(n * validation_portion)
        validation_col[n_train:n_train + n_valid] = 1
        if test_portion > 0:
            validation_col[n_train + n_valid:] = 2
    
    result_df['Validation'] = validation_col
    
    # Print summary like JMP
    counts = pd.Series(validation_col).value_counts().sort_index()
    print(f"""
Validation Column Created
=========================
Training (0):    {counts.get(0, 0):5d} rows ({counts.get(0, 0)/n*100:.1f}%)
Validation (1):  {counts.get(1, 0):5d} rows ({counts.get(1, 0)/n*100:.1f}%)""")
    if test_portion > 0:
        print(f"Test (2):        {counts.get(2, 0):5d} rows ({counts.get(2, 0)/n*100:.1f}%)")
    print(f"Total:           {n:5d} rows")
    
    return result_df


def validate_model(
    y_train: Union[pd.Series, np.ndarray],
    X_train: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    X_test: pd.DataFrame,
    features: Optional[List[str]] = None,
    model_name: str = "Model"
) -> ModelValidationResults:
    """
    Train a model on training data and evaluate on both train and test sets.
    
    Parameters
    ----------
    y_train : array-like
        Training response variable.
    X_train : DataFrame
        Training predictors.
    y_test : array-like
        Test response variable.
    X_test : DataFrame
        Test predictors.
    features : list of str, optional
        Specific features to use. If None, uses all columns in X_train.
    model_name : str, default="Model"
        Name for identifying this model in comparisons.
        
    Returns
    -------
    ModelValidationResults
        Dataclass with train/test metrics and comparison.
        
    Example
    -------
    >>> split = train_test_split(df['y'], df[predictors], test_size=0.2)
    >>> results = validate_model(split.y_train, split.X_train, 
    ...                          split.y_test, split.X_test,
    ...                          model_name="Full Model")
    >>> print(results)
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required for validate_model")
    
    # Select features
    if features is None:
        features = list(X_train.columns)
    
    X_train_subset = X_train[features]
    X_test_subset = X_test[features]
    
    # Convert to arrays
    y_train_arr = np.asarray(y_train).flatten()
    y_test_arr = np.asarray(y_test).flatten()
    
    # Fit model on training data
    X_train_const = sm.add_constant(X_train_subset)
    model = sm.OLS(y_train_arr, X_train_const).fit()
    
    # Training predictions and metrics
    train_pred = model.predict(X_train_const)
    train_resid = y_train_arr - train_pred
    train_rmse = np.sqrt(np.mean(train_resid**2))
    train_mae = np.mean(np.abs(train_resid))
    train_ss_tot = np.sum((y_train_arr - np.mean(y_train_arr))**2)
    train_ss_res = np.sum(train_resid**2)
    train_r2 = 1 - (train_ss_res / train_ss_tot)
    n_train = len(y_train_arr)
    p = len(features)
    train_adj_r2 = 1 - (1 - train_r2) * (n_train - 1) / (n_train - p - 1)
    
    # Avoid division by zero for MAPE
    nonzero_train = y_train_arr != 0
    if np.any(nonzero_train):
        train_mape = np.mean(np.abs(train_resid[nonzero_train] / y_train_arr[nonzero_train])) * 100
    else:
        train_mape = np.nan
    
    # Test predictions and metrics
    X_test_const = sm.add_constant(X_test_subset, has_constant='add')
    test_pred = model.predict(X_test_const)
    test_resid = y_test_arr - test_pred
    test_rmse = np.sqrt(np.mean(test_resid**2))
    test_mae = np.mean(np.abs(test_resid))
    test_ss_tot = np.sum((y_test_arr - np.mean(y_test_arr))**2)
    test_ss_res = np.sum(test_resid**2)
    test_r2 = 1 - (test_ss_res / test_ss_tot)
    n_test = len(y_test_arr)
    test_adj_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - p - 1)
    
    nonzero_test = y_test_arr != 0
    if np.any(nonzero_test):
        test_mape = np.mean(np.abs(test_resid[nonzero_test] / y_test_arr[nonzero_test])) * 100
    else:
        test_mape = np.nan
    
    # Comparison metrics
    r2_diff = test_r2 - train_r2
    rmse_diff = test_rmse - train_rmse
    rmse_pct_increase = (rmse_diff / train_rmse) * 100 if train_rmse > 0 else 0
    
    # Overfit warning if test RMSE is significantly higher than train RMSE
    overfit_warning = rmse_pct_increase > 20 or r2_diff < -0.1
    
    return ModelValidationResults(
        model_name=model_name,
        selected_features=features,
        train_r_squared=float(train_r2),
        train_adj_r_squared=float(train_adj_r2),
        train_rmse=float(train_rmse),
        train_mae=float(train_mae),
        train_mape=float(train_mape),
        test_r_squared=float(test_r2),
        test_adj_r_squared=float(test_adj_r2),
        test_rmse=float(test_rmse),
        test_mae=float(test_mae),
        test_mape=float(test_mape),
        r_squared_diff=float(r2_diff),
        rmse_diff=float(rmse_diff),
        rmse_pct_increase=float(rmse_pct_increase),
        overfit_warning=overfit_warning,
        train_actual=y_train_arr,
        train_predicted=train_pred,
        test_actual=y_test_arr,
        test_predicted=test_pred,
        train_model=model
    )


def compare_models(
    y_train: Union[pd.Series, np.ndarray],
    X_train: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    X_test: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    verbose: bool = True
) -> ModelComparisonResults:
    """
    Compare multiple models (feature sets) on train/test data.
    
    NOTE: When determining the "best" model, any model with a name containing
    "full", "full_model", or "full model" is automatically excluded to prevent
    selecting an overfit model. The full model is still included in the 
    comparison table for reference.
    
    Parameters
    ----------
    y_train : array-like
        Training response variable.
    X_train : DataFrame
        Training predictors (should contain all features from all feature_sets).
    y_test : array-like
        Test response variable.
    X_test : DataFrame
        Test predictors.
    feature_sets : dict
        Dictionary mapping model names to lists of feature names.
        Example: {'Full': ['x1','x2','x3'], 'Reduced': ['x1','x2']}
    verbose : bool, default=True
        Print progress.
        
    Returns
    -------
    ModelComparisonResults
        Dataclass with comparison DataFrame and best models.
        Note: best_model_by_test_rmse and best_model_by_test_r2 will never
        be a "Full Model" variant.
        
    Example
    -------
    >>> split = train_test_split(df['y'], df[predictors])
    >>> models = {
    ...     'Full Model': predictors,
    ...     'Stepwise': step.selected_features,
    ...     'Simple': ['x1', 'x2']
    ... }
    >>> comparison = compare_models(split.y_train, split.X_train,
    ...                             split.y_test, split.X_test, models)
    >>> print(comparison)  # Best model will be Stepwise or Simple, not Full Model
    """
    results = []
    
    for name, features in feature_sets.items():
        if verbose:
            print(f"Evaluating: {name} ({len(features)} features)")
        
        try:
            result = validate_model(
                y_train, X_train, y_test, X_test,
                features=features, model_name=name
            )
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  Error with {name}: {e}")
    
    # Build comparison DataFrame
    comparison_data = []
    for r in results:
        comparison_data.append({
            'Model': r.model_name,
            'N_Features': len(r.selected_features),
            'RSquare_Train': r.train_r_squared,
            'RSquare_Valid': r.test_r_squared,
            'R2_Diff': r.r_squared_diff,
            'RASE_Train': r.train_rmse,
            'RASE_Valid': r.test_rmse,
            'Overfit': '⚠️' if r.overfit_warning or r.test_r_squared < 0 else '✓'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Find best models (excluding Full Model variants)
    # This prevents overfitting by not selecting the model with all predictors
    full_model_patterns = ['full_model', 'full model', 'fullmodel', 'full']
    
    def is_full_model(name):
        """Check if model name indicates it's a full model."""
        name_lower = name.lower().replace('_', ' ').replace('-', ' ')
        return any(pattern in name_lower for pattern in full_model_patterns)
    
    # Filter out full models for best model selection
    non_full_df = comparison_df[~comparison_df['Model'].apply(is_full_model)]
    
    if len(non_full_df) > 0:
        # Select best from non-full models
        best_by_rmse = non_full_df.loc[non_full_df['RASE_Valid'].idxmin(), 'Model']
        best_by_r2 = non_full_df.loc[non_full_df['RSquare_Valid'].idxmax(), 'Model']
    else:
        # Fallback if only full models exist (shouldn't happen normally)
        best_by_rmse = comparison_df.loc[comparison_df['RASE_Valid'].idxmin(), 'Model']
        best_by_r2 = comparison_df.loc[comparison_df['RSquare_Valid'].idxmax(), 'Model']
    
    return ModelComparisonResults(
        model_results=results,
        comparison_df=comparison_df,
        best_model_by_test_rmse=best_by_rmse,
        best_model_by_test_r2=best_by_r2
    )


def compare_all_criteria(
    y: Union[pd.Series, np.ndarray],
    X: pd.DataFrame,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    direction: str = 'both',
    include_full_model: bool = True,
    verbose: bool = True
) -> ModelComparisonResults:
    """
    Run stepwise regression with all criteria and compare on train/test split.
    
    This is a convenience function that:
    1. Splits data into train/test
    2. Runs stepwise with each criterion on training data
    3. Evaluates all models on test data
    4. Returns comparison
    
    NOTE: When selecting the "best" model, the Full Model is automatically
    excluded to prevent overfitting. The Full Model is included in the
    comparison table for reference but will never be selected as best.
    
    Parameters
    ----------
    y : array-like
        Response variable.
    X : DataFrame
        Predictor variables.
    test_size : float, default=0.2
        Proportion for test set.
    random_state : int, optional
        Random seed for reproducibility.
    direction : str, default='both'
        Stepwise direction: 'forward', 'backward', or 'both'.
    include_full_model : bool, default=True
        Whether to include the full model in comparison table.
        Even if True, full model will not be selected as "best".
    verbose : bool, default=True
        Print progress.
        
    Returns
    -------
    ModelComparisonResults
        Comparison of all stepwise criteria models.
        
    Example
    -------
    >>> comparison = compare_all_criteria(df['y'], df[predictors], 
    ...                                   test_size=0.2, random_state=42)
    >>> print(comparison)
    >>> plot_model_comparison(comparison)
    """
    # Split data
    split = train_test_split(y, X, test_size=test_size, random_state=random_state)
    
    if verbose:
        print(split)
        print("Running stepwise with each criterion...")
    
    criteria = ['pvalue', 'aic', 'bic', 'adj_rsq', 'rmse', 'cp']
    feature_sets = {}
    
    for crit in criteria:
        if verbose:
            print(f"  {crit}...", end=" ")
        
        try:
            step_result = stepwise_regression_enhanced(
                split.y_train, split.X_train,
                direction=direction, criterion=crit, verbose=False
            )
            if step_result.selected_features:
                feature_sets[f"Stepwise_{crit.upper()}"] = step_result.selected_features
                if verbose:
                    print(f"selected {len(step_result.selected_features)} features")
            else:
                if verbose:
                    print("no features selected")
        except Exception as e:
            if verbose:
                print(f"error: {e}")
    
    # Add full model for comparison (but it won't be selected as best)
    if include_full_model:
        feature_sets['Full_Model'] = list(X.columns)
    
    if verbose:
        print("\nEvaluating models on test data...")
        if include_full_model:
            print("(Note: Full Model included for reference but excluded from 'best' selection)")
    
    return compare_models(
        split.y_train, split.X_train,
        split.y_test, split.X_test,
        feature_sets, verbose=verbose
    )


def plot_train_test_comparison(
    results: Union[ModelValidationResults, List[ModelValidationResults]],
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Plot train vs test comparison for one or more models.
    
    Creates a multi-panel figure with:
    - Actual vs Predicted scatter plots (train and test)
    - Residual plots
    - Metrics comparison bar chart
    
    Parameters
    ----------
    results : ModelValidationResults or list
        Single result or list of results from validate_model().
    figsize : tuple, default=(14, 10)
        Figure size.
        
    Example
    -------
    >>> results = validate_model(y_train, X_train, y_test, X_test)
    >>> plot_train_test_comparison(results)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    if isinstance(results, ModelValidationResults):
        results = [results]
    
    n_models = len(results)
    
    if n_models == 1:
        # Single model - detailed view
        r = results[0]
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Model Validation: {r.model_name}', fontsize=14, fontweight='bold')
        
        # Train: Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(r.train_actual, r.train_predicted, alpha=0.6, label='Training')
        min_val = min(r.train_actual.min(), r.train_predicted.min())
        max_val = max(r.train_actual.max(), r.train_predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Training: Actual vs Predicted (R²={r.train_r_squared:.4f})')
        ax.legend()
        
        # Test: Actual vs Predicted
        ax = axes[0, 1]
        ax.scatter(r.test_actual, r.test_predicted, alpha=0.6, color='orange', label='Test')
        min_val = min(r.test_actual.min(), r.test_predicted.min())
        max_val = max(r.test_actual.max(), r.test_predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Test: Actual vs Predicted (R²={r.test_r_squared:.4f})')
        ax.legend()
        
        # Train Residuals
        ax = axes[1, 0]
        train_resid = r.train_actual - r.train_predicted
        ax.scatter(r.train_predicted, train_resid, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residual')
        ax.set_title(f'Training Residuals (RMSE={r.train_rmse:.4f})')
        
        # Test Residuals
        ax = axes[1, 1]
        test_resid = r.test_actual - r.test_predicted
        ax.scatter(r.test_predicted, test_resid, alpha=0.6, color='orange')
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residual')
        ax.set_title(f'Test Residuals (RMSE={r.test_rmse:.4f})')
        
    else:
        # Multiple models - comparison view
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Comparison: Train vs Test Performance', fontsize=14, fontweight='bold')
        
        model_names = [r.model_name for r in results]
        x = np.arange(len(model_names))
        width = 0.35
        
        # R-squared comparison
        ax = axes[0, 0]
        train_r2 = [r.train_r_squared for r in results]
        test_r2 = [r.test_r_squared for r in results]
        bars1 = ax.bar(x - width/2, train_r2, width, label='Train', color='steelblue')
        bars2 = ax.bar(x + width/2, test_r2, width, label='Test', color='orange')
        ax.set_ylabel('R-Squared')
        ax.set_title('R-Squared: Train vs Test')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, max(max(train_r2), max(test_r2)) * 1.1)
        
        # RMSE comparison
        ax = axes[0, 1]
        train_rmse = [r.train_rmse for r in results]
        test_rmse = [r.test_rmse for r in results]
        bars1 = ax.bar(x - width/2, train_rmse, width, label='Train', color='steelblue')
        bars2 = ax.bar(x + width/2, test_rmse, width, label='Test', color='orange')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE: Train vs Test')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        
        # RMSE % increase (overfitting indicator)
        ax = axes[1, 0]
        rmse_pct = [r.rmse_pct_increase for r in results]
        colors = ['red' if p > 20 else 'green' for p in rmse_pct]
        ax.bar(x, rmse_pct, color=colors)
        ax.axhline(y=20, color='red', linestyle='--', label='Overfit threshold (20%)')
        ax.set_ylabel('RMSE Increase (%)')
        ax.set_title('Test RMSE Increase (Overfit Indicator)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        
        # Number of features vs Test R²
        ax = axes[1, 1]
        n_features = [len(r.selected_features) for r in results]
        ax.scatter(n_features, test_r2, s=100, c='orange', edgecolors='black')
        for i, name in enumerate(model_names):
            ax.annotate(name, (n_features[i], test_r2[i]), 
                       textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Test R-Squared')
        ax.set_title('Model Complexity vs Test Performance')
    
    plt.tight_layout()
    plt.show()


def plot_model_comparison(comparison: ModelComparisonResults, figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Plot comparison of multiple models from compare_models() or compare_all_criteria().
    
    Parameters
    ----------
    comparison : ModelComparisonResults
        Results from compare_models() or compare_all_criteria().
    figsize : tuple, default=(14, 10)
        Figure size.
        
    Example
    -------
    >>> comparison = compare_all_criteria(df['y'], df[predictors])
    >>> plot_model_comparison(comparison)
    """
    plot_train_test_comparison(comparison.model_results, figsize=figsize)


# =============================================================================
# NEW v2.0: INTERACTIVE LEVERAGE PLOTS AND INFLUENCE DIAGNOSTICS
# =============================================================================

def plot_leverage_interactive(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    highlight_influential: bool = True,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[Any]:
    """
    Create JMP-style interactive leverage plots for each predictor.
    
    Shows the relationship between each predictor and the response,
    controlling for all other predictors (partial regression plots).
    
    Parameters
    ----------
    y : array-like
        Response variable
    X : DataFrame
        Predictor variables
    highlight_influential : bool, default=True
        Highlight high-leverage and high-influence points
    show_labels : bool, default=True
        Show observation labels for influential points
    figsize : tuple, default=(14, 10)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The leverage plots figure
        
    Example
    -------
    >>> plot_leverage_interactive(df['y'], df[['x1', 'x2', 'x3']])
    """
    if not HAS_MATPLOTLIB or not HAS_STATSMODELS:
        print("matplotlib and statsmodels required for leverage plots")
        return None
    
    y = np.asarray(y).flatten()
    
    if isinstance(X, pd.DataFrame):
        X_names = list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        X_names = [f'X{i+1}' for i in range(X_arr.shape[1])]
    
    # Remove missing values
    mask = ~(np.isnan(y) | np.any(np.isnan(X_arr), axis=1))
    y = y[mask]
    X_arr = X_arr[mask]
    n = len(y)
    p = X_arr.shape[1]
    
    # Fit full model for influence diagnostics
    X_const = sm.add_constant(X_arr)
    full_model = sm.OLS(y, X_const).fit()
    influence = full_model.get_influence()
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]
    
    # Thresholds
    leverage_threshold = 2 * (p + 1) / n
    cooks_threshold = 4 / n
    
    # Determine grid size
    n_cols = min(3, p)
    n_rows = int(np.ceil(p / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if p == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, var_name in enumerate(X_names):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # Create partial regression plot (added variable plot)
        other_vars = [i for i in range(p) if i != idx]
        if len(other_vars) > 0:
            X_other = sm.add_constant(X_arr[:, other_vars])
            model_y = sm.OLS(y, X_other).fit()
            resid_y = model_y.resid
            
            model_x = sm.OLS(X_arr[:, idx], X_other).fit()
            resid_x = model_x.resid
        else:
            resid_y = y - np.mean(y)
            resid_x = X_arr[:, idx] - np.mean(X_arr[:, idx])
        
        # Color points by influence
        colors = np.array(['steelblue'] * n)
        sizes = np.array([50.0] * n)
        
        if highlight_influential:
            high_leverage = leverage > leverage_threshold
            high_cooks = cooks_d > cooks_threshold
            
            colors[high_leverage] = 'orange'
            colors[high_cooks] = 'red'
            sizes[high_leverage | high_cooks] = 100
        
        ax.scatter(resid_x, resid_y, c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add regression line
        slope, intercept = np.polyfit(resid_x, resid_y, 1)
        x_line = np.linspace(resid_x.min(), resid_x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-', lw=2)
        
        # Label influential points
        if show_labels and highlight_influential:
            for i in np.where(high_leverage | high_cooks)[0]:
                ax.annotate(str(i), (resid_x[i], resid_y[i]), 
                           fontsize=8, ha='left', va='bottom')
        
        ax.set_xlabel(f'{var_name} | Others')
        ax.set_ylabel('Y | Others')
        ax.set_title(f'Leverage Plot: {var_name}')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Hide unused subplots
    for idx in range(p, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    # Add legend
    if highlight_influential:
        legend_elements = [
            plt.scatter([], [], c='steelblue', s=50, label='Normal'),
            plt.scatter([], [], c='orange', s=100, label='High Leverage'),
            plt.scatter([], [], c='red', s=100, label='High Influence (Cook\'s D)')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    plt.suptitle('Leverage Plots (Partial Regression)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_influence_dashboard(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    figsize: Tuple[int, int] = (16, 12)
) -> Optional[Any]:
    """
    Create comprehensive influence diagnostics dashboard (JMP-style).
    
    Includes:
    - Studentized residuals vs leverage (with Cook's D contours)
    - Cook's D bar chart
    - DFFITS plot
    - Residuals vs fitted with influence sizing
    
    Parameters
    ----------
    y : array-like
        Response variable
    X : DataFrame or array
        Predictor variables
    figsize : tuple, default=(16, 12)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The influence dashboard figure
        
    Example
    -------
    >>> plot_influence_dashboard(df['y'], df[['x1', 'x2', 'x3']])
    """
    if not HAS_MATPLOTLIB or not HAS_STATSMODELS:
        print("matplotlib and statsmodels required")
        return None
    
    y = np.asarray(y).flatten()
    
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
    
    mask = ~(np.isnan(y) | np.any(np.isnan(X_arr), axis=1))
    y = y[mask]
    X_arr = X_arr[mask]
    n = len(y)
    p = X_arr.shape[1]
    
    X_const = sm.add_constant(X_arr)
    model = sm.OLS(y, X_const).fit()
    influence = model.get_influence()
    
    leverage = influence.hat_matrix_diag
    stud_resid = influence.resid_studentized_external
    cooks_d = influence.cooks_distance[0]
    dffits = influence.dffits[0]
    predicted = model.predict(X_const)
    
    # Thresholds
    leverage_threshold = 2 * (p + 1) / n
    cooks_threshold = 4 / n
    dffits_threshold = 2 * np.sqrt((p + 1) / n)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Studentized Residuals vs Leverage
    ax = axes[0, 0]
    
    h_range = np.linspace(0.001, leverage.max() * 1.1, 100)
    for d_val in [0.5, 1.0]:
        r_pos = np.sqrt(d_val * (p + 1) * (1 - h_range) / h_range)
        r_neg = -r_pos
        valid = ~np.isnan(r_pos) & ~np.isinf(r_pos)
        ax.plot(h_range[valid], r_pos[valid], 'r--', alpha=0.5)
        ax.plot(h_range[valid], r_neg[valid], 'r--', alpha=0.5)
    
    colors = np.where(cooks_d > cooks_threshold, 'red', 
                     np.where(leverage > leverage_threshold, 'orange', 'steelblue'))
    
    ax.scatter(leverage, stud_resid, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.axhline(y=-2, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=leverage_threshold, color='orange', linestyle='--', alpha=0.5, label='Leverage threshold')
    ax.set_xlabel('Leverage (Hat Value)')
    ax.set_ylabel('Studentized Residual')
    ax.set_title('Influence Plot: Residuals vs Leverage')
    ax.legend(loc='best')
    
    top_influence = np.argsort(cooks_d)[-5:]
    for i in top_influence:
        if cooks_d[i] > cooks_threshold:
            ax.annotate(str(i), (leverage[i], stud_resid[i]), fontsize=8)
    
    # Plot 2: Cook's D Bar Chart
    ax = axes[0, 1]
    indices = np.arange(n)
    colors = np.where(cooks_d > cooks_threshold, 'red', 'steelblue')
    ax.bar(indices, cooks_d, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axhline(y=cooks_threshold, color='red', linestyle='--', label=f'Threshold (4/n = {cooks_threshold:.4f})')
    ax.axhline(y=1.0, color='darkred', linestyle='-', alpha=0.5, label='Cook\'s D = 1')
    ax.set_xlabel('Observation Index')
    ax.set_ylabel("Cook's Distance")
    ax.set_title("Cook's Distance by Observation")
    ax.legend(loc='best')
    
    # Plot 3: DFFITS
    ax = axes[1, 0]
    colors = np.where(np.abs(dffits) > dffits_threshold, 'red', 'steelblue')
    ax.scatter(indices, dffits, c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.axhline(y=dffits_threshold, color='red', linestyle='--', label=f'Threshold (±{dffits_threshold:.3f})')
    ax.axhline(y=-dffits_threshold, color='red', linestyle='--')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Observation Index')
    ax.set_ylabel('DFFITS')
    ax.set_title('DFFITS (Standardized Difference in Fits)')
    ax.legend(loc='best')
    
    # Plot 4: Residuals vs Fitted
    ax = axes[1, 1]
    residuals = model.resid
    
    sizes = 50 + 500 * (cooks_d / (cooks_d.max() + 1e-10))
    colors = np.where(cooks_d > cooks_threshold, 'red',
                     np.where(leverage > leverage_threshold, 'orange', 'steelblue'))
    
    ax.scatter(predicted, residuals, c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--')
    
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, predicted, frac=0.5)
        ax.plot(smoothed[:, 0], smoothed[:, 1], 'g-', lw=2, label='LOWESS')
    except:
        pass
    
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted (size = Cook\'s D)')
    
    plt.suptitle('Influence Diagnostics Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_partial_regression(
    y: Union[pd.Series, np.ndarray],
    X: pd.DataFrame,
    variable: str,
    figsize: Tuple[int, int] = (10, 8)
) -> Optional[Any]:
    """Create partial regression plot for a single variable."""
    if not HAS_MATPLOTLIB or not HAS_STATSMODELS:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    sm.graphics.plot_partregress(endog=y, exog_i=variable, exog_others=X.drop(columns=[variable]),
                                  obs_labels=False, ax=ax)
    ax.set_title(f'Partial Regression Plot: {variable}')
    plt.tight_layout()
    return fig


def plot_added_variable(
    y: Union[pd.Series, np.ndarray],
    X: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[Any]:
    """Create added variable plots for all predictors (JMP-style)."""
    if not HAS_MATPLOTLIB or not HAS_STATSMODELS:
        return None
    
    fig = sm.graphics.plot_partregress_grid(
        sm.OLS(y, sm.add_constant(X)).fit(),
        fig=plt.figure(figsize=figsize)
    )
    plt.suptitle('Added Variable Plots (Partial Regression)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_component_residual(
    y: Union[pd.Series, np.ndarray],
    X: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[Any]:
    """Create component-plus-residual plots (CCPR plots) for all predictors."""
    if not HAS_MATPLOTLIB or not HAS_STATSMODELS:
        return None
    
    model = sm.OLS(y, sm.add_constant(X)).fit()
    fig = sm.graphics.plot_ccpr_grid(model, fig=plt.figure(figsize=figsize))
    plt.suptitle('Component-Plus-Residual Plots', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# NEW v2.0: PREDICTION PROFILER
# =============================================================================

@dataclass
class PredictionProfiler:
    """Container for prediction profiler results."""
    model: Any
    X_names: List[str]
    X_ranges: Dict[str, Tuple[float, float]]
    X_current: Dict[str, float]
    y_name: str
    prediction: float
    conf_interval: Tuple[float, float]
    pred_interval: Tuple[float, float]
    
    def predict_at(self, **kwargs) -> Dict[str, Any]:
        """Make prediction at specified values."""
        X_vals = self.X_current.copy()
        X_vals.update(kwargs)
        
        X_arr = np.array([[X_vals[name] for name in self.X_names]])
        X_const = sm.add_constant(X_arr, has_constant='add')
        
        pred = self.model.get_prediction(X_const)
        
        return {
            'prediction': float(pred.predicted_mean[0]),
            'conf_interval': tuple(pred.conf_int(alpha=0.05)[0]),
            'pred_interval': tuple(pred.conf_int(obs=True, alpha=0.05)[0]),
            'x_values': X_vals
        }
    
    def __str__(self):
        x_str = ', '.join([f'{k}={v:.4f}' for k, v in self.X_current.items()])
        return f"""
Prediction Profiler
===================
Response: {self.y_name}
Predictors: {', '.join(self.X_names)}

Current Settings:
  {x_str}

Prediction:        {self.prediction:.6f}
95% Confidence:    [{self.conf_interval[0]:.6f}, {self.conf_interval[1]:.6f}]
95% Prediction:    [{self.pred_interval[0]:.6f}, {self.pred_interval[1]:.6f}]
"""


def prediction_profiler(
    y: Union[pd.Series, np.ndarray],
    X: pd.DataFrame,
    y_name: str = 'Y',
    initial_values: Optional[Dict[str, float]] = None
) -> PredictionProfiler:
    """
    Create a JMP-style prediction profiler for exploring model predictions.
    
    Parameters
    ----------
    y : array-like
        Response variable
    X : DataFrame
        Predictor variables
    y_name : str, default='Y'
        Name of the response variable
    initial_values : dict, optional
        Initial values for predictors. Defaults to means.
        
    Returns
    -------
    PredictionProfiler
        Interactive profiler object
        
    Example
    -------
    >>> profiler = prediction_profiler(df['sales'], df[['price', 'advertising']])
    >>> print(profiler)
    >>> result = profiler.predict_at(price=10.0, advertising=5000)
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required")
    
    X_names = list(X.columns)
    
    y = np.asarray(y).flatten()
    mask = ~(np.isnan(y) | X.isna().any(axis=1))
    y_clean = y[mask]
    X_clean = X[mask]
    
    X_const = sm.add_constant(X_clean)
    model = sm.OLS(y_clean, X_const).fit()
    
    X_ranges = {col: (X_clean[col].min(), X_clean[col].max()) for col in X_names}
    
    if initial_values is None:
        X_current = {col: X_clean[col].mean() for col in X_names}
    else:
        X_current = {col: initial_values.get(col, X_clean[col].mean()) for col in X_names}
    
    X_arr = np.array([[X_current[name] for name in X_names]])
    X_pred = sm.add_constant(X_arr, has_constant='add')
    pred = model.get_prediction(X_pred)
    
    return PredictionProfiler(
        model=model,
        X_names=X_names,
        X_ranges=X_ranges,
        X_current=X_current,
        y_name=y_name,
        prediction=float(pred.predicted_mean[0]),
        conf_interval=tuple(pred.conf_int(alpha=0.05)[0]),
        pred_interval=tuple(pred.conf_int(obs=True, alpha=0.05)[0])
    )


def plot_prediction_profiler(
    profiler: PredictionProfiler,
    n_points: int = 50,
    figsize: Tuple[int, int] = (14, 4)
) -> Optional[Any]:
    """
    Create visual representation of the prediction profiler (JMP-style).
    
    Parameters
    ----------
    profiler : PredictionProfiler
        Profiler object from prediction_profiler()
    n_points : int, default=50
        Number of points for each profile line
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    if not HAS_MATPLOTLIB:
        return None
    
    n_vars = len(profiler.X_names)
    fig_width = max(14, n_vars * 3.5)
    
    fig, axes = plt.subplots(1, n_vars, figsize=(fig_width, figsize[1]))
    if n_vars == 1:
        axes = [axes]
    
    for idx, var_name in enumerate(profiler.X_names):
        ax = axes[idx]
        
        var_min, var_max = profiler.X_ranges[var_name]
        var_values = np.linspace(var_min, var_max, n_points)
        
        predictions = []
        conf_low = []
        conf_high = []
        
        for val in var_values:
            X_vals = profiler.X_current.copy()
            X_vals[var_name] = val
            X_arr = np.array([[X_vals[name] for name in profiler.X_names]])
            X_const = sm.add_constant(X_arr, has_constant='add')
            
            pred = profiler.model.get_prediction(X_const)
            predictions.append(pred.predicted_mean[0])
            ci = pred.conf_int(alpha=0.05)[0]
            conf_low.append(ci[0])
            conf_high.append(ci[1])
        
        ax.plot(var_values, predictions, 'b-', lw=2, label='Prediction')
        ax.fill_between(var_values, conf_low, conf_high, alpha=0.2, color='blue', label='95% CI')
        
        current_val = profiler.X_current[var_name]
        ax.axvline(x=current_val, color='red', linestyle='--', lw=1.5, label='Current')
        ax.axhline(y=profiler.prediction, color='gray', linestyle=':', alpha=0.5)
        ax.scatter([current_val], [profiler.prediction], color='red', s=100, zorder=5)
        
        ax.set_xlabel(var_name)
        ax.set_ylabel(profiler.y_name if idx == 0 else '')
        ax.set_title(f'{var_name}')
        
        if idx == 0:
            ax.legend(loc='best', fontsize=8)
    
    plt.suptitle(f'Prediction Profiler: {profiler.y_name} = {profiler.prediction:.4f}', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# NEW v2.0: DESIGN OF EXPERIMENTS (DoE)
# =============================================================================

@dataclass
class FactorialDesignResults:
    """Container for factorial design analysis results."""
    design_matrix: pd.DataFrame
    effects: Dict[str, float]
    effects_se: Dict[str, float]
    effects_tstat: Dict[str, float]
    effects_pvalue: Dict[str, float]
    anova_table: pd.DataFrame
    significant_effects: List[str]
    r_squared: float
    adj_r_squared: float
    
    def __str__(self):
        sig_effects = ', '.join(self.significant_effects) if self.significant_effects else 'None'
        effects_str = '\n'.join([f"  {k}: {v:.4f} (p={self.effects_pvalue[k]:.4f})" 
                                  for k, v in list(self.effects.items())[:10]])
        return f"""
Factorial Design Analysis
=========================
R-Squared:              {self.r_squared:.6f}
Adj R-Squared:          {self.adj_r_squared:.6f}

Significant Effects (α=0.05):
  {sig_effects}

Effect Estimates:
{effects_str}
"""


@dataclass 
class EffectScreeningResults:
    """Container for effect screening results."""
    effects: pd.DataFrame
    significant: List[str]
    pareto_fig: Optional[Any]
    half_normal_fig: Optional[Any]
    lenth_pse: float
    
    def __str__(self):
        return f"""
Effect Screening Results
========================
Lenth's PSE:            {self.lenth_pse:.6f}
Significant Effects:    {', '.join(self.significant) if self.significant else 'None'}

Effect Magnitudes:
{self.effects.head(10).to_string()}
"""


@dataclass
class ResponseSurfaceResults:
    """Container for response surface design and analysis."""
    design_matrix: pd.DataFrame
    design_type: str
    factors: List[str]
    center_points: int
    model: Optional[Any]
    optimal_settings: Optional[Dict[str, float]]
    predicted_optimum: Optional[float]
    
    def __str__(self):
        opt_str = ""
        if self.optimal_settings:
            opt_str = f"\nOptimal Settings:\n"
            for k, v in self.optimal_settings.items():
                opt_str += f"  {k}: {v:.4f}\n"
            opt_str += f"\nPredicted Optimum: {self.predicted_optimum:.6f}"
        
        return f"""
Response Surface Design
=======================
Design Type:            {self.design_type}
Factors:                {', '.join(self.factors)}
Center Points:          {self.center_points}
Total Runs:             {len(self.design_matrix)}
{opt_str}
"""


def fractional_factorial_design(
    factors: Dict[str, Tuple[float, float]],
    resolution: int = 4,
    center_points: int = 0,
    replicates: int = 1
) -> pd.DataFrame:
    """
    Create a fractional factorial design matrix.
    
    Parameters
    ----------
    factors : dict
        Dictionary of factor names to (low, high) value tuples
    resolution : int, default=4
        Design resolution (3, 4, or 5)
    center_points : int, default=0
        Number of center points to add
    replicates : int, default=1
        Number of replicates
        
    Returns
    -------
    pd.DataFrame
        Design matrix with factor values
    """
    factor_names = list(factors.keys())
    k = len(factor_names)
    
    if k <= 4:
        n_runs = 2 ** k
        coded = np.array(list(np.ndindex(*([2] * k)))) * 2 - 1
    else:
        if resolution >= 5:
            p = 1
        elif resolution == 4:
            p = max(1, k - 4)
        else:
            p = max(1, k - 3)
        
        n_base = k - p
        base_design = np.array(list(np.ndindex(*([2] * n_base)))) * 2 - 1
        
        coded = np.zeros((len(base_design), k))
        coded[:, :n_base] = base_design
        
        for i in range(p):
            cols_to_multiply = list(range(min(i + 2, n_base)))
            coded[:, n_base + i] = np.prod(base_design[:, cols_to_multiply], axis=1)
    
    design = pd.DataFrame(coded, columns=factor_names)
    for col in factor_names:
        low, high = factors[col]
        mid = (low + high) / 2
        half_range = (high - low) / 2
        design[col] = mid + design[col] * half_range
    
    if center_points > 0:
        center_row = {col: (factors[col][0] + factors[col][1]) / 2 for col in factor_names}
        center_df = pd.DataFrame([center_row] * center_points)
        design = pd.concat([design, center_df], ignore_index=True)
    
    if replicates > 1:
        design = pd.concat([design] * replicates, ignore_index=True)
    
    design.insert(0, 'Run', range(1, len(design) + 1))
    design['RunOrder'] = np.random.permutation(len(design)) + 1
    
    return design.sort_values('RunOrder').reset_index(drop=True)


def response_surface_design(
    factors: Dict[str, Tuple[float, float]],
    design_type: str = 'ccd',
    center_points: int = 5,
    alpha: Optional[float] = None
) -> ResponseSurfaceResults:
    """
    Create a response surface design (Central Composite or Box-Behnken).
    
    Parameters
    ----------
    factors : dict
        Dictionary of factor names to (low, high) value tuples
    design_type : str, default='ccd'
        Type of design: 'ccd', 'ccf', 'cci', or 'bbd'
    center_points : int, default=5
        Number of center points
    alpha : float, optional
        Axial distance for CCD
        
    Returns
    -------
    ResponseSurfaceResults
    """
    factor_names = list(factors.keys())
    k = len(factor_names)
    
    if design_type.lower() in ['ccd', 'ccf', 'cci']:
        factorial = np.array(list(np.ndindex(*([2] * k)))) * 2 - 1
        
        if alpha is None:
            if design_type.lower() == 'ccf':
                alpha = 1.0
            elif design_type.lower() == 'cci':
                alpha = 1.0 / np.sqrt(k)
            else:
                alpha = (2 ** k) ** 0.25
        
        axial = np.zeros((2 * k, k))
        for i in range(k):
            axial[2*i, i] = -alpha
            axial[2*i + 1, i] = alpha
        
        center = np.zeros((center_points, k))
        coded = np.vstack([factorial, axial, center])
        
    elif design_type.lower() == 'bbd':
        if k < 3:
            raise ValueError("Box-Behnken design requires at least 3 factors")
        
        from itertools import combinations
        
        points = []
        for pair in combinations(range(k), 2):
            for signs in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                point = [0] * k
                point[pair[0]] = signs[0]
                point[pair[1]] = signs[1]
                points.append(point)
        
        for _ in range(center_points):
            points.append([0] * k)
        
        coded = np.array(points)
    else:
        raise ValueError(f"Unknown design type: {design_type}")
    
    design = pd.DataFrame(coded, columns=factor_names)
    for col in factor_names:
        low, high = factors[col]
        mid = (low + high) / 2
        half_range = (high - low) / 2
        design[col] = mid + design[col] * half_range
    
    design.insert(0, 'Run', range(1, len(design) + 1))
    design['RunOrder'] = np.random.permutation(len(design)) + 1
    design = design.sort_values('RunOrder').reset_index(drop=True)
    
    return ResponseSurfaceResults(
        design_matrix=design,
        design_type=design_type.upper(),
        factors=factor_names,
        center_points=center_points,
        model=None,
        optimal_settings=None,
        predicted_optimum=None
    )


def optimal_design(
    factors: Dict[str, Tuple[float, float]],
    n_runs: int,
    model_type: str = 'quadratic',
    criterion: str = 'd-optimal',
    n_starts: int = 10
) -> pd.DataFrame:
    """
    Generate D-optimal or I-optimal design.
    
    Parameters
    ----------
    factors : dict
        Factor names to (low, high) tuples
    n_runs : int
        Number of experimental runs
    model_type : str, default='quadratic'
        Type of model: 'linear', 'interaction', 'quadratic'
    criterion : str, default='d-optimal'
        Optimality criterion
    n_starts : int, default=10
        Number of random starts
        
    Returns
    -------
    pd.DataFrame
        Optimal design matrix
    """
    factor_names = list(factors.keys())
    k = len(factor_names)
    
    n_candidates = max(100, n_runs * 10)
    candidates = np.random.uniform(-1, 1, (n_candidates, k))
    
    corners = np.array(list(np.ndindex(*([2] * k)))) * 2 - 1
    center = np.zeros((1, k))
    axials = np.vstack([np.eye(k), -np.eye(k)])
    
    candidates = np.vstack([candidates, corners, center, axials])
    
    def expand_model_matrix(X, model_type):
        n = X.shape[0]
        cols = [np.ones(n)]
        
        for j in range(k):
            cols.append(X[:, j])
        
        if model_type in ['interaction', 'quadratic']:
            for i in range(k):
                for j in range(i + 1, k):
                    cols.append(X[:, i] * X[:, j])
        
        if model_type == 'quadratic':
            for j in range(k):
                cols.append(X[:, j] ** 2)
        
        return np.column_stack(cols)
    
    best_design = None
    best_criterion = -np.inf if criterion == 'd-optimal' else np.inf
    
    for _ in range(n_starts):
        indices = np.random.choice(len(candidates), n_runs, replace=False)
        design = candidates[indices].copy()
        
        for iteration in range(50):
            improved = False
            for i in range(n_runs):
                current_X = expand_model_matrix(design, model_type)
                
                try:
                    if criterion == 'd-optimal':
                        current_det = np.linalg.det(current_X.T @ current_X)
                    else:
                        current_trace = np.trace(np.linalg.inv(current_X.T @ current_X))
                except:
                    continue
                
                for j, candidate in enumerate(candidates[:100]):
                    if any(np.allclose(design[ii], candidate) for ii in range(n_runs) if ii != i):
                        continue
                    
                    test_design = design.copy()
                    test_design[i] = candidate
                    test_X = expand_model_matrix(test_design, model_type)
                    
                    try:
                        if criterion == 'd-optimal':
                            test_det = np.linalg.det(test_X.T @ test_X)
                            if test_det > current_det * 1.001:
                                design = test_design
                                current_det = test_det
                                improved = True
                        else:
                            test_trace = np.trace(np.linalg.inv(test_X.T @ test_X))
                            if test_trace < current_trace * 0.999:
                                design = test_design
                                current_trace = test_trace
                                improved = True
                    except:
                        continue
            
            if not improved:
                break
        
        final_X = expand_model_matrix(design, model_type)
        try:
            if criterion == 'd-optimal':
                crit_val = np.linalg.det(final_X.T @ final_X)
                if crit_val > best_criterion:
                    best_criterion = crit_val
                    best_design = design.copy()
            else:
                crit_val = np.trace(np.linalg.inv(final_X.T @ final_X))
                if crit_val < best_criterion:
                    best_criterion = crit_val
                    best_design = design.copy()
        except:
            continue
    
    if best_design is None:
        best_design = candidates[:n_runs]
    
    result = pd.DataFrame(best_design, columns=factor_names)
    for col in factor_names:
        low, high = factors[col]
        mid = (low + high) / 2
        half_range = (high - low) / 2
        result[col] = mid + result[col] * half_range
    
    result.insert(0, 'Run', range(1, len(result) + 1))
    return result


def analyze_factorial(
    design: pd.DataFrame,
    response: str,
    factors: List[str],
    include_interactions: bool = True,
    alpha: float = 0.05
) -> FactorialDesignResults:
    """
    Analyze a factorial experiment.
    
    Parameters
    ----------
    design : DataFrame
        Design matrix with response
    response : str
        Name of response column
    factors : list
        Names of factor columns
    include_interactions : bool, default=True
        Include 2-way interactions
    alpha : float, default=0.05
        Significance level
        
    Returns
    -------
    FactorialDesignResults
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required")
    
    y = design[response].values
    
    X_coded = pd.DataFrame()
    factor_means = {}
    factor_ranges = {}
    
    for factor in factors:
        vals = design[factor].values
        factor_means[factor] = vals.mean()
        factor_ranges[factor] = (vals.max() - vals.min()) / 2
        if factor_ranges[factor] == 0:
            factor_ranges[factor] = 1
        X_coded[factor] = (vals - factor_means[factor]) / factor_ranges[factor]
    
    if include_interactions:
        for i, f1 in enumerate(factors):
            for f2 in factors[i+1:]:
                interaction_name = f'{f1}*{f2}'
                X_coded[interaction_name] = X_coded[f1] * X_coded[f2]
    
    X_const = sm.add_constant(X_coded)
    model = sm.OLS(y, X_const).fit()
    
    effects = {}
    effects_se = {}
    effects_tstat = {}
    effects_pvalue = {}
    
    for i, name in enumerate(X_coded.columns):
        idx = i + 1
        effects[name] = float(model.params[idx] * 2)
        effects_se[name] = float(model.bse[idx] * 2)
        effects_tstat[name] = float(model.tvalues[idx])
        effects_pvalue[name] = float(model.pvalues[idx])
    
    significant = [name for name, pval in effects_pvalue.items() if pval < alpha]
    
    anova_data = {
        'Source': list(effects.keys()) + ['Error', 'Total'],
        'DF': [1] * len(effects) + [int(model.df_resid), int(model.df_model + model.df_resid)],
        'F Ratio': [float(model.tvalues[i+1]**2) for i in range(len(effects))] + [np.nan, np.nan],
        'Prob > F': list(effects_pvalue.values()) + [np.nan, np.nan]
    }
    anova_df = pd.DataFrame(anova_data)
    
    return FactorialDesignResults(
        design_matrix=design,
        effects=effects,
        effects_se=effects_se,
        effects_tstat=effects_tstat,
        effects_pvalue=effects_pvalue,
        anova_table=anova_df,
        significant_effects=significant,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj)
    )


def effect_screening(
    design: pd.DataFrame,
    response: str,
    factors: List[str],
    method: str = 'lenth',
    alpha: float = 0.05,
    plot: bool = True
) -> EffectScreeningResults:
    """
    Screen effects using Lenth's method.
    
    Parameters
    ----------
    design : DataFrame
        Design matrix with response
    response : str
        Response column name
    factors : list
        Factor column names
    method : str, default='lenth'
        Screening method
    alpha : float, default=0.05
        Significance level
    plot : bool, default=True
        Generate plots
        
    Returns
    -------
    EffectScreeningResults
    """
    y = design[response].values
    
    effects_list = []
    for factor in factors:
        vals = design[factor].values
        val_range = vals.max() - vals.min()
        if val_range == 0:
            val_range = 1
        coded = (vals - vals.mean()) / (val_range / 2)
        corr = np.corrcoef(coded, y)[0, 1]
        if np.isnan(corr):
            corr = 0
        effect = 2 * corr * np.std(y) / (np.std(coded) + 1e-10)
        effects_list.append({'Factor': factor, 'Effect': effect, 'AbsEffect': abs(effect)})
    
    effects_df = pd.DataFrame(effects_list).sort_values('AbsEffect', ascending=False)
    
    abs_effects = effects_df['AbsEffect'].values
    abs_effects = abs_effects[abs_effects > 0]
    if len(abs_effects) == 0:
        abs_effects = np.array([1.0])
    
    s0 = 1.5 * np.median(abs_effects)
    filtered = abs_effects[abs_effects < 2.5 * s0]
    if len(filtered) == 0:
        filtered = abs_effects
    pse = 1.5 * np.median(filtered)
    
    m = len(effects_list)
    t_crit = 2.0 + 0.5 * np.log(max(m, 1))
    me = t_crit * pse
    
    significant = list(effects_df[effects_df['AbsEffect'] > me]['Factor'])
    
    pareto_fig = None
    half_normal_fig = None
    
    if plot and HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if f in significant else 'steelblue' for f in effects_df['Factor']]
        ax.barh(range(len(effects_df)), effects_df['AbsEffect'].values, color=colors)
        ax.set_yticks(range(len(effects_df)))
        ax.set_yticklabels(effects_df['Factor'].values)
        ax.axvline(x=me, color='red', linestyle='--', label=f'Margin of Error ({me:.4f})')
        ax.set_xlabel('|Effect|')
        ax.set_title('Pareto Chart of Effects')
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        pareto_fig = fig
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        all_abs = effects_df['AbsEffect'].values
        sorted_effects = np.sort(all_abs)
        n = len(sorted_effects)
        expected = [norm.ppf((i + 0.5) / n) for i in range(n)]
        
        colors = ['red' if e > me else 'steelblue' for e in sorted_effects]
        ax2.scatter(expected, sorted_effects, c=colors, s=60)
        
        ax2.set_xlabel('Half-Normal Quantiles')
        ax2.set_ylabel('|Effect|')
        ax2.set_title('Half-Normal Plot of Effects')
        plt.tight_layout()
        half_normal_fig = fig2
    
    return EffectScreeningResults(
        effects=effects_df,
        significant=significant,
        pareto_fig=pareto_fig,
        half_normal_fig=half_normal_fig,
        lenth_pse=pse
    )


def interaction_plot(
    design: pd.DataFrame,
    response: str,
    factor1: str,
    factor2: str,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[Any]:
    """Create interaction plot for two factors."""
    if not HAS_MATPLOTLIB:
        return None
    
    levels1 = sorted(design[factor1].unique())
    levels2 = sorted(design[factor2].unique())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for level2 in levels2:
        means = []
        for level1 in levels1:
            mask = (design[factor1] == level1) & (design[factor2] == level2)
            means.append(design.loc[mask, response].mean())
        
        ax.plot(range(len(levels1)), means, 'o-', label=f'{factor2}={level2}', markersize=8)
    
    ax.set_xticks(range(len(levels1)))
    ax.set_xticklabels([str(l) for l in levels1])
    ax.set_xlabel(factor1)
    ax.set_ylabel(f'Mean {response}')
    ax.set_title(f'Interaction Plot: {factor1} × {factor2}')
    ax.legend(title=factor2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main_effects_plot(
    design: pd.DataFrame,
    response: str,
    factors: List[str],
    figsize: Tuple[int, int] = (14, 4)
) -> Optional[Any]:
    """Create main effects plots for all factors."""
    if not HAS_MATPLOTLIB:
        return None
    
    n_factors = len(factors)
    fig, axes = plt.subplots(1, n_factors, figsize=(figsize[0], figsize[1]))
    if n_factors == 1:
        axes = [axes]
    
    overall_mean = design[response].mean()
    
    for idx, factor in enumerate(factors):
        ax = axes[idx]
        levels = sorted(design[factor].unique())
        means = [design[design[factor] == level][response].mean() for level in levels]
        
        ax.plot(range(len(levels)), means, 'bo-', markersize=10)
        ax.axhline(y=overall_mean, color='gray', linestyle='--', label='Grand Mean')
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f'{l:.2f}' if isinstance(l, float) else str(l) for l in levels])
        ax.set_xlabel(factor)
        ax.set_ylabel(f'Mean {response}' if idx == 0 else '')
        ax.set_title(factor)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Main Effects Plots', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def contour_profiler(
    model,
    factor1: str,
    factor2: str,
    factor_ranges: Dict[str, Tuple[float, float]],
    fixed_values: Optional[Dict[str, float]] = None,
    response_name: str = 'Y',
    n_points: int = 50,
    figsize: Tuple[int, int] = (10, 8)
) -> Optional[Any]:
    """Create contour plot profiler for response surface."""
    if not HAS_MATPLOTLIB:
        return None
    
    factors = list(factor_ranges.keys())
    
    if fixed_values is None:
        fixed_values = {f: (factor_ranges[f][0] + factor_ranges[f][1]) / 2 
                       for f in factors if f not in [factor1, factor2]}
    
    x1_range = np.linspace(factor_ranges[factor1][0], factor_ranges[factor1][1], n_points)
    x2_range = np.linspace(factor_ranges[factor2][0], factor_ranges[factor2][1], n_points)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    Z = np.zeros_like(X1)
    for i in range(n_points):
        for j in range(n_points):
            X_dict = fixed_values.copy()
            X_dict[factor1] = X1[i, j]
            X_dict[factor2] = X2[i, j]
            X_arr = np.array([[X_dict[f] for f in factors]])
            X_const = sm.add_constant(X_arr, has_constant='add')
            Z[i, j] = model.predict(X_const)[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cs = ax.contourf(X1, X2, Z, levels=20, cmap='RdYlGn')
    plt.colorbar(cs, ax=ax, label=response_name)
    ax.contour(X1, X2, Z, levels=10, colors='black', linewidths=0.5)
    
    ax.set_xlabel(factor1)
    ax.set_ylabel(factor2)
    ax.set_title(f'Contour Profiler: {response_name}')
    
    opt_idx = np.unravel_index(np.argmax(Z), Z.shape)
    ax.plot(X1[opt_idx], X2[opt_idx], 'w*', markersize=15, markeredgecolor='black')
    
    plt.tight_layout()
    return fig


def pareto_of_effects(
    effects: Dict[str, float],
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[Any]:
    """Create Pareto chart of effects."""
    if not HAS_MATPLOTLIB:
        return None
    
    sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [e[0] for e in sorted_effects]
    values = [abs(e[1]) for e in sorted_effects]
    signs = ['positive' if e[1] > 0 else 'negative' for e in sorted_effects]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['steelblue' if s == 'positive' else 'coral' for s in signs]
    ax.barh(range(len(names)), values, color=colors)
    
    if threshold is not None:
        ax.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
        ax.legend()
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('|Effect|')
    ax.set_title('Pareto Chart of Effects')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


# =============================================================================
# NEW v2.0: TIME SERIES ANALYSIS
# =============================================================================

@dataclass
class ARIMAResults:
    """Container for ARIMA model results."""
    order: Tuple[int, int, int]
    seasonal_order: Optional[Tuple[int, int, int, int]]
    model: Any
    aic: float
    bic: float
    coefficients: Dict[str, float]
    residuals: np.ndarray
    fitted_values: np.ndarray
    forecast: Optional[np.ndarray]
    forecast_ci: Optional[np.ndarray]
    diagnostics: Dict[str, Any]
    
    def __str__(self):
        order_str = f"ARIMA{self.order}"
        if self.seasonal_order:
            order_str += f"x{self.seasonal_order}"
        
        coef_str = '\n'.join([f"  {k}: {v:.6f}" for k, v in list(self.coefficients.items())[:5]])
        
        return f"""
ARIMA Model Results
===================
Model: {order_str}
AIC: {self.aic:.2f}
BIC: {self.bic:.2f}

Coefficients:
{coef_str}

Residual Diagnostics:
  Ljung-Box p-value: {self.diagnostics.get('ljung_box_pvalue', np.nan):.4f}
"""
    
    def predict(self, steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts with confidence intervals."""
        forecast = self.model.get_forecast(steps=steps)
        return forecast.predicted_mean.values, forecast.conf_int().values


@dataclass
class ExponentialSmoothingResults:
    """Container for exponential smoothing results."""
    method: str
    alpha: float
    beta: Optional[float]
    gamma: Optional[float]
    seasonal_periods: Optional[int]
    model: Any
    aic: float
    fitted_values: np.ndarray
    residuals: np.ndarray
    forecast: Optional[np.ndarray]
    forecast_ci: Optional[np.ndarray]
    
    def __str__(self):
        params = f"α={self.alpha:.4f}"
        if self.beta is not None:
            params += f", β={self.beta:.4f}"
        if self.gamma is not None:
            params += f", γ={self.gamma:.4f}"
        
        return f"""
Exponential Smoothing Results
=============================
Method: {self.method}
Parameters: {params}
Seasonal Periods: {self.seasonal_periods or 'None'}
AIC: {self.aic:.2f}
"""


@dataclass
class TimeSeriesDecomposition:
    """Container for time series decomposition."""
    observed: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    period: int
    method: str
    
    def __str__(self):
        return f"""
Time Series Decomposition
=========================
Method: {self.method}
Period: {self.period}
Observations: {len(self.observed)}
"""


@dataclass
class AutocorrelationResults:
    """Container for ACF/PACF analysis."""
    acf_values: np.ndarray
    pacf_values: np.ndarray
    acf_conf_int: np.ndarray
    pacf_conf_int: np.ndarray
    n_lags: int
    significant_acf_lags: List[int]
    significant_pacf_lags: List[int]
    suggested_ar_order: int
    suggested_ma_order: int
    
    def __str__(self):
        return f"""
Autocorrelation Analysis
========================
Lags Analyzed: {self.n_lags}
Significant ACF Lags: {self.significant_acf_lags[:10]}
Significant PACF Lags: {self.significant_pacf_lags[:10]}

Suggested Orders:
  AR (p): {self.suggested_ar_order}
  MA (q): {self.suggested_ma_order}
"""


def arima(
    y: Union[pd.Series, np.ndarray],
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    auto_select: bool = True,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    forecast_steps: int = 0,
    verbose: bool = False
) -> ARIMAResults:
    """
    Fit ARIMA model with optional automatic order selection.
    
    Parameters
    ----------
    y : array-like
        Time series data
    order : tuple, optional
        (p, d, q) order
    seasonal_order : tuple, optional
        (P, D, Q, s) seasonal order
    auto_select : bool, default=True
        Automatically select optimal order
    max_p, max_d, max_q : int
        Maximum orders for auto selection
    forecast_steps : int, default=0
        Number of steps to forecast
    verbose : bool
        Print progress
        
    Returns
    -------
    ARIMAResults
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required")
    
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    
    if order is None and auto_select:
        d = 0
        y_diff = y.copy()
        for i in range(max_d + 1):
            try:
                adf_stat, adf_pval, *_ = adfuller(y_diff, maxlag=int(np.sqrt(len(y_diff))))
                if adf_pval < 0.05:
                    d = i
                    break
                if i < max_d:
                    y_diff = np.diff(y_diff)
            except:
                break
        else:
            d = max_d
        
        if verbose:
            print(f"Selected d = {d}")
        
        best_aic = np.inf
        best_order = (0, d, 0)
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(y, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        if verbose:
                            print(f"  ARIMA{(p, d, q)}: AIC = {fitted.aic:.2f}")
                except:
                    continue
        
        order = best_order
        if verbose:
            print(f"Selected order: ARIMA{order}")
    
    if seasonal_order is not None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
    else:
        model = ARIMA(y, order=order)
    
    fitted = model.fit()
    
    try:
        coefficients = dict(zip(fitted.params.index, fitted.params.values))
    except:
        coefficients = {f'param_{i}': v for i, v in enumerate(fitted.params)}
    
    residuals = fitted.resid
    try:
        lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pval = float(lb_result['lb_pvalue'].iloc[0])
    except:
        lb_pval = np.nan
    
    diagnostics = {
        'ljung_box_pvalue': lb_pval,
    }
    
    forecast = None
    forecast_ci = None
    if forecast_steps > 0:
        fc = fitted.get_forecast(steps=forecast_steps)
        forecast = fc.predicted_mean.values
        forecast_ci = fc.conf_int().values
    
    return ARIMAResults(
        order=order,
        seasonal_order=seasonal_order,
        model=fitted,
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        coefficients=coefficients,
        residuals=residuals,
        fitted_values=fitted.fittedvalues,
        forecast=forecast,
        forecast_ci=forecast_ci,
        diagnostics=diagnostics
    )


def exponential_smoothing(
    y: Union[pd.Series, np.ndarray],
    method: str = 'auto',
    seasonal_periods: Optional[int] = None,
    damped: bool = False,
    forecast_steps: int = 0
) -> ExponentialSmoothingResults:
    """
    Fit exponential smoothing model.
    
    Parameters
    ----------
    y : array-like
        Time series data
    method : str, default='auto'
        'simple', 'holt', 'holt-winters', or 'auto'
    seasonal_periods : int, optional
        Number of periods in seasonal cycle
    damped : bool, default=False
        Use damped trend
    forecast_steps : int, default=0
        Number of steps to forecast
        
    Returns
    -------
    ExponentialSmoothingResults
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required")
    
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HW
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    
    if method == 'auto':
        candidates = []
        
        try:
            m = SimpleExpSmoothing(y).fit()
            candidates.append(('simple', m, m.aic))
        except:
            pass
        
        try:
            m = HW(y, trend='add', damped_trend=damped).fit()
            candidates.append(('holt', m, m.aic))
        except:
            pass
        
        if seasonal_periods is not None and seasonal_periods > 1:
            try:
                m = HW(y, trend='add', seasonal='add', seasonal_periods=seasonal_periods, 
                       damped_trend=damped).fit()
                candidates.append(('holt-winters', m, m.aic))
            except:
                pass
        
        if not candidates:
            raise ValueError("Could not fit any model")
        
        best = min(candidates, key=lambda x: x[2])
        method, fitted, _ = best
    else:
        if method == 'simple':
            fitted = SimpleExpSmoothing(y).fit()
        elif method in ['holt', 'double']:
            fitted = HW(y, trend='add', damped_trend=damped).fit()
            method = 'holt'
        elif method in ['holt-winters', 'triple']:
            if seasonal_periods is None:
                raise ValueError("seasonal_periods required for Holt-Winters")
            fitted = HW(y, trend='add', seasonal='add', seasonal_periods=seasonal_periods,
                       damped_trend=damped).fit()
            method = 'holt-winters'
        else:
            raise ValueError(f"Unknown method: {method}")
    
    alpha = getattr(fitted, 'params', {}).get('smoothing_level', 
                   getattr(fitted, 'smoothing_level', np.nan))
    beta = getattr(fitted, 'params', {}).get('smoothing_trend',
                  getattr(fitted, 'smoothing_trend', None))
    gamma = getattr(fitted, 'params', {}).get('smoothing_seasonal',
                   getattr(fitted, 'smoothing_seasonal', None))
    
    forecast = None
    forecast_ci = None
    if forecast_steps > 0:
        forecast = fitted.forecast(steps=forecast_steps)
        residual_std = np.std(fitted.resid)
        forecast_ci = np.column_stack([
            forecast - 1.96 * residual_std * np.sqrt(np.arange(1, forecast_steps + 1)),
            forecast + 1.96 * residual_std * np.sqrt(np.arange(1, forecast_steps + 1))
        ])
    
    return ExponentialSmoothingResults(
        method=method,
        alpha=alpha if alpha is not None else np.nan,
        beta=beta,
        gamma=gamma,
        seasonal_periods=seasonal_periods,
        model=fitted,
        aic=float(fitted.aic),
        fitted_values=fitted.fittedvalues,
        residuals=fitted.resid,
        forecast=forecast,
        forecast_ci=forecast_ci
    )


def autocorrelation_analysis(
    y: Union[pd.Series, np.ndarray],
    n_lags: int = 40,
    alpha: float = 0.05
) -> AutocorrelationResults:
    """
    Compute ACF and PACF with significance testing.
    
    Parameters
    ----------
    y : array-like
        Time series data
    n_lags : int, default=40
        Number of lags
    alpha : float, default=0.05
        Significance level
        
    Returns
    -------
    AutocorrelationResults
    """
    from statsmodels.tsa.stattools import acf, pacf
    
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    
    n = len(y)
    n_lags = min(n_lags, n // 2 - 1)
    
    acf_vals, acf_ci = acf(y, nlags=n_lags, alpha=alpha, fft=True)
    pacf_vals, pacf_ci = pacf(y, nlags=n_lags, alpha=alpha)
    
    threshold = norm.ppf(1 - alpha / 2) / np.sqrt(n)
    
    sig_acf = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > threshold]
    sig_pacf = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > threshold]
    
    suggested_ar = max(sig_pacf) if sig_pacf else 0
    
    suggested_ma = 0
    for i, sig in enumerate(sig_acf):
        if i == 0 or sig == sig_acf[i-1] + 1:
            suggested_ma = sig
        else:
            break
    
    return AutocorrelationResults(
        acf_values=acf_vals,
        pacf_values=pacf_vals,
        acf_conf_int=acf_ci,
        pacf_conf_int=pacf_ci,
        n_lags=n_lags,
        significant_acf_lags=sig_acf,
        significant_pacf_lags=sig_pacf,
        suggested_ar_order=min(suggested_ar, 5),
        suggested_ma_order=min(suggested_ma, 5)
    )


def seasonal_decomposition(
    y: Union[pd.Series, np.ndarray],
    period: int,
    method: str = 'additive'
) -> TimeSeriesDecomposition:
    """
    Decompose time series into trend, seasonal, and residual.
    
    Parameters
    ----------
    y : array-like
        Time series data
    period : int
        Seasonal period
    method : str, default='additive'
        'additive' or 'multiplicative'
        
    Returns
    -------
    TimeSeriesDecomposition
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    y = np.asarray(y).flatten()
    
    result = seasonal_decompose(y, model=method, period=period)
    
    return TimeSeriesDecomposition(
        observed=result.observed,
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.resid,
        period=period,
        method=method
    )


def time_series_forecast(
    y: Union[pd.Series, np.ndarray],
    steps: int = 10,
    method: str = 'auto',
    seasonal_periods: Optional[int] = None
) -> Dict[str, Any]:
    """
    Unified forecasting interface.
    
    Parameters
    ----------
    y : array-like
        Time series data
    steps : int, default=10
        Forecast horizon
    method : str, default='auto'
        'arima', 'ets', or 'auto'
    seasonal_periods : int, optional
        Seasonal period
        
    Returns
    -------
    dict
        Forecasts and confidence intervals
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    
    if method == 'auto':
        try:
            arima_result = arima(y, auto_select=True, forecast_steps=steps)
            arima_aic = arima_result.aic
        except:
            arima_aic = np.inf
            arima_result = None
        
        try:
            ets_result = exponential_smoothing(y, method='auto', 
                                               seasonal_periods=seasonal_periods,
                                               forecast_steps=steps)
            ets_aic = ets_result.aic
        except:
            ets_aic = np.inf
            ets_result = None
        
        if arima_aic <= ets_aic and arima_result is not None:
            return {
                'method': f'ARIMA{arima_result.order}',
                'forecast': arima_result.forecast,
                'conf_int': arima_result.forecast_ci,
                'aic': arima_aic,
                'model': arima_result
            }
        elif ets_result is not None:
            return {
                'method': f'ETS ({ets_result.method})',
                'forecast': ets_result.forecast,
                'conf_int': ets_result.forecast_ci,
                'aic': ets_aic,
                'model': ets_result
            }
        else:
            raise ValueError("Could not fit any model")
    
    elif method == 'arima':
        result = arima(y, auto_select=True, forecast_steps=steps)
        return {
            'method': f'ARIMA{result.order}',
            'forecast': result.forecast,
            'conf_int': result.forecast_ci,
            'aic': result.aic,
            'model': result
        }
    
    else:
        result = exponential_smoothing(y, method=method if method != 'ets' else 'auto',
                                       seasonal_periods=seasonal_periods,
                                       forecast_steps=steps)
        return {
            'method': f'ETS ({result.method})',
            'forecast': result.forecast,
            'conf_int': result.forecast_ci,
            'aic': result.aic,
            'model': result
        }


def plot_time_series_diagnostics(
    y: Union[pd.Series, np.ndarray],
    model_results: Optional[Any] = None,
    n_lags: int = 24,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[Any]:
    """
    Create comprehensive time series diagnostic plots.
    
    Parameters
    ----------
    y : array-like
        Time series data
    model_results : optional
        Fitted model results (ARIMAResults or ExponentialSmoothingResults)
    n_lags : int, default=24
        Number of lags for ACF/PACF
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    if not HAS_MATPLOTLIB:
        return None
    
    from statsmodels.tsa.stattools import acf, pacf
    
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)
    
    if model_results is not None:
        residuals = model_results.residuals
    else:
        residuals = y - np.mean(y)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Time Series
    ax = axes[0, 0]
    ax.plot(y, 'b-', lw=1, label='Observed')
    if model_results is not None and hasattr(model_results, 'fitted_values'):
        ax.plot(model_results.fitted_values, 'r-', lw=1, alpha=0.7, label='Fitted')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Time Series')
    ax.legend()
    
    # Plot 2: ACF
    ax = axes[0, 1]
    n_lags_plot = min(n_lags, n // 2 - 1)
    acf_vals = acf(residuals, nlags=n_lags_plot, fft=True)
    ax.bar(range(len(acf_vals)), acf_vals, color='steelblue', alpha=0.7)
    conf = 1.96 / np.sqrt(n)
    ax.axhline(y=conf, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-conf, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_title('Autocorrelation Function (Residuals)')
    
    # Plot 3: PACF
    ax = axes[1, 0]
    pacf_vals = pacf(residuals, nlags=n_lags_plot)
    ax.bar(range(len(pacf_vals)), pacf_vals, color='steelblue', alpha=0.7)
    ax.axhline(y=conf, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-conf, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('PACF')
    ax.set_title('Partial Autocorrelation Function (Residuals)')
    
    # Plot 4: Residual histogram
    ax = axes[1, 1]
    ax.hist(residuals, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, norm.pdf(x, np.mean(residuals), np.std(residuals)), 'r-', lw=2, label='Normal')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend()
    
    plt.suptitle('Time Series Diagnostics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def adf_test(y: Union[pd.Series, np.ndarray], max_lag: Optional[int] = None) -> Dict[str, Any]:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    Parameters
    ----------
    y : array-like
        Time series data
    max_lag : int, optional
        Maximum lag for test
        
    Returns
    -------
    dict
        Test statistic, p-value, and conclusion
    """
    from statsmodels.tsa.stattools import adfuller
    
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    
    if max_lag is None:
        max_lag = int(np.sqrt(len(y)))
    
    result = adfuller(y, maxlag=max_lag)
    
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'stationary': result[1] < 0.05,
        'conclusion': 'Stationary' if result[1] < 0.05 else 'Non-stationary'
    }


def kpss_test(y: Union[pd.Series, np.ndarray], regression: str = 'c') -> Dict[str, Any]:
    """
    KPSS test for stationarity.
    
    Parameters
    ----------
    y : array-like
        Time series data
    regression : str, default='c'
        'c' for constant, 'ct' for constant and trend
        
    Returns
    -------
    dict
        Test statistic, p-value, and conclusion
    """
    from statsmodels.tsa.stattools import kpss
    
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(y, regression=regression)
    
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'critical_values': result[3],
        'stationary': result[1] > 0.05,
        'conclusion': 'Stationary' if result[1] > 0.05 else 'Non-stationary'
    }


def ljung_box_test(y: Union[pd.Series, np.ndarray], lags: int = 10) -> Dict[str, Any]:
    """
    Ljung-Box test for autocorrelation.
    
    Parameters
    ----------
    y : array-like
        Time series (or residuals)
    lags : int, default=10
        Number of lags
        
    Returns
    -------
    dict
        Test statistic, p-value, and conclusion
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    
    result = acorr_ljungbox(y, lags=[lags], return_df=True)
    
    return {
        'test_statistic': float(result['lb_stat'].iloc[0]),
        'p_value': float(result['lb_pvalue'].iloc[0]),
        'lags': lags,
        'no_autocorrelation': float(result['lb_pvalue'].iloc[0]) > 0.05,
        'conclusion': 'No significant autocorrelation' if float(result['lb_pvalue'].iloc[0]) > 0.05 
                      else 'Significant autocorrelation detected'
    }


def plot_acf_pacf(
    y: Union[pd.Series, np.ndarray],
    n_lags: int = 24,
    alpha: float = 0.05,
    figsize: Tuple[int, int] = (12, 5)
) -> Optional[Any]:
    """
    Plot ACF and PACF side by side.
    
    Parameters
    ----------
    y : array-like
        Time series data
    n_lags : int, default=24
        Number of lags
    alpha : float, default=0.05
        Significance level
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    if not HAS_MATPLOTLIB:
        return None
    
    from statsmodels.tsa.stattools import acf, pacf
    
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)
    n_lags = min(n_lags, n // 2 - 1)
    
    acf_vals = acf(y, nlags=n_lags, fft=True)
    pacf_vals = pacf(y, nlags=n_lags)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    conf = norm.ppf(1 - alpha / 2) / np.sqrt(n)
    
    ax = axes[0]
    ax.bar(range(len(acf_vals)), acf_vals, color='steelblue', alpha=0.7)
    ax.axhline(y=conf, color='red', linestyle='--', alpha=0.5, label=f'{int((1-alpha)*100)}% CI')
    ax.axhline(y=-conf, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_title('Autocorrelation Function')
    ax.legend()
    
    ax = axes[1]
    ax.bar(range(len(pacf_vals)), pacf_vals, color='steelblue', alpha=0.7)
    ax.axhline(y=conf, color='red', linestyle='--', alpha=0.5, label=f'{int((1-alpha)*100)}% CI')
    ax.axhline(y=-conf, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('PACF')
    ax.set_title('Partial Autocorrelation Function')
    ax.legend()
    
    plt.tight_layout()
    return fig


# =============================================================================
# BOOTSTRAP METHODS (Added v2.5.0)
# =============================================================================

@dataclass
class BootstrapResults:
    """
    Results container for bootstrap analysis (JMP-style).
    
    Attributes
    ----------
    statistic_name : str
        Name of the statistic being bootstrapped
    observed : float
        Observed value of the statistic
    boot_mean : float
        Mean of bootstrap distribution
    boot_std : float
        Standard deviation of bootstrap distribution (bootstrap SE)
    bias : float
        Estimated bias (boot_mean - observed)
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    ci_level : float
        Confidence level (e.g., 0.90 for 90%)
    ci_method : str
        Method used for CI ('percentile', 'bca', 'basic')
    n_bootstrap : int
        Number of bootstrap samples
    bootstrap_distribution : np.ndarray
        Array of bootstrap statistics
    """
    statistic_name: str
    observed: float
    boot_mean: float
    boot_std: float
    bias: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    ci_method: str
    n_bootstrap: int
    bootstrap_distribution: np.ndarray
    
    def __repr__(self):
        lines = []
        lines.append("=" * 60)
        lines.append(f"Bootstrap Results: {self.statistic_name}")
        lines.append("=" * 60)
        lines.append(f"Observed Value:        {self.observed:.6f}")
        lines.append(f"Bootstrap Mean:        {self.boot_mean:.6f}")
        lines.append(f"Bootstrap Std Error:   {self.boot_std:.6f}")
        lines.append(f"Bias:                  {self.bias:.6f}")
        lines.append("")
        lines.append(f"{self.ci_level*100:.0f}% Confidence Interval ({self.ci_method}):")
        lines.append(f"  Lower:               {self.ci_lower:.6f}")
        lines.append(f"  Upper:               {self.ci_upper:.6f}")
        lines.append(f"")
        lines.append(f"Number of Bootstraps:  {self.n_bootstrap}")
        return "\n".join(lines)


def bootstrap(
    data: Union[pd.DataFrame, np.ndarray],
    statistic_func: callable,
    n_bootstrap: int = 1000,
    ci_level: float = 0.90,
    ci_method: str = 'percentile',
    random_state: Optional[int] = None,
    statistic_name: str = 'Statistic'
) -> BootstrapResults:
    """
    Bootstrap confidence intervals for any statistic (JMP-style).
    
    Replicates JMP's Analyze > Distribution > Bootstrap or 
    the Bootstrap option available in various JMP platforms.
    
    Parameters
    ----------
    data : DataFrame or array
        Data to resample
    statistic_func : callable
        Function that takes data and returns a scalar statistic.
        E.g., lambda x: np.mean(x) or lambda x: np.std(x)
    n_bootstrap : int, default=1000
        Number of bootstrap resamples
    ci_level : float, default=0.90
        Confidence level (0.90 = 90% CI)
    ci_method : str, default='percentile'
        Method for CI: 'percentile', 'basic', or 'bca'
    random_state : int, optional
        Random seed for reproducibility
    statistic_name : str, default='Statistic'
        Name of the statistic (for display)
        
    Returns
    -------
    BootstrapResults
        Object containing bootstrap CI and distribution
        
    Examples
    --------
    >>> # Bootstrap mean
    >>> result = bootstrap(df['price'], lambda x: np.mean(x), ci_level=0.95)
    >>> print(result)
    
    >>> # Bootstrap correlation
    >>> result = bootstrap(df[['x', 'y']], lambda d: d['x'].corr(d['y']))
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert to numpy if needed
    if isinstance(data, pd.DataFrame):
        data_arr = data.values
        is_df = True
        columns = data.columns
        index_name = data.index.name
    elif isinstance(data, pd.Series):
        data_arr = data.values
        is_df = False
    else:
        data_arr = np.asarray(data)
        is_df = False
    
    n = len(data_arr)
    
    # Calculate observed statistic
    if is_df:
        observed = statistic_func(data)
    else:
        observed = statistic_func(data_arr)
    
    # Bootstrap resampling
    boot_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.randint(0, n, size=n)
        
        if is_df:
            boot_sample = pd.DataFrame(data_arr[indices], columns=columns)
            boot_stats[i] = statistic_func(boot_sample)
        else:
            boot_sample = data_arr[indices]
            boot_stats[i] = statistic_func(boot_sample)
    
    # Calculate bootstrap statistics
    boot_mean = np.mean(boot_stats)
    boot_std = np.std(boot_stats, ddof=1)
    bias = boot_mean - observed
    
    # Calculate confidence interval
    alpha = 1 - ci_level
    
    if ci_method == 'percentile':
        ci_lower = np.percentile(boot_stats, alpha/2 * 100)
        ci_upper = np.percentile(boot_stats, (1 - alpha/2) * 100)
    elif ci_method == 'basic':
        # Basic bootstrap: 2*observed - percentiles
        ci_lower = 2*observed - np.percentile(boot_stats, (1 - alpha/2) * 100)
        ci_upper = 2*observed - np.percentile(boot_stats, alpha/2 * 100)
    elif ci_method == 'bca':
        # BCa (bias-corrected and accelerated) - simplified version
        # Calculate z0 (bias correction)
        z0 = norm.ppf(np.mean(boot_stats < observed))
        
        # Calculate acceleration (jackknife)
        jack_stats = np.zeros(n)
        for j in range(n):
            jack_mask = np.ones(n, dtype=bool)
            jack_mask[j] = False
            if is_df:
                jack_sample = pd.DataFrame(data_arr[jack_mask], columns=columns)
                jack_stats[j] = statistic_func(jack_sample)
            else:
                jack_stats[j] = statistic_func(data_arr[jack_mask])
        
        jack_mean = np.mean(jack_stats)
        acc = np.sum((jack_mean - jack_stats)**3) / (6 * np.sum((jack_mean - jack_stats)**2)**1.5)
        
        # Adjusted percentiles
        z_alpha_lower = norm.ppf(alpha/2)
        z_alpha_upper = norm.ppf(1 - alpha/2)
        
        adj_lower = norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - acc * (z0 + z_alpha_lower)))
        adj_upper = norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - acc * (z0 + z_alpha_upper)))
        
        ci_lower = np.percentile(boot_stats, adj_lower * 100)
        ci_upper = np.percentile(boot_stats, adj_upper * 100)
    else:
        raise ValueError(f"Unknown ci_method: {ci_method}. Use 'percentile', 'basic', or 'bca'")
    
    return BootstrapResults(
        statistic_name=statistic_name,
        observed=observed,
        boot_mean=boot_mean,
        boot_std=boot_std,
        bias=bias,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        ci_method=ci_method,
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=boot_stats
    )


def bootstrap_rmse(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    n_bootstrap: int = 1000,
    ci_level: float = 0.90,
    ci_method: str = 'percentile',
    random_state: Optional[int] = None
) -> BootstrapResults:
    """
    Bootstrap confidence interval for RMSE (JMP-style).
    
    This replicates JMP's Bootstrap command for regression RMSE,
    which resamples the data, refits the model, and calculates RMSE
    for each bootstrap sample.
    
    Parameters
    ----------
    y : array-like
        Response variable
    X : DataFrame or array-like
        Predictor variables
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    ci_level : float, default=0.90
        Confidence level (0.90 = 90% CI)
    ci_method : str, default='percentile'
        Method for CI: 'percentile', 'basic', or 'bca'
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    BootstrapResults
        Bootstrap confidence interval for RMSE
        
    Examples
    --------
    >>> result = bootstrap_rmse(df['y'], df[['x1', 'x2']], ci_level=0.90)
    >>> print(f"90% CI: ({result.ci_lower:.4f}, {result.ci_upper:.4f})")
    
    Notes
    -----
    JMP Pro's Bootstrap command for regression refits the model on each
    bootstrap sample and computes the statistic. This function does the same.
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required for bootstrap_rmse. Install: pip install statsmodels")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert inputs
    y = np.asarray(y).flatten()
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
        feature_names = list(X.columns)
    else:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        feature_names = [f'X{i+1}' for i in range(X_arr.shape[1])]
    
    # Remove missing values
    mask = ~(np.isnan(y) | np.any(np.isnan(X_arr), axis=1))
    y_clean = y[mask]
    X_clean = X_arr[mask]
    
    n = len(y_clean)
    
    # Calculate observed RMSE
    X_design = sm.add_constant(X_clean)
    model = sm.OLS(y_clean, X_design).fit()
    observed_rmse = np.sqrt(model.mse_resid)
    
    # Bootstrap resampling
    boot_rmse = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.randint(0, n, size=n)
        y_boot = y_clean[indices]
        X_boot = X_clean[indices]
        
        # Fit model on bootstrap sample
        X_boot_design = sm.add_constant(X_boot)
        try:
            boot_model = sm.OLS(y_boot, X_boot_design).fit()
            boot_rmse[i] = np.sqrt(boot_model.mse_resid)
        except:
            boot_rmse[i] = np.nan
    
    # Remove any failed fits
    boot_rmse = boot_rmse[~np.isnan(boot_rmse)]
    
    # Calculate bootstrap statistics
    boot_mean = np.mean(boot_rmse)
    boot_std = np.std(boot_rmse, ddof=1)
    bias = boot_mean - observed_rmse
    
    # Calculate confidence interval
    alpha = 1 - ci_level
    
    if ci_method == 'percentile':
        ci_lower = np.percentile(boot_rmse, alpha/2 * 100)
        ci_upper = np.percentile(boot_rmse, (1 - alpha/2) * 100)
    elif ci_method == 'basic':
        ci_lower = 2*observed_rmse - np.percentile(boot_rmse, (1 - alpha/2) * 100)
        ci_upper = 2*observed_rmse - np.percentile(boot_rmse, alpha/2 * 100)
    else:
        ci_lower = np.percentile(boot_rmse, alpha/2 * 100)
        ci_upper = np.percentile(boot_rmse, (1 - alpha/2) * 100)
    
    return BootstrapResults(
        statistic_name='RMSE',
        observed=observed_rmse,
        boot_mean=boot_mean,
        boot_std=boot_std,
        bias=bias,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        ci_method=ci_method,
        n_bootstrap=len(boot_rmse),
        bootstrap_distribution=boot_rmse
    )


def plot_bootstrap_distribution(
    result: BootstrapResults,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[Any]:
    """
    Plot bootstrap distribution with confidence interval.
    
    Parameters
    ----------
    result : BootstrapResults
        Results from bootstrap() or bootstrap_rmse()
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram of bootstrap distribution
    ax.hist(result.bootstrap_distribution, bins=50, density=True, 
            alpha=0.7, color='steelblue', edgecolor='white')
    
    # Mark observed value
    ax.axvline(result.observed, color='red', linestyle='-', linewidth=2,
               label=f'Observed = {result.observed:.4f}')
    
    # Mark CI
    ax.axvline(result.ci_lower, color='green', linestyle='--', linewidth=2,
               label=f'{result.ci_level*100:.0f}% CI Lower = {result.ci_lower:.4f}')
    ax.axvline(result.ci_upper, color='green', linestyle='--', linewidth=2,
               label=f'{result.ci_level*100:.0f}% CI Upper = {result.ci_upper:.4f}')
    
    # Shade CI region
    ax.axvspan(result.ci_lower, result.ci_upper, alpha=0.2, color='green')
    
    ax.set_xlabel(result.statistic_name, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Bootstrap Distribution of {result.statistic_name}\n'
                 f'({result.n_bootstrap} resamples, {result.ci_method} CI)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig


# =============================================================================
# K NEAREST NEIGHBORS - JMP Style (Added v2.5.0)
# =============================================================================

@dataclass
class KNNResult:
    """
    Results container for K Nearest Neighbors analysis (JMP-style).
    
    Attributes
    ----------
    response_type : str
        'classification' or 'regression'
    optimal_k : int
        The K value with best performance on validation data
    k_values : np.ndarray
        Array of K values evaluated
    train_errors : np.ndarray
        Error rates/RMSE on training data for each K
    validation_errors : np.ndarray
        Error rates/RMSE on validation data for each K
    test_errors : np.ndarray or None
        Error rates/RMSE on test data for each K (if test set provided)
    confusion_matrix : pd.DataFrame or None
        Confusion matrix at optimal K (classification only)
    confusion_rates : pd.DataFrame or None
        Misclassification rates by class (classification only)
    model : object
        Fitted KNN model at optimal K
    scaler : object
        Fitted scaler used for feature standardization
    predictions : pd.Series
        Predictions on validation data at optimal K
    probabilities : pd.DataFrame or None
        Class probabilities (classification only)
    feature_names : list
        Names of features used
    classes : np.ndarray or None
        Class labels (classification only)
    summary_table : pd.DataFrame
        Summary of performance across all K values
    """
    response_type: str
    optimal_k: int
    k_values: np.ndarray
    train_errors: np.ndarray
    validation_errors: np.ndarray
    test_errors: Optional[np.ndarray]
    confusion_matrix: Optional[pd.DataFrame]
    confusion_rates: Optional[pd.DataFrame]
    model: object
    scaler: object
    predictions: pd.Series
    probabilities: Optional[pd.DataFrame]
    feature_names: list
    classes: Optional[np.ndarray]
    summary_table: pd.DataFrame
    X_train_scaled: np.ndarray = None
    y_train: np.ndarray = None
    
    def __repr__(self):
        lines = []
        lines.append("=" * 70)
        lines.append("K Nearest Neighbors Results")
        lines.append("=" * 70)
        lines.append(f"Response Type: {self.response_type.title()}")
        lines.append(f"Number of Features: {len(self.feature_names)}")
        lines.append(f"Features: {', '.join(self.feature_names[:5])}" + 
                    ("..." if len(self.feature_names) > 5 else ""))
        lines.append("")
        
        lines.append("-" * 70)
        lines.append("Optimal K Selection")
        lines.append("-" * 70)
        lines.append(f"Optimal K: {self.optimal_k}")
        
        opt_idx = list(self.k_values).index(self.optimal_k) if self.optimal_k in self.k_values else 0
        
        if self.response_type == 'classification':
            lines.append(f"Training Misclassification Rate: {self.train_errors[opt_idx]:.4f}")
            lines.append(f"Validation Misclassification Rate: {self.validation_errors[opt_idx]:.4f}")
            if self.test_errors is not None:
                lines.append(f"Test Misclassification Rate: {self.test_errors[opt_idx]:.4f}")
        else:
            lines.append(f"Training RMSE: {self.train_errors[opt_idx]:.4f}")
            lines.append(f"Validation RMSE: {self.validation_errors[opt_idx]:.4f}")
            if self.test_errors is not None:
                lines.append(f"Test RMSE: {self.test_errors[opt_idx]:.4f}")
        
        lines.append("")
        lines.append("-" * 70)
        lines.append("Performance Summary by K")
        lines.append("-" * 70)
        lines.append(self.summary_table.to_string())
        
        if self.response_type == 'classification' and self.confusion_matrix is not None:
            lines.append("")
            lines.append("-" * 70)
            lines.append(f"Confusion Matrix (K = {self.optimal_k})")
            lines.append("-" * 70)
            lines.append(self.confusion_matrix.to_string())
            lines.append("")
            lines.append("Confusion Rates:")
            lines.append(self.confusion_rates.to_string())
        
        return "\n".join(lines)
    
    def predict(self, X_new: pd.DataFrame) -> pd.Series:
        """Make predictions on new data."""
        X_scaled = self.scaler.transform(X_new[self.feature_names])
        preds = self.model.predict(X_scaled)
        return pd.Series(preds, index=X_new.index, name='Predicted')
    
    def predict_proba(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Get class probabilities for new data (classification only)."""
        if self.response_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        X_scaled = self.scaler.transform(X_new[self.feature_names])
        probs = self.model.predict_proba(X_scaled)
        return pd.DataFrame(probs, index=X_new.index, columns=self.classes)


def k_nearest_neighbors(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    validation_portion: float = 0.3,
    validation_column: Optional[pd.Series] = None,
    k_range: Optional[range] = None,
    max_k: int = 20,
    metric: str = 'euclidean',
    weights: str = 'uniform',
    random_state: Optional[int] = None,
    plot: bool = True,
    figsize: Tuple[int, int] = (12, 5)
) -> KNNResult:
    """
    K Nearest Neighbors analysis (JMP-style).
    
    Performs KNN classification or regression with automatic K selection
    based on validation data performance, replicating JMP's K Nearest 
    Neighbors platform (Analyze > Predictive Modeling > K Nearest Neighbors).
    
    Parameters
    ----------
    y : pd.Series or np.ndarray
        Response variable. Categorical for classification, continuous for regression.
    X : pd.DataFrame or np.ndarray
        Predictor variables (factors).
    validation_portion : float, default=0.3
        Proportion of data to use for validation (ignored if validation_column provided).
    validation_column : pd.Series, optional
        Column indicating train/validation/test split.
        Values: 0 or 'Training' = train, 1 or 'Validation' = validation, 
                2 or 'Test' = test
    k_range : range, optional
        Range of K values to evaluate. Default is range(1, max_k+1).
    max_k : int, default=20
        Maximum K to evaluate if k_range not specified.
    metric : str, default='euclidean'
        Distance metric. Options: 'euclidean', 'manhattan', 'minkowski'.
    weights : str, default='uniform'
        Weight function. 'uniform' = all neighbors weighted equally,
        'distance' = closer neighbors have more influence.
    random_state : int, optional
        Random seed for reproducibility.
    plot : bool, default=True
        Whether to display the K selection plot.
    figsize : tuple, default=(12, 5)
        Figure size for plots.
        
    Returns
    -------
    KNNResult
        Object containing all KNN analysis results including:
        - optimal_k: Best K based on validation error
        - confusion_matrix: For classification tasks
        - summary_table: Performance across all K values
        - model: Fitted model for predictions
        
    Examples
    --------
    >>> # Classification
    >>> result = k_nearest_neighbors(df['Risk'], df[['Income', 'Age', 'Credit']])
    >>> print(result)
    >>> print(f"Optimal K: {result.optimal_k}")
    >>> print(result.confusion_matrix)
    
    >>> # Regression
    >>> result = k_nearest_neighbors(df['Price'], df[['Size', 'Rooms', 'Age']])
    >>> print(f"Optimal K: {result.optimal_k}")
    >>> print(f"Validation RMSE: {result.validation_errors[result.optimal_k-1]:.4f}")
    
    >>> # With validation column (JMP-style)
    >>> result = k_nearest_neighbors(
    ...     df['Response'], df[predictors],
    ...     validation_column=df['Validation']
    ... )
    
    >>> # Make predictions on new data
    >>> new_preds = result.predict(new_data)
    >>> probs = result.predict_proba(new_data)  # classification only
    
    Notes
    -----
    - Features are automatically standardized (critical for KNN)
    - Optimal K is selected based on validation error (not training error)
    - Classification uses misclassification rate, regression uses RMSE
    - Use validation_column for JMP-style train/validation/test splits
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for KNN. Install with: pip install scikit-learn")
    
    # Convert to pandas if needed
    if isinstance(y, np.ndarray):
        y = pd.Series(y, name='Response')
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
    
    # Get feature names
    feature_names = list(X.columns)
    
    # Determine response type
    if y.dtype == 'object' or y.dtype.name == 'category' or y.nunique() <= 10:
        response_type = 'classification'
        classes = np.sort(y.unique())
    else:
        response_type = 'regression'
        classes = None
    
    # Handle train/validation/test split
    if validation_column is not None:
        val_col = validation_column.copy()
        if val_col.dtype == 'object':
            val_col = val_col.str.lower()
            train_mask = val_col.isin(['training', 'train', '0'])
            val_mask = val_col.isin(['validation', 'validate', '1'])
            test_mask = val_col.isin(['test', '2'])
        else:
            train_mask = val_col == 0
            val_mask = val_col == 1
            test_mask = val_col == 2
        
        X_train = X[train_mask].copy()
        y_train = y[train_mask].copy()
        X_val = X[val_mask].copy()
        y_val = y[val_mask].copy()
        
        if test_mask.sum() > 0:
            X_test = X[test_mask].copy()
            y_test = y[test_mask].copy()
            has_test = True
        else:
            X_test, y_test, has_test = None, None, False
    else:
        if random_state is None:
            random_state = 42
        X_train, X_val, y_train, y_val = sklearn_train_test_split(
            X, y, test_size=validation_portion, random_state=random_state,
            stratify=y if response_type == 'classification' else None
        )
        X_test, y_test, has_test = None, None, False
    
    # Standardize features (critical for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    if has_test:
        X_test_scaled = scaler.transform(X_test)
    
    # Set K range
    if k_range is None:
        max_k = min(max_k, len(X_train) - 1)
        k_range = range(1, max_k + 1)
    
    k_values = np.array(list(k_range))
    
    # Evaluate across K values
    train_errors, val_errors = [], []
    test_errors = [] if has_test else None
    
    for k in k_values:
        if response_type == 'classification':
            model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
            model.fit(X_train_scaled, y_train)
            train_errors.append(np.mean(model.predict(X_train_scaled) != y_train))
            val_errors.append(np.mean(model.predict(X_val_scaled) != y_val))
            if has_test:
                test_errors.append(np.mean(model.predict(X_test_scaled) != y_test))
        else:
            model = KNeighborsRegressor(n_neighbors=k, metric=metric, weights=weights)
            model.fit(X_train_scaled, y_train)
            train_errors.append(np.sqrt(np.mean((model.predict(X_train_scaled) - y_train) ** 2)))
            val_errors.append(np.sqrt(np.mean((model.predict(X_val_scaled) - y_val) ** 2)))
            if has_test:
                test_errors.append(np.sqrt(np.mean((model.predict(X_test_scaled) - y_test) ** 2)))
    
    train_errors = np.array(train_errors)
    val_errors = np.array(val_errors)
    if has_test:
        test_errors = np.array(test_errors)
    
    # Find optimal K (minimum validation error)
    optimal_idx = np.argmin(val_errors)
    optimal_k = k_values[optimal_idx]
    
    # Fit final model with optimal K
    if response_type == 'classification':
        final_model = KNeighborsClassifier(n_neighbors=optimal_k, metric=metric, weights=weights)
    else:
        final_model = KNeighborsRegressor(n_neighbors=optimal_k, metric=metric, weights=weights)
    final_model.fit(X_train_scaled, y_train)
    
    # Predictions at optimal K
    val_predictions = pd.Series(final_model.predict(X_val_scaled), index=X_val.index, name='Predicted')
    
    if response_type == 'classification':
        val_probabilities = pd.DataFrame(
            final_model.predict_proba(X_val_scaled), index=X_val.index, columns=classes
        )
        # Confusion matrix
        conf_matrix = pd.crosstab(y_val, val_predictions, rownames=['Actual'], colnames=['Predicted'])
        for cls in classes:
            if cls not in conf_matrix.index:
                conf_matrix.loc[cls] = 0
            if cls not in conf_matrix.columns:
                conf_matrix[cls] = 0
        conf_matrix = conf_matrix.loc[classes, classes]
        
        # Confusion rates (misclassification by class)
        rates_data = []
        for cls in classes:
            actual_count = (y_val == cls).sum()
            if actual_count > 0:
                correct = conf_matrix.loc[cls, cls]
                incorrect = actual_count - correct
                rates_data.append({
                    'Actual': cls, 'Count': actual_count, 'Correct': correct,
                    'Incorrect': incorrect, 'Correct Rate': correct / actual_count,
                    'Misclassification Rate': incorrect / actual_count
                })
        confusion_rates = pd.DataFrame(rates_data)
    else:
        val_probabilities, conf_matrix, confusion_rates = None, None, None
    
    # Summary table
    summary_data = {'K': k_values, 'Training': train_errors, 'Validation': val_errors}
    if has_test:
        summary_data['Test'] = test_errors
    summary_table = pd.DataFrame(summary_data)
    if response_type == 'classification':
        summary_table.columns = ['K', 'Train Misclass Rate', 'Valid Misclass Rate'] + \
                               (['Test Misclass Rate'] if has_test else [])
    else:
        summary_table.columns = ['K', 'Train RMSE', 'Valid RMSE'] + \
                               (['Test RMSE'] if has_test else [])
    
    # Create result object
    result = KNNResult(
        response_type=response_type, optimal_k=optimal_k, k_values=k_values,
        train_errors=train_errors, validation_errors=val_errors, test_errors=test_errors,
        confusion_matrix=conf_matrix, confusion_rates=confusion_rates,
        model=final_model, scaler=scaler, predictions=val_predictions,
        probabilities=val_probabilities, feature_names=feature_names,
        classes=classes, summary_table=summary_table
    )
    result.X_train_scaled = X_train_scaled
    result.y_train = np.array(y_train)
    
    if plot and HAS_MATPLOTLIB:
        plot_knn_results(result, figsize=figsize)
    
    return result


def plot_knn_results(result: KNNResult, figsize: Tuple[int, int] = (12, 5)) -> Optional[Any]:
    """
    Plot K Nearest Neighbors results (JMP-style).
    
    Creates plots showing error rates across K values and confusion matrix.
    
    Parameters
    ----------
    result : KNNResult
        Results from k_nearest_neighbors()
    figsize : tuple, default=(12, 5)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None
    
    n_plots = 2 if result.response_type == 'classification' and result.confusion_matrix is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: Error rates vs K
    ax1 = axes[0]
    ax1.plot(result.k_values, result.train_errors, 'b-o', label='Training', markersize=5, linewidth=1.5)
    ax1.plot(result.k_values, result.validation_errors, 'r-s', label='Validation', markersize=5, linewidth=1.5)
    if result.test_errors is not None:
        ax1.plot(result.k_values, result.test_errors, 'g-^', label='Test', markersize=5, linewidth=1.5)
    
    # Mark optimal K
    opt_idx = list(result.k_values).index(result.optimal_k) if result.optimal_k in result.k_values else 0
    ax1.axvline(x=result.optimal_k, color='gray', linestyle='--', alpha=0.7, label=f'Optimal K = {result.optimal_k}')
    ax1.scatter([result.optimal_k], [result.validation_errors[opt_idx]], 
                s=150, c='red', marker='*', zorder=5, edgecolors='black')
    
    ylabel = 'Misclassification Rate' if result.response_type == 'classification' else 'RMSE'
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_xlabel('Number of Neighbors (K)', fontsize=11)
    ax1.set_title(f'K Nearest Neighbors - {ylabel} by K', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confusion matrix (classification only)
    if result.response_type == 'classification' and n_plots > 1:
        ax2 = axes[1]
        conf_normalized = result.confusion_matrix.div(
            result.confusion_matrix.sum(axis=1), axis=0
        ).fillna(0)
        im = ax2.imshow(conf_normalized.values, cmap='Blues', aspect='auto')
        
        for i in range(len(result.classes)):
            for j in range(len(result.classes)):
                count = result.confusion_matrix.iloc[i, j]
                pct = conf_normalized.iloc[i, j] * 100
                color = 'white' if conf_normalized.iloc[i, j] > 0.5 else 'black'
                ax2.text(j, i, f'{count}\n({pct:.1f}%)', ha='center', va='center', 
                        color=color, fontsize=9)
        
        ax2.set_xticks(range(len(result.classes)))
        ax2.set_yticks(range(len(result.classes)))
        ax2.set_xticklabels(result.classes, fontsize=10)
        ax2.set_yticklabels(result.classes, fontsize=10)
        ax2.set_xlabel('Predicted', fontsize=11)
        ax2.set_ylabel('Actual', fontsize=11)
        ax2.set_title(f'Confusion Matrix (K = {result.optimal_k})', fontsize=12)
        plt.colorbar(im, ax=ax2, label='Proportion')
    
    plt.tight_layout()
    plt.show()
    return fig


def knn_classification(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    validation_portion: float = 0.3,
    validation_column: Optional[pd.Series] = None,
    k_range: Optional[range] = None,
    max_k: int = 20,
    metric: str = 'euclidean',
    weights: str = 'uniform',
    random_state: Optional[int] = None,
    plot: bool = True
) -> KNNResult:
    """
    K Nearest Neighbors Classification (JMP-style).
    
    Convenience wrapper for k_nearest_neighbors() for classification tasks.
    
    Parameters
    ----------
    y : pd.Series or np.ndarray
        Categorical response variable
    X : pd.DataFrame or np.ndarray
        Predictor variables
    validation_portion : float, default=0.3
        Proportion for validation
    validation_column : pd.Series, optional
        Column indicating train/validation/test
    k_range : range, optional
        Range of K values to evaluate
    max_k : int, default=20
        Maximum K to evaluate
    metric : str, default='euclidean'
        Distance metric
    weights : str, default='uniform'
        Weight function
    random_state : int, optional
        Random seed
    plot : bool, default=True
        Whether to display plots
        
    Returns
    -------
    KNNResult
        Classification results
        
    Examples
    --------
    >>> result = knn_classification(df['Risk'], df[['Income', 'Age']])
    >>> print(f"Optimal K: {result.optimal_k}")
    >>> print(result.confusion_matrix)
    """
    return k_nearest_neighbors(
        y=y, X=X, validation_portion=validation_portion,
        validation_column=validation_column, k_range=k_range, max_k=max_k,
        metric=metric, weights=weights, random_state=random_state, plot=plot
    )


def knn_regression(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    validation_portion: float = 0.3,
    validation_column: Optional[pd.Series] = None,
    k_range: Optional[range] = None,
    max_k: int = 20,
    metric: str = 'euclidean',
    weights: str = 'distance',
    random_state: Optional[int] = None,
    plot: bool = True
) -> KNNResult:
    """
    K Nearest Neighbors Regression (JMP-style).
    
    Convenience wrapper for k_nearest_neighbors() for regression tasks.
    
    Parameters
    ----------
    y : pd.Series or np.ndarray
        Continuous response variable
    X : pd.DataFrame or np.ndarray
        Predictor variables
    validation_portion : float, default=0.3
        Proportion for validation
    validation_column : pd.Series, optional
        Column indicating train/validation/test
    k_range : range, optional
        Range of K values to evaluate
    max_k : int, default=20
        Maximum K to evaluate
    metric : str, default='euclidean'
        Distance metric
    weights : str, default='distance'
        Weight function (default 'distance' for regression)
    random_state : int, optional
        Random seed
    plot : bool, default=True
        Whether to display plots
        
    Returns
    -------
    KNNResult
        Regression results
        
    Examples
    --------
    >>> result = knn_regression(df['Price'], df[['Size', 'Rooms']])
    >>> print(f"Optimal K: {result.optimal_k}")
    >>> print(f"Validation RMSE: {result.validation_errors[result.optimal_k-1]:.4f}")
    """
    return k_nearest_neighbors(
        y=y, X=X, validation_portion=validation_portion,
        validation_column=validation_column, k_range=k_range, max_k=max_k,
        metric=metric, weights=weights, random_state=random_state, plot=plot
    )


def save_knn_predictions(
    result: KNNResult,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Save KNN predictions to a DataFrame (like JMP's Save Predicteds).
    
    Parameters
    ----------
    result : KNNResult
        Fitted KNN result from k_nearest_neighbors()
    X : pd.DataFrame
        Data to predict on (must have same columns as training data)
    y : pd.Series, optional
        Actual values (for computing residuals/accuracy)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predictions, probabilities, and optionally residuals
        
    Examples
    --------
    >>> result = k_nearest_neighbors(df['y'], df[predictors])
    >>> predictions_df = save_knn_predictions(result, df[predictors], df['y'])
    >>> print(predictions_df.head())
    """
    X_scaled = result.scaler.transform(X[result.feature_names])
    predictions = result.model.predict(X_scaled)
    
    output = pd.DataFrame(index=X.index)
    output['Predicted'] = predictions
    
    if result.response_type == 'classification':
        probs = result.model.predict_proba(X_scaled)
        for i, cls in enumerate(result.classes):
            output[f'Prob[{cls}]'] = probs[:, i]
        output['Most Likely'] = predictions
        if y is not None:
            output['Actual'] = y.values
            output['Correct'] = (predictions == y.values).astype(int)
    else:
        if y is not None:
            output['Actual'] = y.values
            output['Residual'] = y.values - predictions
    
    return output


def knn_lift_curve(
    result: KNNResult,
    y_actual: pd.Series,
    positive_class: Optional[str] = None,
    n_deciles: int = 10,
    plot: bool = True
) -> pd.DataFrame:
    """
    Create lift curve for KNN classification (JMP-style).
    
    Parameters
    ----------
    result : KNNResult
        Fitted KNN classification result
    y_actual : pd.Series
        Actual class labels for validation set
    positive_class : str, optional
        The positive class to measure lift for.
        Defaults to the last class alphabetically.
    n_deciles : int, default=10
        Number of deciles for lift calculation
    plot : bool, default=True
        Whether to display lift curve
        
    Returns
    -------
    pd.DataFrame
        Lift curve data by decile
        
    Examples
    --------
    >>> result = k_nearest_neighbors(df['Risk'], df[predictors])
    >>> lift_data = knn_lift_curve(result, y_val, positive_class='Bad')
    >>> print(lift_data)
    """
    if result.response_type != 'classification':
        raise ValueError("Lift curves only available for classification")
    if positive_class is None:
        positive_class = result.classes[-1]
    if result.probabilities is None:
        raise ValueError("Probabilities not available")
    
    probs = result.probabilities[positive_class].values
    actual = (y_actual == positive_class).astype(int).values
    sorted_idx = np.argsort(-probs)
    actual_sorted = actual[sorted_idx]
    
    n = len(actual)
    decile_size = n // n_deciles
    lift_data = []
    cumulative_positives = 0
    baseline_rate = actual.mean()
    
    for i in range(n_deciles):
        start = i * decile_size
        end = (i + 1) * decile_size if i < n_deciles - 1 else n
        decile_positives = actual_sorted[start:end].sum()
        cumulative_positives += decile_positives
        decile_rate = decile_positives / (end - start)
        cumulative_rate = cumulative_positives / end
        lift_data.append({
            'Decile': i + 1,
            'Decile_Positive_Rate': decile_rate,
            'Cumulative_Positive_Rate': cumulative_rate,
            'Decile_Lift': decile_rate / baseline_rate if baseline_rate > 0 else 0,
            'Cumulative_Lift': cumulative_rate / baseline_rate if baseline_rate > 0 else 0,
            'Cumulative_Pct_Population': (i + 1) / n_deciles * 100,
            'Cumulative_Pct_Positives': cumulative_positives / actual.sum() * 100 if actual.sum() > 0 else 0
        })
    
    lift_df = pd.DataFrame(lift_data)
    
    if plot and HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax1, ax2 = axes
        
        ax1.plot(lift_df['Decile'], lift_df['Cumulative_Lift'], 'b-o', 
                 linewidth=2, markersize=8, label='Model')
        ax1.axhline(y=1, color='gray', linestyle='--', label='Baseline')
        ax1.set_xlabel('Decile', fontsize=11)
        ax1.set_ylabel('Cumulative Lift', fontsize=11)
        ax1.set_title('Lift Curve', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(1, n_deciles + 1))
        
        ax2.plot(lift_df['Cumulative_Pct_Population'], 
                 lift_df['Cumulative_Pct_Positives'], 'b-o',
                 linewidth=2, markersize=8, label='Model')
        ax2.plot([0, 100], [0, 100], 'gray', linestyle='--', label='Baseline')
        ax2.set_xlabel('% Population', fontsize=11)
        ax2.set_ylabel(f'% {positive_class} Captured', fontsize=11)
        ax2.set_title('Cumulative Gains Chart', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return lift_df


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__version__ = "2.5.0"
__all__ = [
    # Data classes
    'DescriptiveStats', 'NormalityTest', 'RegressionResults', 'ResidualDiagnostics',
    'CorrelationResults', 'ANOVAResults', 'TTestResults', 
    'HatMatrixResults', 'SubsetRegressionResults', 'StepwiseResults',
    'CovariateCombinations', 'PredictionIntervalResults',
    'TrainTestSplit', 'ModelValidationResults', 'ModelComparisonResults',
    
    # NEW v2.0 Data classes
    'PredictionProfiler', 'ARIMAResults', 'ExponentialSmoothingResults',
    'TimeSeriesDecomposition', 'AutocorrelationResults',
    'FactorialDesignResults', 'EffectScreeningResults', 'ResponseSurfaceResults',
    
    # Data import
    'read_csv', 'read_excel',
    
    # Core analysis
    'describe', 'test_normality', 'linear_regression', 'linear_regression_formula',
    'residual_diagnostics', 'prediction_interval', 'correlation', 'correlation_matrix', 
    'oneway_anova', 'ttest_1sample', 'ttest_2sample', 'ttest_paired',
    'normal_probability',
    
    # Hat matrix and influence
    'hat_matrix',
    
    # Model selection
    'subset_regression', 'stepwise_regression', 
    'stepwise_regression_enhanced', 'compare_stepwise_criteria',
    
    # Covariate combinations / feature engineering
    'covariate_combinations', 'full_factorial_design', 'polynomial_features',
    
    # Train/test validation (JMP-style)
    'train_test_split', 'make_validation_column', 'validate_model', 'compare_models', 
    'compare_all_criteria', 'plot_train_test_comparison', 'plot_model_comparison',
    
    # Convenience functions
    'fit_y_by_x', 'fit_model', 'distribution_analysis', 'multivariate_analysis',
    
    # Visualization
    'plot_distribution', 'plot_regression_diagnostics', 'plot_scatter_with_regression',
    'plot_correlation_matrix', 'plot_control_chart',
    
    # NEW v2.0: Interactive Visualization / Leverage Plots
    'plot_leverage_interactive', 'plot_influence_dashboard', 'plot_partial_regression',
    'plot_added_variable', 'plot_component_residual',
    
    # NEW v2.0: Prediction Profiler
    'prediction_profiler', 'plot_prediction_profiler',
    
    # NEW v2.0: Design of Experiments (DoE)
    'fractional_factorial_design', 'response_surface_design', 'optimal_design',
    'analyze_factorial', 'effect_screening', 'interaction_plot', 'main_effects_plot',
    'contour_profiler', 'pareto_of_effects',
    
    # NEW v2.0: Time Series Analysis
    'arima', 'exponential_smoothing', 'autocorrelation_analysis',
    'seasonal_decomposition', 'time_series_forecast', 'plot_time_series_diagnostics',
    'adf_test', 'kpss_test', 'ljung_box_test', 'plot_acf_pacf',
    
    # NEW v2.5.0: K Nearest Neighbors
    'KNNResult', 'k_nearest_neighbors', 'knn_classification', 'knn_regression',
    'plot_knn_results', 'save_knn_predictions', 'knn_lift_curve',
    
    # NEW v2.5.0: Bootstrap Methods
    'BootstrapResults', 'bootstrap', 'bootstrap_rmse', 'plot_bootstrap_distribution',
    
    # Utilities
    'detect_outliers', 'recode', 'log_transform',
]
