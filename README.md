# JMP-Stats: JMP-Style Statistical Analysis for Python

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version 2.2.0](https://img.shields.io/badge/version-2.2.0-green.svg)](https://github.com/siegfrkn/STAT-7220-JMP-Statistics-for-Python)

A comprehensive Python library that replicates JMP's statistical analysis capabilities for predictive analytics. Designed for students and practitioners who want to perform the same analyses in Python that they would do in JMP.

**Perfect for:** STAT 7220 - Predictive Analytics, STAT 7320, and similar courses.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
  - [Data Import](#data-import)
  - [Descriptive Statistics](#descriptive-statistics)
  - [Linear Regression](#linear-regression)
  - [Prediction Intervals](#prediction-intervals)
  - [Model Selection](#model-selection)
  - [Train/Test Validation](#traintest-validation)
  - [Influence Diagnostics](#influence-diagnostics)
  - [Prediction Profiler](#prediction-profiler)
  - [Design of Experiments](#design-of-experiments)
  - [Time Series Analysis](#time-series-analysis)
- [Function Reference](#function-reference)
- [Examples](#examples)
- [Tips & Best Practices](#tips--best-practices)
- [Future Development](#future-development)

---

## Installation

### Requirements

```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn
```

### Setup

Simply place `jmp_stats.py` in your working directory or Python path, then import:

```python
import jmp_stats as jmp
```

---

## Quick Start

```python
import jmp_stats as jmp
import pandas as pd

# Load data
df = jmp.read_csv('your_data.csv')

# Quick descriptive statistics
stats = jmp.describe(df['response'])
print(stats)

# Simple linear regression
results = jmp.linear_regression(df['y'], df[['x1', 'x2', 'x3']])
print(results)

# Prediction intervals for new observations
pi = jmp.prediction_interval(df['y'], df[['x1', 'x2']], X_new)
print(pi.to_dataframe())

# Stepwise regression with BIC
step = jmp.stepwise_regression_enhanced(df['y'], df[predictors], criterion='bic')
print(step)

# Train/test validation
comparison = jmp.compare_all_criteria(df['y'], df[predictors], test_size=0.3)
print(comparison)
```

---

## Core Features

### Data Import

Import data with automatic type detection and summary statistics.

```python
# CSV files
df = jmp.read_csv('data.csv')

# Excel files
df = jmp.read_excel('data.xlsx', sheet_name='Sheet1')
```

**Output includes:**
- Row and column counts
- Column types (numeric, categorical, datetime)
- Missing value summary
- Column listing with data types

---

### Descriptive Statistics

Get JMP-style distribution statistics for any numeric variable.

```python
stats = jmp.describe(df['price'])
print(stats)
```

**Output includes:**
- N, Mean, Std Dev, Std Err
- Min, Max, Range
- Quartiles (Q1, Median, Q3, IQR)
- Skewness, Kurtosis
- Percentiles (1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%)

```python
# Full distribution analysis with plots
analysis = jmp.distribution_analysis(df['price'], plot=True)
```

---

### Linear Regression

Perform regression with full JMP-style output.

```python
results = jmp.linear_regression(df['y'], df[['x1', 'x2', 'x3']])
print(results)
```

**Output includes:**
- R-Squared, Adjusted R-Squared
- RMSE, MAE, MAPE
- ANOVA table (SS, MS, F-ratio, p-value)
- Parameter estimates with t-ratios, p-values, and confidence intervals
- VIF (Variance Inflation Factors) for multicollinearity
- Durbin-Watson statistic

```python
# Get residual diagnostics
diag = jmp.residual_diagnostics(df['y'], df[['x1', 'x2']])
print(diag)
```

**Diagnostic tests include:**
- Breusch-Pagan test (heteroscedasticity)
- White test (heteroscedasticity)
- Shapiro-Wilk test (normality of residuals)
- Durbin-Watson (autocorrelation)
- Leverage points and influential observations

---

### Prediction Intervals

Calculate prediction intervals for new observations. Based on ISLP pg 110: "Prediction intervals will always be wider than confidence intervals because they account for the uncertainty associated with epsilon, the irreducible error."

```python
# Forecast with prediction intervals
X_new = pd.DataFrame({'x1': [10, 15], 'x2': [5, 8]})

results = jmp.prediction_interval(
    y=df['y'], 
    X=df[['x1', 'x2']], 
    X_new=X_new,
    alpha=0.05  # 95% intervals
)

print(results.to_dataframe())
```

**Output includes:**
- Point predictions
- SE for fitted value (confidence interval)
- SE for prediction (prediction interval)
- Lower/Upper bounds for both CI and PI

**Key distinction:**
- **Confidence Interval (CI)**: Where the true mean response is expected to fall
- **Prediction Interval (PI)**: Where an individual observation is expected to fall (always wider)

---

### Model Selection

#### Stepwise Regression

```python
# Forward selection with AIC
step = jmp.stepwise_regression_enhanced(
    df['y'], df[predictors],
    direction='forward',    # 'forward', 'backward', or 'both'
    criterion='aic'         # 'aic', 'bic', 'pvalue', 'adj_rsq', 'rmse', 'cp', 'cv'
)
print(step)
print(f"Selected features: {step.selected_features}")
```

**Available stopping criteria:**

| Criterion | Description |
|-----------|-------------|
| `'pvalue'` | P-value threshold (add if p < 0.05, remove if p > 0.10) |
| `'aic'` | Minimize Akaike Information Criterion |
| `'bic'` | Minimize Bayesian Information Criterion |
| `'adj_rsq'` | Maximize Adjusted R-squared |
| `'rmse'` | Minimize Root Mean Square Error |
| `'cp'` | Minimize Mallows' Cp |
| `'cv'` | Minimize Cross-Validated RMSE |

#### All Subsets Regression

```python
subsets = jmp.subset_regression(df['y'], df[predictors], max_vars=10)
print(subsets)
print(subsets.best_overall)  # Best model by each criterion
```

#### Compare All Criteria

```python
# Run all stepwise criteria and compare
comparison = jmp.compare_stepwise_criteria(df['y'], df[predictors])
print(comparison)
```

---

### Train/Test Validation

Split data and validate models like JMP's validation column approach.

```python
# Split data (70% train, 30% test)
split = jmp.train_test_split(df['y'], df[predictors], test_size=0.3, random_state=42)
print(split)

# Validate a single model
results = jmp.validate_model(
    split.y_train, split.X_train,
    split.y_test, split.X_test,
    features=['x1', 'x2', 'x3'],
    model_name="My Model"
)
print(results)
```

#### Compare Multiple Models

```python
models = {
    'Full Model': list(predictors),
    'Stepwise BIC': step_bic.selected_features,
    'Stepwise AIC': step_aic.selected_features,
    'Simple': ['x1', 'x2']
}

comparison = jmp.compare_models(
    split.y_train, split.X_train,
    split.y_test, split.X_test,
    models
)
print(comparison)

# Visualize comparison
jmp.plot_model_comparison(comparison)
```

> **Note:** The "best" model selection automatically excludes Full Model variants to prevent selecting overfit models. Full Model metrics are still shown in the comparison table for reference.

#### One-Step Comparison

```python
# Run all stepwise criteria and compare on train/test
comparison = jmp.compare_all_criteria(
    df['y'], df[predictors],
    test_size=0.3,
    random_state=42
)
print(comparison)
jmp.plot_model_comparison(comparison)
```

---

### Influence Diagnostics

#### Hat Matrix and Influence Measures

```python
influence = jmp.hat_matrix(df['y'], df[['x1', 'x2', 'x3']])
print(influence)

# Get detailed DataFrame
influence_df = influence.to_dataframe()
print(influence_df[influence_df['High_Influence']])  # Show influential points
```

**Includes:**
- Leverage (hat values)
- Cook's Distance
- DFFITS
- DFBETAS
- Studentized residuals

#### Leverage Plots (JMP-Style)

```python
# Interactive leverage plots for each predictor
jmp.plot_leverage_interactive(df['y'], df[['x1', 'x2', 'x3']])

# Comprehensive influence dashboard
jmp.plot_influence_dashboard(df['y'], df[['x1', 'x2', 'x3']])

# Added variable plots
jmp.plot_added_variable(df['y'], df[predictors])

# Component-plus-residual plots (detect non-linearity)
jmp.plot_component_residual(df['y'], df[predictors])
```

---

### Prediction Profiler

Create JMP-style prediction profilers to explore how predictions change with inputs.

```python
# Create profiler
profiler = jmp.prediction_profiler(
    df['sales'], 
    df[['price', 'advertising', 'store_size']],
    y_name='Sales'
)
print(profiler)

# Make predictions at specific values
result = profiler.predict_at(price=25.0, advertising=5000, store_size=2000)
print(f"Predicted sales: {result['prediction']:.2f}")
print(f"95% CI: {result['conf_interval']}")
print(f"95% PI: {result['pred_interval']}")

# Visualize the profiler
jmp.plot_prediction_profiler(profiler)
```

---

### Design of Experiments

#### Factorial Designs

```python
# Define factors with (low, high) levels
factors = {
    'Temperature': (150, 200),
    'Pressure': (10, 20),
    'Time': (30, 60)
}

# Full factorial design
design = jmp.full_factorial_design(factors)

# Fractional factorial design
design = jmp.fractional_factorial_design(
    factors, 
    resolution=4,      # Resolution III, IV, or V
    center_points=3,   # Add center points
    replicates=2       # Number of replicates
)
print(design)
```

#### Response Surface Designs

```python
factors = {'X1': (-1, 1), 'X2': (-1, 1), 'X3': (-1, 1)}

# Central Composite Design
rsd = jmp.response_surface_design(
    factors,
    design_type='ccd',   # 'ccd', 'ccf' (face-centered), 'cci' (inscribed), 'bbd' (Box-Behnken)
    center_points=5
)
print(rsd.design_matrix)
```

#### Optimal Designs

```python
# D-optimal design
design = jmp.optimal_design(
    factors,
    n_runs=15,
    model_type='quadratic',  # 'linear', 'interaction', 'quadratic'
    criterion='d-optimal'    # 'd-optimal' or 'i-optimal'
)
```

#### Analyze Factorial Experiments

```python
# After running the experiment and collecting response data
results = jmp.analyze_factorial(
    design,                    # Design matrix with response
    response='Yield',          # Response column name
    factors=['Temp', 'Pressure', 'Time'],
    include_interactions=True
)
print(results)
print(f"Significant effects: {results.significant_effects}")
```

#### Effect Screening

```python
screening = jmp.effect_screening(
    design, 
    response='Yield',
    factors=['A', 'B', 'C', 'D'],
    plot=True  # Creates Pareto and half-normal plots
)
print(f"Significant factors: {screening.significant}")
```

#### Visualization

```python
# Main effects plot
jmp.main_effects_plot(design, 'Yield', ['Temp', 'Pressure', 'Time'])

# Interaction plot
jmp.interaction_plot(design, 'Yield', 'Temp', 'Pressure')

# Contour profiler (response surface)
jmp.contour_profiler(
    model,              # Fitted model
    'Temp', 'Pressure', # X and Y axes
    factor_ranges,      # Dict of factor ranges
    response_name='Yield'
)

# Pareto chart of effects
jmp.pareto_of_effects(results.effects, threshold=2.0)
```

---

### Time Series Analysis

#### ARIMA Modeling

```python
# Automatic ARIMA with order selection
result = jmp.arima(
    df['sales'],
    auto_select=True,    # Automatically find best (p,d,q)
    max_p=5, max_q=5,    # Search limits
    forecast_steps=12    # Forecast 12 periods ahead
)
print(result)

# Manual order specification
result = jmp.arima(df['sales'], order=(1, 1, 1), forecast_steps=12)

# SARIMA (seasonal)
result = jmp.arima(
    df['sales'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),  # (P, D, Q, s)
    forecast_steps=12
)

# Get forecasts
forecast, conf_int = result.predict(steps=24)
```

#### Exponential Smoothing

```python
# Automatic method selection
result = jmp.exponential_smoothing(df['sales'], method='auto', forecast_steps=12)

# Specific methods
result = jmp.exponential_smoothing(df['sales'], method='simple')      # Simple ES
result = jmp.exponential_smoothing(df['sales'], method='holt')        # Holt's linear trend
result = jmp.exponential_smoothing(
    df['sales'], 
    method='holt-winters',
    seasonal_periods=12,   # Monthly data with yearly seasonality
    forecast_steps=12
)
print(result)
```

#### Autocorrelation Analysis

```python
# ACF/PACF analysis with suggested orders
acf_result = jmp.autocorrelation_analysis(df['sales'], n_lags=24)
print(acf_result)
print(f"Suggested AR order: {acf_result.suggested_ar_order}")
print(f"Suggested MA order: {acf_result.suggested_ma_order}")

# Plot ACF and PACF
jmp.plot_acf_pacf(df['sales'], n_lags=24)
```

#### Stationarity Tests

```python
# Augmented Dickey-Fuller test
adf = jmp.adf_test(df['sales'])
print(f"ADF Statistic: {adf['test_statistic']:.4f}")
print(f"P-value: {adf['p_value']:.4f}")
print(f"Conclusion: {adf['conclusion']}")

# KPSS test
kpss = jmp.kpss_test(df['sales'])
print(f"Conclusion: {kpss['conclusion']}")

# Ljung-Box test (for residual autocorrelation)
lb = jmp.ljung_box_test(residuals, lags=10)
print(f"Conclusion: {lb['conclusion']}")
```

#### Seasonal Decomposition

```python
decomp = jmp.seasonal_decomposition(
    df['sales'],
    period=12,           # Seasonal period
    method='additive'    # 'additive' or 'multiplicative'
)
print(decomp)
# Access components: decomp.trend, decomp.seasonal, decomp.residual
```

#### Time Series Diagnostics

```python
# Comprehensive diagnostic plots
jmp.plot_time_series_diagnostics(df['sales'], model_results=arima_result)
```

#### Unified Forecasting

```python
# Automatic model selection and forecasting
forecast = jmp.time_series_forecast(
    df['sales'],
    steps=12,
    method='auto',        # Chooses between ARIMA and ETS
    seasonal_periods=12
)
print(f"Method used: {forecast['method']}")
print(f"Forecasts: {forecast['forecast']}")
```

---

## Function Reference

### Data Import & Utilities

| Function | Description |
|----------|-------------|
| `read_csv()` | Import CSV with automatic type detection |
| `read_excel()` | Import Excel files |
| `recode()` | Recode values in a series |
| `log_transform()` | Apply log transformation |
| `detect_outliers()` | Detect outliers using IQR or z-score |

### Descriptive Statistics

| Function | Description |
|----------|-------------|
| `describe()` | Comprehensive descriptive statistics |
| `test_normality()` | Shapiro-Wilk, D'Agostino, Jarque-Bera tests |
| `distribution_analysis()` | Full distribution analysis with plots |

### Hypothesis Testing

| Function | Description |
|----------|-------------|
| `ttest_1sample()` | One-sample t-test |
| `ttest_2sample()` | Two-sample t-test (independent) |
| `ttest_paired()` | Paired t-test |
| `oneway_anova()` | One-way ANOVA with Tukey HSD |
| `correlation()` | Pearson, Spearman, Kendall correlations |
| `correlation_matrix()` | Correlation matrix with p-values |

### Regression & Model Building

| Function | Description |
|----------|-------------|
| `linear_regression()` | OLS regression with full output |
| `prediction_interval()` | Prediction and confidence intervals for new data |
| `residual_diagnostics()` | Residual analysis and tests |
| `fit_y_by_x()` | JMP-style bivariate fit |
| `fit_model()` | Multiple regression with interactions |
| `hat_matrix()` | Leverage and influence diagnostics |

### Model Selection

| Function | Description |
|----------|-------------|
| `stepwise_regression()` | Basic stepwise regression |
| `stepwise_regression_enhanced()` | Stepwise with multiple criteria |
| `subset_regression()` | All possible subsets |
| `compare_stepwise_criteria()` | Compare all stopping rules |

### Train/Test Validation

| Function | Description |
|----------|-------------|
| `train_test_split()` | Split data into train/test |
| `make_validation_column()` | Create JMP-style validation column |
| `validate_model()` | Train and evaluate on test set |
| `compare_models()` | Compare multiple models |
| `compare_all_criteria()` | Run all criteria and compare |

### Feature Engineering

| Function | Description |
|----------|-------------|
| `covariate_combinations()` | Generate interactions and polynomials |
| `polynomial_features()` | Sklearn-style polynomial features |
| `encode_categorical()` | Create dummy variables for regression |
| `encode_effect()` | Effect coding (-1, 0, 1) like JMP's default |

### Visualization

| Function | Description |
|----------|-------------|
| `plot_distribution()` | Histogram, boxplot, Q-Q plot |
| `plot_regression_diagnostics()` | 4-panel diagnostic plots |
| `plot_scatter_with_regression()` | Scatter with fit line and CI |
| `plot_correlation_matrix()` | Correlation heatmap |
| `plot_control_chart()` | Statistical process control chart |
| `plot_leverage_interactive()` | Leverage plots by predictor |
| `plot_influence_dashboard()` | Comprehensive influence dashboard |
| `plot_train_test_comparison()` | Train vs test performance plots |
| `plot_model_comparison()` | Model comparison visualization |

### Prediction Profiler

| Function | Description |
|----------|-------------|
| `prediction_profiler()` | Create interactive profiler |
| `plot_prediction_profiler()` | Visualize prediction profiles |

### Design of Experiments

| Function | Description |
|----------|-------------|
| `full_factorial_design()` | Full 2^k factorial design |
| `fractional_factorial_design()` | Fractional factorial designs |
| `response_surface_design()` | CCD, Box-Behnken designs |
| `optimal_design()` | D-optimal, I-optimal designs |
| `analyze_factorial()` | Analyze factorial experiments |
| `effect_screening()` | Lenth's method for screening |
| `main_effects_plot()` | Main effects visualization |
| `interaction_plot()` | Two-factor interaction plot |
| `contour_profiler()` | Response surface contours |
| `pareto_of_effects()` | Pareto chart of effects |

### Time Series

| Function | Description |
|----------|-------------|
| `arima()` | ARIMA/SARIMA modeling |
| `exponential_smoothing()` | Simple, Holt, Holt-Winters |
| `autocorrelation_analysis()` | ACF/PACF with order suggestions |
| `seasonal_decomposition()` | Trend/seasonal decomposition |
| `time_series_forecast()` | Unified forecasting interface |
| `adf_test()` | Augmented Dickey-Fuller test |
| `kpss_test()` | KPSS stationarity test |
| `ljung_box_test()` | Ljung-Box autocorrelation test |
| `plot_acf_pacf()` | ACF/PACF plots |
| `plot_time_series_diagnostics()` | TS diagnostic plots |

---

## Examples

### Complete Predictive Modeling Workflow

```python
import jmp_stats as jmp
import pandas as pd

# 1. Load and explore data
df = jmp.read_csv('sales_data.csv')

# 2. Check distributions
jmp.distribution_analysis(df['sales'], plot=True)

# 3. Check correlations
corr, pvals = jmp.correlation_matrix(df[['sales', 'price', 'advertising', 'competitors']])
jmp.plot_correlation_matrix(df)

# 4. Split data
predictors = ['price', 'advertising', 'competitors', 'store_size']
split = jmp.train_test_split(df['sales'], df[predictors], test_size=0.3, random_state=42)

# 5. Run stepwise with different criteria
step_bic = jmp.stepwise_regression_enhanced(
    split.y_train, split.X_train,
    direction='both', criterion='bic'
)
print(f"BIC selected: {step_bic.selected_features}")

# 6. Compare models
comparison = jmp.compare_all_criteria(df['sales'], df[predictors], test_size=0.3)
print(comparison)
jmp.plot_model_comparison(comparison)

# 7. Check influence diagnostics on best model
best_features = step_bic.selected_features
jmp.plot_influence_dashboard(df['sales'], df[best_features])

# 8. Create prediction profiler
profiler = jmp.prediction_profiler(df['sales'], df[best_features], y_name='Sales')
jmp.plot_prediction_profiler(profiler)

# 9. Final model with prediction intervals
final = jmp.linear_regression(df['sales'], df[best_features])
print(final)

# 10. Forecast new observations
X_new = pd.DataFrame({'price': [25], 'advertising': [5000]})
pi = jmp.prediction_interval(df['sales'], df[best_features], X_new)
print(pi.to_dataframe())
```

### DOE Analysis Workflow

```python
import jmp_stats as jmp

# 1. Create design
factors = {
    'Temperature': (150, 200),
    'Pressure': (10, 30),
    'Catalyst': (1, 5)
}
design = jmp.fractional_factorial_design(factors, center_points=3)
print(design)

# 2. [Run experiments and add 'Yield' column to design]
# design['Yield'] = [your experimental results]

# 3. Analyze
results = jmp.analyze_factorial(design, 'Yield', list(factors.keys()))
print(results)

# 4. Screen effects
screening = jmp.effect_screening(design, 'Yield', list(factors.keys()))
print(f"Significant: {screening.significant}")

# 5. Visualize
jmp.main_effects_plot(design, 'Yield', list(factors.keys()))
jmp.interaction_plot(design, 'Yield', 'Temperature', 'Pressure')
```

---

## Tips & Best Practices

### Model Selection
1. **Always use train/test validation** - Don't just rely on training metrics
2. **Full Model is excluded from "best" selection** - This prevents overfitting
3. **Compare multiple criteria** - BIC, AIC, and cross-validation often give different answers
4. **Check for overfitting** - Look for large gaps between train and test R-squared

### Regression Diagnostics
1. **Check VIF values** - VIF > 10 suggests multicollinearity
2. **Review influence diagnostics** - Remove or investigate high-leverage points
3. **Validate assumptions** - Check residual plots for patterns

### Prediction Intervals
1. **Use PI for individual predictions** - CI is only for the mean response
2. **PI is always wider than CI** - It includes irreducible error
3. **Check calibration first** - Intervals assume model assumptions are met

### Time Series
1. **Test for stationarity first** - Use `adf_test()` before fitting ARIMA
2. **Use ACF/PACF** - Let `autocorrelation_analysis()` suggest orders
3. **Compare methods** - Try both ARIMA and exponential smoothing

### DOE
1. **Use center points** - They help detect curvature
2. **Randomize run order** - The `RunOrder` column is provided for this
3. **Screen first** - Use fractional factorials to identify important factors

---

## Future Development

Priority additions for upcoming versions:

1. **Categorical Variable Handling in Regression**
   - Formula-style syntax support (e.g., `'y ~ x1 + C(category)'`)
   - Automatic dummy/effect coding within `linear_regression()`
   - Reference level selection for categorical predictors

2. **Mixed Effects Models**
   - `random_effects()`: Support for random intercepts and slopes
   - `mixed_model()`: Linear mixed effects regression (like JMP's Mixed Model)

3. **Logistic Regression**
   - `logistic_regression()`: Binary and multinomial logistic regression
   - ROC curves and AUC calculation
   - Classification metrics (confusion matrix, precision, recall)

4. **Enhanced Model Selection**
   - `cross_validation()`: K-fold CV with multiple metrics
   - `regularized_regression()`: Ridge, Lasso, Elastic Net
   - `model_averaging()`: Combine predictions from multiple models

5. **Additional Diagnostics**
   - `vif()`: Variance Inflation Factors for multicollinearity
   - `partial_regression_plots()`: Added variable plots
   - `component_residual_plots()`: Partial residual plots

---

## License

MIT License - Feel free to use and modify for academic and commercial purposes.

---

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.

---

## Version History

- **v2.2.0** - Added prediction intervals (`prediction_interval()`) for forecasting with CI and PI
- **v2.1.0** - Added categorical variable encoding (`encode_categorical()`, `encode_effect()`)
- **v2.0.0** - Added leverage plots, prediction profiler, DOE, time series analysis
- **v1.4.0** - Added train/test validation, model comparison
- **v1.3.0** - Added covariate combinations, enhanced stepwise
- **v1.2.0** - Added hat matrix, subset regression
- **v1.1.0** - Added data import functions
- **v1.0.0** - Initial release with core regression and diagnostics

---

## Author

Written by **Katrina Siegfried** with the help of [Claude Code](https://claude.ai).
