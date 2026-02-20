# JMP-Stats: JMP-Style Statistical Analysis for Python

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version 2.7.0](https://img.shields.io/badge/version-2.7.0-green.svg)](https://github.com/siegfrkn/STAT-7220-JMP-Statistics-for-Python)

A comprehensive Python library that replicates JMP's statistical analysis capabilities for predictive analytics. Designed for students and practitioners who want to perform the same analyses in Python that they would do in JMP.

**Perfect for:** STAT 7220 - Predictive Analytics, STAT 7230 - Advanced Regression & Classification, STAT 7320, and similar courses.

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
  - [Logistic Regression](#logistic-regression)
  - [Bootstrap Methods](#bootstrap-methods)
  - [K Nearest Neighbors](#k-nearest-neighbors)
  - [STAT 7230: Advanced Regression & Classification](#stat-7230-advanced-regression--classification)
- [Function Reference](#function-reference)
- [Examples](#examples)
- [Tips & Best Practices](#tips--best-practices)
- [Future Development](#future-development)

---

## Installation

### Requirements

```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn scikit-learn patsy
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

# Polynomial regression (auto-generates x^2 terms)
results = jmp.linear_regression(df['Sales'], df[['Period']], poly_degree=2)

# Log-transformed regression
results = jmp.linear_regression(df['Revenue'], df[['Commission', 'Sales']], 
                                 log_y=True, log_X=['Commission', 'Sales'])

# Formula-based regression with categorical variables
results = jmp.linear_regression_formula(
    'Revenue ~ Price + C(Region) + C(Category)',
    data=df
)

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

#### Polynomial Regression

Automatically generate polynomial terms with `poly_degree`:

```python
# Quadratic model: y = b0 + b1*x + b2*x^2
results = jmp.linear_regression(df['Sales'], df[['Period']], poly_degree=2)
print(results.slopes)  # {'Period': 40.09, 'Period^2': 3.28}

# Cubic model
results = jmp.linear_regression(df['y'], df[['x']], poly_degree=3)
```

#### Log Transformations

Apply log transforms directly in the regression call:

```python
# Log-transform response only
results = jmp.linear_regression(df['Revenue'], df[['x1', 'x2']], log_y=True)

# Log-transform specific predictors
results = jmp.linear_regression(
    df['Revenue'], 
    df[['Commission', 'Selling', 'Age']],
    log_y=True,
    log_X=['Commission', 'Selling']  # Age stays untransformed
)
```

#### Formula-Based Regression with Categorical Variables

Use R-style formulas for automatic categorical encoding:

```python
# Categorical variables with C()
results = jmp.linear_regression_formula(
    'Sales ~ Price + C(Region) + Advertising',
    data=df
)

# Multiple categorical variables
results = jmp.linear_regression_formula(
    'Log_Revenue ~ Log_Commission + C(Territory) + C(Gender) + C(Marital_Status)',
    data=hiring
)

# Use Q() for variable names with spaces
results = jmp.linear_regression_formula(
    'Revenue ~ Q("Test Score") + C(Q("Marital Status"))',
    data=df
)

# Log transforms in formula
results = jmp.linear_regression_formula(
    'np.log(Revenue) ~ np.log(Commission) + C(Status)',
    data=df
)
```

#### Residual Diagnostics

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

### Logistic Regression

Perform binary logistic regression with JMP-style output (JMP's Fit Y by X / Fit Model with categorical Y).

```python
# Binary logistic regression
result = jmp.logistic_regression(
    df['purchased'],           # Binary response (0/1)
    df[['income', 'age']],     # Predictors
    alpha=0.05
)
print(result)

# Access results
print(f"AUC: {result.auc:.4f}")
print(f"Misclassification Rate: {result.misclassification_rate:.4f}")
print(result.confusion_matrix)

# ROC curve
jmp.plot_logistic_roc(result)

# Standalone ROC curve calculation
fpr, tpr, thresholds, auc = jmp.roc_curve(y_true, y_predicted_probs)

# Classification metrics
metrics = jmp.confusion_matrix_stats(y_true, y_predicted)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

**Output includes:**
- Coefficients, standard errors, z-values, p-values
- Odds ratios with confidence intervals
- Confusion matrix and misclassification rate
- AUC (Area Under ROC Curve)
- Log-likelihood, AIC, BIC

---

### Bootstrap Methods

Compute bootstrap confidence intervals for any statistic, including RMSE (JMP's Bootstrap command).

```python
# Bootstrap confidence interval for RMSE
boot = jmp.bootstrap_rmse(
    y=df['response'],
    X=df[['x1', 'x2']],
    n_bootstrap=1000,
    ci_level=0.90,
    random_state=42
)
print(f"Observed RMSE: {boot.observed:.4f}")
print(f"90% CI: [{boot.ci_lower:.4f}, {boot.ci_upper:.4f}]")

# Visualize bootstrap distribution
jmp.plot_bootstrap_distribution(boot)

# Generic bootstrap for any statistic
result = jmp.bootstrap(
    data=df['price'],
    statistic_func=np.mean,    # Any function
    n_bootstrap=1000,
    ci_level=0.95
)
print(f"Bootstrap mean: {result.observed:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

**Output includes:**
- Observed statistic value
- Bootstrap distribution (all resampled values)
- Confidence intervals (percentile, basic, and BCa methods)
- Standard error of the bootstrap distribution

---

### K Nearest Neighbors

Perform KNN classification or regression with automatic K selection (JMP's Analyze > Predictive Modeling > K Nearest Neighbors).

```python
# Auto-detect classification vs regression
result = jmp.k_nearest_neighbors(
    y=df['species'],                # Categorical → classification
    X=df[['sepal_length', 'petal_width']],
    k_values=[3, 5, 7, 9, 11],     # K values to try
    test_size=0.3,
    random_state=42
)
print(result)
print(f"Best K: {result.best_k}")
print(f"Accuracy: {result.accuracy:.4f}")

# Convenience wrappers
clf = jmp.knn_classification(df['species'], df[predictors], k=5)
reg = jmp.knn_regression(df['price'], df[predictors], k=7)

# Visualize K selection
jmp.plot_knn_results(result)

# Save predictions back to DataFrame
df_with_preds = jmp.save_knn_predictions(result, df)

# Lift curve for classification
jmp.knn_lift_curve(result)
```

**Output includes:**
- Best K value based on validation error
- Error rates for each K value tested
- Confusion matrix (classification) or RMSE (regression)
- Misclassification rate or R-squared

---

### STAT 7230: Advanced Regression & Classification

Utilities from STAT 7230 for regression diagnostics, classification comparison, and post-hoc analysis.

#### Confidence Interval for a Mean

```python
avg, se, ci_low, ci_high = jmp.ci_mean(df['response'], level=0.95)
# Prints: Sample Mean: 42.50
#         95.0% Confidence Interval: (40.12, 44.88)
```

#### Likelihood Ratio Test

Compare nested logistic regression models:

```python
import statsmodels.formula.api as smf

base = smf.logit('y ~ x1', data=df).fit()
full = smf.logit('y ~ x1 + x2 + x3', data=df).fit()

result = jmp.lr_test(base, full)
# Prints: LR = 12.345, df = 2, p = 0.00209
```

#### QQ Plot with Lilliefors Confidence Bounds

Normal quantile plot with Lilliefors simultaneous confidence bands (Conover 1980), matching JMP's QQ plot:

```python
fig, ax = plt.subplots()
jmp.qq_plot(residuals, ax, alpha=0.05)
```

#### RMSE from a Fitted Model

```python
import statsmodels.formula.api as smf

model = smf.ols('y ~ x1 + x2', data=df).fit()
rmse_val = jmp.rmse_from_model(model)
# Prints: RMSE = 4.23
```

#### Reference Line (abline)

```python
fig, ax = plt.subplots()
ax.scatter(x, y)
jmp.abline(ax, b=0, m=1, color='red', linestyle='--')  # y = x line
```

#### Tukey LS Means with Compact Letter Display

Compute regression-adjusted least-squares means and pairwise comparisons with a publication-ready figure:

```python
import statsmodels.formula.api as smf

model = smf.ols('Sales ~ C(Region) + Advertising + Price', data=df).fit()
cld_table = jmp.tukey_lsmeans(model, factor='Region', alpha=0.05)
print(cld_table)
#   Region   LSMean  CI_Lower  CI_Upper Group
#    West     125.3    118.2     132.4     A
#    East     119.7    112.8     126.6     A
#    South    105.1     98.3     111.9     B
```

#### Compare Classifiers with Cost Curves

Repeated train/test comparison of two logistic classifiers using custom cost matrices:

```python
result = jmp.compare_classifiers(
    df,
    formula_simple='Outcome ~ Age',
    formula_full='Outcome ~ Age + BloodPressure + BMI',
    cost_TN=0, cost_FP=100, cost_FN=500, cost_TP=50,
    n_splits=30, seed=42
)
print(f"Optimal threshold: {result['tau_star']:.3f}")
print(f"Mean AUC (simple): {result['auc_simple'].mean():.3f}")
print(f"Mean AUC (full):   {result['auc_full'].mean():.3f}")
```

**Output includes:**
- Cost curves (expected cost vs. threshold) with 95% bands
- ROC curves with AUC and cost-optimal operating point
- Per-split AUC and cost arrays for further analysis

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
| `linear_regression()` | OLS regression with poly_degree, log_y, log_X options |
| `linear_regression_formula()` | Formula-based regression with automatic categorical encoding |
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

### Logistic Regression

| Function | Description |
|----------|-------------|
| `logistic_regression()` | Binary logistic regression with odds ratios and AUC |
| `plot_logistic_diagnostics()` | Diagnostic plots for logistic regression |
| `plot_logistic_bivariate()` | Bivariate logistic fit (JMP's Fit Y by X) |
| `plot_logistic_roc()` | ROC curve visualization with AUC |
| `roc_curve()` | Standalone ROC curve calculation |
| `confusion_matrix_stats()` | Classification metrics (accuracy, sensitivity, specificity, F1) |

### Bootstrap Methods

| Function | Description |
|----------|-------------|
| `bootstrap()` | Generic bootstrap for any statistic |
| `bootstrap_rmse()` | Bootstrap confidence interval for RMSE |
| `plot_bootstrap_distribution()` | Visualize bootstrap distribution with CI |

### K Nearest Neighbors

| Function | Description |
|----------|-------------|
| `k_nearest_neighbors()` | KNN with auto-detect classification/regression |
| `knn_classification()` | Convenience wrapper for KNN classification |
| `knn_regression()` | Convenience wrapper for KNN regression |
| `plot_knn_results()` | JMP-style error rate plots |
| `save_knn_predictions()` | Save predictions back to DataFrame |
| `knn_lift_curve()` | Lift curves for classification |

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

### STAT 7230: Advanced Regression & Classification

| Function | Description |
|----------|-------------|
| `ci_mean()` | Confidence interval for a sample mean (normal approximation) |
| `lr_test()` | Likelihood ratio test for nested logistic models |
| `abline()` | Add a slope/intercept reference line to a matplotlib Axes |
| `qq_plot()` | QQ plot with Lilliefors simultaneous confidence bounds (matches JMP) |
| `rmse_from_model()` | Extract RMSE from a fitted statsmodels OLS model |
| `tukey_lsmeans()` | Regression-adjusted LS means with CLD and publication figure |
| `compare_classifiers()` | Repeated train/test cost-based comparison of two logistic classifiers |

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

### Regression with Transformations
1. **Use `poly_degree` for curved relationships** - Saves manual column creation
2. **Use `log_y=True` for skewed responses** - Common in revenue/sales data
3. **Use `log_X` for skewed predictors** - Specify which columns to transform
4. **Check distributions first** - Log transforms require positive values

### Categorical Variables
1. **Use `linear_regression_formula()` for categorical predictors** - Handles encoding automatically
2. **Use `C(varname)` syntax** - Explicitly marks categorical variables  
3. **Use `Q("Variable Name")` for spaces** - Handles special characters in column names
4. **First level is reference** - Coefficients are relative to the first category

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

1. **Mixed Effects Models**
   - `random_effects()`: Support for random intercepts and slopes
   - `mixed_model()`: Linear mixed effects regression (like JMP's Mixed Model)

2. **Enhanced Model Selection**
   - `cross_validation()`: K-fold CV with multiple metrics
   - `regularized_regression()`: Ridge, Lasso, Elastic Net
   - `model_averaging()`: Combine predictions from multiple models

3. **Additional Diagnostics**
   - `vif()`: Standalone Variance Inflation Factors function
   - `partial_regression_plots()`: Added variable plots
   - `component_residual_plots()`: Partial residual plots

4. **Multinomial Logistic Regression**
   - Extend `logistic_regression()` for multi-class outcomes
   - Generalized logit model with multiple response levels

---

## License

MIT License - Feel free to use and modify for academic and commercial purposes.

---

## Contributing

Contributions welcome! Please submit issues and pull requests on GitHub.

---

## Version History

- **v2.7.0** - Added STAT 7230 utilities: `ci_mean()`, `lr_test()`, `abline()`, `qq_plot()`, `rmse_from_model()`, `tukey_lsmeans()` (LS means with CLD), `compare_classifiers()` (cost-based logistic model comparison with ROC)
- **v2.5.1** - Fixed RMSE calculation in `linear_regression()` and `linear_regression_formula()` to use `sqrt(MSE)` = `sqrt(SSE/(n-p))` matching JMP's definition (was previously computing `sqrt(SSE/n)`)
- **v2.5.0** - Added K Nearest Neighbors (`k_nearest_neighbors()`, `knn_classification()`, `knn_regression()`), Bootstrap Methods (`bootstrap()`, `bootstrap_rmse()`), and Logistic Regression (`logistic_regression()`, `plot_logistic_roc()`, `roc_curve()`, `confusion_matrix_stats()`)
- **v2.3.0** - Enhanced `linear_regression()` with `poly_degree`, `log_y`, `log_X` parameters; added `linear_regression_formula()` for categorical variables
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
