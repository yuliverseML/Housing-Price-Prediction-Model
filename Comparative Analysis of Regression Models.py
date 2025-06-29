# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set the style for our plots
plt.style.use('seaborn-v0_8-whitegrid')

###1. Data Loading and Exploration
# Load the Boston Housing dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Get summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Understanding the features
print("\nDataset columns:")
for col in df.columns:
    print(f"- {col}")
  
###2. Data Cleaning and Anomaly Detection
# Function to detect outliers using the IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Identify the target variable
target_column = 'medv'  # Median value of owner-occupied homes in $1000s

# Let's check for outliers in the target variable
outliers, lb, ub = detect_outliers_iqr(df, target_column)
print(f"Outliers in {target_column}: {len(outliers)} rows")
print(f"Lower bound: {lb}, Upper bound: {ub}")

# Visualize the outliers in the target variable
plt.figure(figsize=(10, 6))
plt.boxplot(df[target_column])
plt.title('Boxplot of Housing Prices')
plt.ylabel('Median Value ($1000s)')
plt.grid(True, alpha=0.3)
plt.show()

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df[target_column], kde=True)
plt.title('Distribution of Housing Prices')
plt.xlabel('Median Value ($1000s)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Check for anomalies in all numerical features
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Create a function to plot boxplots for all numerical features
def plot_boxplots(df, columns, ncols=3):
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Plot boxplots for all numerical features
plot_boxplots(df, numerical_cols)

# Create a function to handle outliers
def handle_outliers(df, column, method='cap'):
    """
    Handle outliers in a column using different methods:
    - 'cap': Cap the outliers at the bounds
    - 'remove': Remove the outlier rows
    - 'log': Apply log transformation
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    if method == 'cap':
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    elif method == 'remove':
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'log':
        # Add a small constant to avoid log(0)
        min_val = df[column].min()
        if min_val <= 0:
            df[column] = np.log(df[column] - min_val + 1)
        else:
            df[column] = np.log(df[column])
    
    return df

# Cap outliers in the target variable
df = handle_outliers(df, target_column, method='cap')

# Visualize the target variable after handling outliers
plt.figure(figsize=(10, 6))
sns.histplot(df[target_column], kde=True)
plt.title('Distribution of Housing Prices (After Handling Outliers)')
plt.xlabel('Median Value ($1000s)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

###3. Feature Analysis and Engineering
# Correlation analysis
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.show()

# Top correlated features with the target
top_corr_features = correlation_matrix[target_column].sort_values(ascending=False)
print("\nTop correlated features with the target variable:")
print(top_corr_features)

# Pairplot of top correlated features
top_features = top_corr_features.index[:5]  # Top 5 features
plt.figure(figsize=(15, 10))
sns.pairplot(df[top_features], diag_kind='kde')
plt.suptitle('Pairplot of Top Correlated Features', y=1.02)
plt.show()

# Feature engineering: Create new features
# 1. Ratio of rooms per dwelling (rm) to number of rooms (ptratio)
df['rooms_per_ptratio'] = df['rm'] / df['ptratio']

# 2. Accessibility score (combining distance and access features)
df['accessibility_score'] = df['dis'] / (1 + df['rad'])

# 3. Socioeconomic status
df['socio_status'] = df['lstat'] * df['tax']

# Check the correlation of new features with the target
new_correlation = df[['rooms_per_ptratio', 'accessibility_score', 'socio_status', target_column]].corr()[target_column]
print("\nCorrelation of new features with the target variable:")
print(new_correlation)

# Update the feature list after engineering
features = df.drop(columns=[target_column]).columns


###4. White Noise and Stationarity Testing
# Function to test for white noise using Ljung-Box test
def ljung_box_test(residuals, lags=None):
    if lags is None:
        lags = min(10, len(residuals) // 5)
    
    result = sm.stats.acorr_ljungbox(residuals, lags=lags)
    return result

# Creating a simple linear model to test for white noise in residuals
X = df.drop(columns=[target_column])
y = df[target_column]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit a simple linear model
model = LinearRegression()
model.fit(X_scaled, y)
predictions = model.predict(X_scaled)
residuals = y - predictions

# Test for white noise in residuals
lb_result = ljung_box_test(residuals)
print("\nLjung-Box Test for White Noise in Residuals:")
print(lb_result)

# Plot the residuals
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot ACF and PACF of residuals
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sm.graphics.tsa.plot_acf(residuals, lags=20, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sm.graphics.tsa.plot_pacf(residuals, lags=20, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# White's test for heteroskedasticity
X_with_const = sm.add_constant(X_scaled)
white_test = het_white(residuals, X_with_const)
print("\nWhite's Test for Heteroskedasticity:")
print(f"LM Statistic: {white_test[0]}")
print(f"P-value: {white_test[1]}")
print(f"F-Statistic: {white_test[2]}")
print(f"F-Test p-value: {white_test[3]}")

###5. Data Transformation and Decomposition
# Check for skewness in the target variable
skewness = stats.skew(df[target_column])
print(f"\nSkewness of {target_column}: {skewness:.4f}")

# Apply Box-Cox transformation to the target if it's skewed
if abs(skewness) > 0.5:
    pt = PowerTransformer(method='box-cox')
    df[f'{target_column}_transformed'] = pt.fit_transform(df[[target_column]])
    
    # Plot the original vs transformed target
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df[target_column], kde=True)
    plt.title(f'Original {target_column} Distribution')
    plt.xlabel(target_column)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sns.histplot(df[f'{target_column}_transformed'], kde=True)
    plt.title(f'Transformed {target_column} Distribution')
    plt.xlabel(f'{target_column} (Box-Cox)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Update the target
    y_transformed = df[f'{target_column}_transformed'].values
    use_transformed = True
else:
    y_transformed = y
    use_transformed = False

# Check skewness in features and apply transformations if needed
skewed_features = []
for feature in features:
    skewness = stats.skew(df[feature])
    if abs(skewness) > 0.5:
        skewed_features.append(feature)

print(f"\nNumber of skewed features: {len(skewed_features)}")
print("Skewed features:", skewed_features)

# Apply transformation to skewed features
for feature in skewed_features:
    # Handle features with zeros or negative values
    min_val = df[feature].min()
    if min_val <= 0:
        # Use Yeo-Johnson transformation (works with negative values)
        pt = PowerTransformer(method='yeo-johnson')
        df[f'{feature}_transformed'] = pt.fit_transform(df[[feature]])
    else:
        # Use Box-Cox transformation
        pt = PowerTransformer(method='box-cox')
        df[f'{feature}_transformed'] = pt.fit_transform(df[[feature]])

# Visualize a few transformed features
if skewed_features:
    features_to_show = skewed_features[:min(3, len(skewed_features))]
    
    for feature in features_to_show:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Original {feature} Distribution')
        plt.xlabel(feature)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        sns.histplot(df[f'{feature}_transformed'], kde=True)
        plt.title(f'Transformed {feature} Distribution')
        plt.xlabel(f'{feature} (Transformed)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Time series decomposition (assuming the data has a temporal component)
# For this dataset, we would typically need a time/date column, but for demonstration:
# We'll create a synthetic time series using the target variable
def create_synthetic_time_series(data, feature):
    # Sort data by the feature to create a pseudo time series
    data_sorted = data.sort_values(by=feature).reset_index(drop=True)
    return data_sorted

# Create a synthetic time series based on 'lstat' (% lower status of the population)
ts_data = create_synthetic_time_series(df, 'lstat')

# Apply seasonal decomposition
# Note: This is for demonstration only; real estate data might not have clear seasonality
try:
    # Create a pandas Series with a DatetimeIndex
    ts = pd.Series(ts_data[target_column].values)
    
    # Decompose the time series (with a period that makes sense for the data)
    decomposition = seasonal_decompose(ts, model='additive', period=12)
    
    # Plot the decomposition
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(ts)
    plt.title('Original Time Series (Synthetic)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend)
    plt.title('Trend Component')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonal Component')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid)
    plt.title('Residual Component')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Could not perform time series decomposition: {e}")

###6. Prepare Data for Modeling
# Prepare the data for modeling
# Use transformed features where available
X_prepared = df.drop(columns=[target_column] + ([f'{target_column}_transformed'] if use_transformed else []))

# Replace skewed features with their transformed versions
for feature in skewed_features:
    if f'{feature}_transformed' in X_prepared.columns:
        X_prepared = X_prepared.drop(columns=[feature])

# Use the transformed target if available
y_prepared = y_transformed if use_transformed else y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_prepared, y_prepared, test_size=0.2, random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

###7. Model Training and Evaluation
# Define the models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet': ElasticNet(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR()
}

# Function to evaluate a model using cross-validation
def evaluate_model(model, X, y, cv=5):
    # Cross-validation scores
    mse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(mse_scores)
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    return {
        'RMSE': rmse_scores.mean(),
        'R²': r2_scores.mean()
    }

# Evaluate all models
results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    results[name] = evaluate_model(model, X_train, y_train)
    print(f"Cross-validation results: RMSE = {results[name]['RMSE']:.4f}, R² = {results[name]['R²']:.4f}")

# Find the best model based on RMSE
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
print(f"\nBest model based on RMSE: {best_model_name}")
print(f"RMSE: {results[best_model_name]['RMSE']:.4f}, R²: {results[best_model_name]['R²']:.4f}")

# Visualize model comparison
plt.figure(figsize=(14, 6))

# RMSE comparison
plt.subplot(1, 2, 1)
model_names = list(results.keys())
rmse_values = [results[model]['RMSE'] for model in model_names]
bars = plt.bar(model_names, rmse_values, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title('Model Comparison - RMSE (Lower is Better)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.grid(True, alpha=0.3, axis='y')

# Highlight the best model
best_index = model_names.index(best_model_name)
bars[best_index].set_color('gold')

# R² comparison
plt.subplot(1, 2, 2)
r2_values = [results[model]['R²'] for model in model_names]
bars = plt.bar(model_names, r2_values, color='lightgreen')
plt.xticks(rotation=45, ha='right')
plt.title('Model Comparison - R² (Higher is Better)')
plt.ylabel('R-squared (R²)')
plt.grid(True, alpha=0.3, axis='y')

# Highlight the best model based on R²
best_r2_model = max(results, key=lambda x: results[x]['R²'])
best_r2_index = model_names.index(best_r2_model)
bars[best_r2_index].set_color('gold')

plt.tight_layout()
plt.show()

###8. Fine-tuning the Best Model
# Get the best model
best_model = models[best_model_name]

# Define hyperparameter grids for different models
param_grids = {
    'Linear Regression': {},  # Linear Regression doesn't have hyperparameters to tune
    'Ridge Regression': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
    'Lasso Regression': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
    'ElasticNet': {
        'alpha': [0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    },
    'SVR': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }
}

# Perform grid search for the best model
if best_model_name in param_grids and param_grids[best_model_name]:
    print(f"\nFine-tuning {best_model_name} with Grid Search...")
    
    grid_search = GridSearchCV(
        best_model,
        param_grids[best_model_name],
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    # Update the best model with the tuned one
    best_model = grid_search.best_estimator_
else:
    print(f"\n{best_model_name} does not require hyperparameter tuning or no grid defined.")
    # Train the model on the full training set
    best_model.fit(X_train, y_train)

# Evaluate the final model on the test set
y_pred = best_model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nFinal Model Evaluation on Test Set:")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# Visualize the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Housing Prices')
plt.grid(True, alpha=0.3)
plt.show()

# Visualize residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)
plt.show()

###9. Feature Importance Analysis
# Analyze feature importance (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # Get feature importance from the model
    importances = best_model.feature_importances_
    feature_names = X_train.columns
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Visualize feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
else:
    # For non-tree models, use permutation importance
    perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
    
    # Sort feature importances in descending order
    feature_names = X_test.columns
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    
    # Visualize permutation importances
    plt.figure(figsize=(12, 8))
    plt.title('Permutation Feature Importance')
    plt.bar(range(X_test.shape[1]), perm_importance.importances_mean[sorted_idx], align='center')
    plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in sorted_idx], rotation=90)
    plt.tight_layout()
    plt.show()

###10. Final Model and Conclusion
# Inverse transform the predictions if we used a transformed target
if use_transformed:
    # Create a placeholder for inverse transformation
    y_pred_original_scale = np.zeros_like(y_pred)
    
    # Reshape for inverse transform
    y_pred_reshaped = y_pred.reshape(-1, 1)
    
    # Inverse transform
    y_pred_original_scale = pt.inverse_transform(y_pred_reshaped).flatten()
    
    # Get the original scale test data
    y_test_original = df.loc[y_test.index, target_column].values
    
    # Calculate metrics on the original scale
    rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original_scale))
    r2_original = r2_score(y_test_original, y_pred_original_scale)
    mae_original = mean_absolute_error(y_test_original, y_pred_original_scale)
    
    print("\nFinal Model Evaluation on Original Scale:")
    print(f"RMSE: {rmse_original:.4f}")
    print(f"R²: {r2_original:.4f}")
    print(f"MAE: {mae_original:.4f}")
    
    # Visualize the predictions vs actual values on the original scale
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_pred_original_scale, alpha=0.7)
    plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
    plt.xlabel('Actual Values (Original Scale)')
    plt.ylabel('Predicted Values (Original Scale)')
    plt.title('Actual vs Predicted Housing Prices (Original Scale)')
    plt.grid(True, alpha=0.3)
    plt.show()

# Save the best model (in practice, you would save to disk)
best_model_final = best_model

print("\nBest Model Summary:")
print(f"Model Type: {best_model_name}")
print(f"Performance on Test Set: RMSE = {rmse:.4f}, R² = {r2:.4f}")
print("\nThe model is ready for deployment.")
