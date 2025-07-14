import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Read the training and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print("Original data shape:")
print(f"Train: {train_data.shape}, Test: {test_data.shape}")

# Display basic info about the data
print("\nTrain data info:")
print(train_data.info())
print("\nMissing values in train data:")
print(train_data.isnull().sum())

# Data preprocessing function
def preprocess_data(data, is_train=True):
    # Drop irrelevant columns (id, CustomerId, Surname)
    data = data.drop(['id', 'CustomerId', 'Surname'], axis=1)
    
    # Handle missing values if any
    # For numerical columns, fill with median
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                     'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    for col in numerical_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    # Encode categorical variables
    # Geography: France, Germany, Spain
    geography_encoder = LabelEncoder()
    if is_train:
        data['Geography_encoded'] = geography_encoder.fit_transform(data['Geography'])
        # Save encoder for test data
        global geography_le
        geography_le = geography_encoder
    else:
        data['Geography_encoded'] = geography_le.transform(data['Geography'])
    
    # Gender: Male, Female
    gender_encoder = LabelEncoder()
    if is_train:
        data['Gender_encoded'] = gender_encoder.fit_transform(data['Gender'])
        # Save encoder for test data
        global gender_le
        gender_le = gender_encoder
    else:
        data['Gender_encoded'] = gender_le.transform(data['Gender'])
    
    # Drop original categorical columns
    data = data.drop(['Geography', 'Gender'], axis=1)
    
    return data

# Preprocess training data
train_processed = preprocess_data(train_data, is_train=True)
print(f"\nProcessed train data shape: {train_processed.shape}")

# Preprocess test data
test_processed = preprocess_data(test_data, is_train=False)
print(f"Processed test data shape: {test_processed.shape}")

# Separate features and target for training data
if 'Exited' in train_processed.columns:
    X_train = train_processed.drop('Exited', axis=1)
    y_train = train_processed['Exited']
else:
    # If no target column, assume all columns are features
    X_train = train_processed
    y_train = None

# For test data (assuming no target column)
X_test = test_processed

# Feature scaling (optional for XGBoost but can help)
print("\nApplying feature scaling (optional for XGBoost)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nFinal feature shapes:")
print(f"X_train: {X_train_scaled.shape}")
print(f"X_test: {X_test_scaled.shape}")
if y_train is not None:
    print(f"y_train: {y_train.shape}")
    print(f"Target distribution:\n{y_train.value_counts()}")

# Train the model using scaled features
if y_train is not None:
    
    # Method 1: Simple XGBoost with default parameters
    print("\n" + "="*60)
    print("Method 1: Simple XGBoost with default parameters")
    print("="*60)
    
    xgb_simple = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',  # Suppress warnings
        verbosity=0  # Reduce output
    )
    
    # Cross-validation evaluation
    cv_scores = cross_val_score(xgb_simple, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Method 2: XGBoost with basic tuning
    print("\n" + "="*60)
    print("Method 2: XGBoost with Basic Parameter Tuning")
    print("="*60)
    
    # Define parameter grid for basic tuning
    param_grid_basic = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    print("Starting Basic Grid Search...")
    
    xgb_basic = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    
    grid_search_basic = GridSearchCV(
        xgb_basic,
        param_grid_basic,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search_basic.fit(X_train_scaled, y_train)
    
    print(f"\nBasic tuning - Best parameters: {grid_search_basic.best_params_}")
    print(f"Basic tuning - Best CV score: {grid_search_basic.best_score_:.4f}")
    
    # Method 3: Advanced XGBoost tuning with RandomizedSearch
    print("\n" + "="*60)
    print("Method 3: XGBoost with Advanced Random Search")
    print("="*60)
    
    # Define parameter distribution for random search
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0.1, 0.5, 1, 2]
    }
    
    print("Starting Advanced Random Search (this may take several minutes)...")
    
    xgb_advanced = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    
    random_search = RandomizedSearchCV(
        xgb_advanced,
        param_dist,
        n_iter=50,  # Number of parameter combinations to try
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train_scaled, y_train)
    
    print(f"\nAdvanced tuning - Best parameters: {random_search.best_params_}")
    print(f"Advanced tuning - Best CV score: {random_search.best_score_:.4f}")
    
    # Use the best model from advanced tuning
    best_xgb = random_search.best_estimator_
    
    # Final cross-validation with best model
    final_cv_scores = cross_val_score(best_xgb, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Final CV accuracy with best params: {final_cv_scores.mean():.4f} (+/- {final_cv_scores.std() * 2:.4f})")
    
    # Train the final model
    best_xgb.fit(X_train_scaled, y_train)
    print("\nFinal XGBoost model training completed!")
    
    # Predict on test set
    y_pred = best_xgb.predict(X_test_scaled)
    y_pred_proba = best_xgb.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1
    
    print(f"\nPredictions generated for {len(y_pred)} test samples")
    print(f"Predicted class distribution:\n{pd.Series(y_pred).value_counts()}")
    print(f"Prediction probabilities range: {y_pred_proba.min():.3f} - {y_pred_proba.max():.3f}")
    
    # Save predictions to file
    test_predictions = pd.DataFrame({
        'id': test_data['id'],
        'Exited': y_pred,
        'Exited_Probability': y_pred_proba
    })
    test_predictions.to_csv('predictions_xgboost.csv', index=False)
    print("\nPredictions saved to 'predictions_xgboost.csv'")
    
    # Feature importance analysis
    print("\n" + "="*60)
    print("Feature Importance Analysis")
    print("="*60)
    
    # Get feature importance
    feature_names = X_train.columns
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Model information
    print(f"\nFinal XGBoost Model Details:")
    print(f"Number of estimators: {best_xgb.n_estimators}")
    print(f"Max depth: {best_xgb.max_depth}")
    print(f"Learning rate: {best_xgb.learning_rate}")
    print(f"Subsample: {best_xgb.subsample}")
    print(f"Colsample bytree: {best_xgb.colsample_bytree}")
    
    # Performance comparison
    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)
    
    # Train simple and basic models for comparison
    xgb_simple.fit(X_train_scaled, y_train)
    simple_cv = cross_val_score(xgb_simple, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    basic_xgb = grid_search_basic.best_estimator_
    basic_cv = cross_val_score(basic_xgb, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    print(f"Simple XGBoost (default params):     {simple_cv.mean():.4f} (+/- {simple_cv.std() * 2:.4f})")
    print(f"Basic tuned XGBoost:                 {basic_cv.mean():.4f} (+/- {basic_cv.std() * 2:.4f})")
    print(f"Advanced tuned XGBoost:              {final_cv_scores.mean():.4f} (+/- {final_cv_scores.std() * 2:.4f})")
    
    improvement_basic = basic_cv.mean() - simple_cv.mean()
    improvement_advanced = final_cv_scores.mean() - simple_cv.mean()
    
    print(f"\nImprovement from basic tuning:       {improvement_basic:.4f} ({improvement_basic*100:.2f}%)")
    print(f"Improvement from advanced tuning:    {improvement_advanced:.4f} ({improvement_advanced*100:.2f}%)")
    
    # Save feature importance
    feature_importance.to_csv('feature_importance_xgboost.csv', index=False)
    print("\nFeature importance saved to 'feature_importance_xgboost.csv'")
    
else:
    print("No target variable found in training data. Cannot train the model.")

print("\n" + "="*60)
print("XGBoost Training Complete!")
print("="*60)
print("Files generated:")
print("- predictions_xgboost.csv (includes probabilities)")
print("- feature_importance_xgboost.csv")
print("="*60)
