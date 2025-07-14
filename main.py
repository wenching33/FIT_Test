import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
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

# Feature scaling (AdaBoost doesn't require it but can help)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nFinal feature shapes:")
print(f"X_train: {X_train_scaled.shape}")
print(f"X_test: {X_test_scaled.shape}")
if y_train is not None:
    print(f"y_train: {y_train.shape}")
    print(f"Target distribution:\n{y_train.value_counts()}")

# Create AdaBoost classifier
clf = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train the model using scaled features
if y_train is not None:
    clf.fit(X_train_scaled, y_train)
    print("\nModel training completed!")
    
    # For model evaluation, we can split training data or use cross-validation
    # Since test data doesn't have labels, let's split training data for evaluation
    
    # Cross-validation on training data
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Predict on test set (for submission)
    y_pred = clf.predict(X_test_scaled)
    print(f"\nPredictions generated for {len(y_pred)} test samples")
    print(f"Predicted class distribution:\n{pd.Series(y_pred).value_counts()}")
    
    # Save predictions to file
    test_predictions = pd.DataFrame({
        'id': test_data['id'],
        'Exited': y_pred
    })
    test_predictions.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to 'predictions.csv'")
    
    # Feature importance
    feature_names = X_train.columns
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
else:
    print("No target variable found in training data. Cannot train the model.")