import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read the training and test data, skip first row and drop first three columns
train_data = pd.read_csv('train.csv', skiprows=1)
test_data = pd.read_csv('test.csv', skiprows=1)

# Drop the first three columns as they are not informative
train_data = train_data.iloc[:, 3:]
test_data = test_data.iloc[:, 3:]

# Assuming the last column is the target variable
X_train = train_data.iloc[:, :-1]  # all columns except the last one
y_train = train_data.iloc[:, -1]   # last column
X_test = test_data.iloc[:, :-1]    # all columns except the last one
y_test = test_data.iloc[:, -1]     # last column

# Create AdaBoost classifier
clf = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")