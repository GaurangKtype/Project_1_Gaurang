import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score, precision_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import joblib


# Load the dataset
csv_file = 'Project 1 Data.csv'
df = pd.read_csv(csv_file)

# Handling missing values
if df.isna().any(axis=0).sum() > 0 or df.isna().any(axis=1).sum() > 0:
    df = df.dropna()
    df = df.reset_index(drop=True)

# Define features and target
featureX = df[['X', 'Y', 'Z']]
y = df['Step']

# Splitting the data using StratifiedShuffleSplit
cut = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in cut.split(featureX, y):
    X_train, X_test = featureX.iloc[train_index], featureX.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
X_train = X_train.reset_index(drop=True)    
X_test = X_test.reset_index(drop=True)   
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Visualizations
sns.set(style="ticks")
sns.pairplot(df, diag_kind="hist", markers="o")
plt.show()

sns.pairplot(df, kind='scatter', hue='Step')
plt.show()

plt.figure(1)
plt.hist(y_train, bins=13, edgecolor='black')
plt.xlabel('Step')
plt.ylabel('Frequency')
plt.title('Histogram of Target Variable (Step)')
plt.grid(True)
plt.show()

corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
# Training models

#################################  MODEL 1 ####################################
# Training an SVM model for multiclass classification
# Initializing a parameter grid list 
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}

svm_model = SVC() 

# Perform Grid Search with Cross-Validation
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, scoring='accuracy', verbose=0)
grid_search_svm.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search_svm.best_params_

best_svm_model = SVC(**best_params)
best_svm_model.fit(X_train, y_train)

svm_model_class_predictions = best_svm_model.predict(X_train)

# Calculate accuracy
svm_train_accuracy = accuracy_score(y_train, svm_model_class_predictions)
print("Accuracy for the SVM model on the train dataset:", svm_train_accuracy)

# Calculate precision
svm_train_precision = precision_score(y_train, svm_model_class_predictions, average='weighted')
print("Precision for the SVM model on the train dataset::", svm_train_precision)

# Calculate F1 score
svm_train_f1 = f1_score(y_train, svm_model_class_predictions, average='weighted')
print("F1 Score for the SVM model on the train dataset:", svm_train_f1)

print("Best SVM parameters for this dataset:", best_params)


# ############################ MODEL 2 ##########################################

# Scale the features using Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid for Logistic Regression
param_grid_logistic = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

logistic_model = LogisticRegression(max_iter=1000)

# Perform Grid Search with Cross-Validation
grid_search_logistic = GridSearchCV(estimator=logistic_model, param_grid=param_grid_logistic, cv=5, scoring='accuracy', verbose=0)
grid_search_logistic.fit(X_train, y_train)

# Get the best parameters
best_params_logistic = grid_search_logistic.best_params_

best_logistic_model = LogisticRegression(**best_params_logistic, max_iter=1000)  # You may adjust max_iter as needed
best_logistic_model.fit(X_train, y_train)

logistic_model_class_predictions = best_logistic_model.predict(X_train)

# Calculate accuracy
logistic_test_accuracy = accuracy_score(y_train, logistic_model_class_predictions)
print("Accuracy for the Logistic Regression model on the train dataset:", logistic_test_accuracy)

# Calculate precision
logistic_test_precision = precision_score(y_train, logistic_model_class_predictions, average='weighted')
print("Precision for the Logistic Regression model on the train dataset:", logistic_test_precision)

# Calculate F1 score
logistic_test_f1 = f1_score(y_train, logistic_model_class_predictions, average='weighted')
print("F1 Score for the Logistic Regression model on the train dataset:", logistic_test_f1)

print("Best Logistic Regression parameters for this dataset:", best_params_logistic)


# ############################ MODEL 3 ##########################################



# Define the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [10, 20, 30],
    'max_depth': [None, 10, 20]
}

rf_model = RandomForestClassifier(random_state=42)

# Perform Grid Search with Cross-Validation
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='accuracy', verbose=0)
grid_search_rf.fit(X_train, y_train)

# Get the best parameters
best_params_rf = grid_search_rf.best_params_

best_rf_model = RandomForestClassifier(**best_params_rf, random_state=42)
best_rf_model.fit(X_train, y_train)

rf_model_class_predictions = best_rf_model.predict(X_train)

# Calculate accuracy
rf_train_accuracy = accuracy_score(y_train, rf_model_class_predictions)
print("Accuracy for the Random Forest model on the train dataset:", rf_train_accuracy)

# Calculate precision
rf_train_precision = precision_score(y_train, rf_model_class_predictions, average='weighted')
print("Precision for the Random Forest model on the train dataset:", rf_train_precision)

# Calculate F1 score
rf_train_f1 = f1_score(y_train, rf_model_class_predictions, average='weighted')
print("F1 Score for the Random Forest model on the train dataset:", rf_train_f1)

print("Best Random Forest parameters for this dataset:", best_params_rf)

###############################################################################

# ######################## MODEL SELECTION AND EVALUATION  ####################

# Evaluate the Random Forest model on the test dataset
rf_test_predictions = best_rf_model.predict(X_test)

# Create a confusion matrix
confusion = confusion_matrix(y_test, rf_test_predictions)

# Calculate accuracy
rf_test_accuracy = accuracy_score(y_test, rf_test_predictions)

# Calculate precision
rf_test_precision = precision_score(y_test, rf_test_predictions, average='weighted')

# Calculate F1 score
rf_test_f1 = f1_score(y_test, rf_test_predictions, average='weighted')

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Show the plot
plt.show()

# Display accuracy, precision, and F1 score
print("Accuracy for the Random Forest model on the test dataset:", rf_test_accuracy)
print("Precision for the Random Forest model on the test dataset:", rf_test_precision)
print("F1 Score for the Random Forest model on the test dataset:", rf_test_f1)

################################## JOBLIB ####################################

# Save the trained Random Forest model to a file
joblib.dump(best_rf_model, 'random_forest_model.joblib')

# Load the Random Forest model from the saved file
loaded_rf_model = joblib.load('random_forest_model.joblib')

# Input coordinates for prediction
input_coordinates = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])

# Standardize the input coordinates using the same scaler used for training
scaler2 = StandardScaler()
input_coordinates = scaler2.fit_transform(input_coordinates)

# Predict the maintenance step for the input coordinates
predicted_steps = best_rf_model.predict(input_coordinates)

# Output the predicted steps
for i, step in enumerate(predicted_steps):
    print(f"Predicted Step {i + 1}: {step}")

