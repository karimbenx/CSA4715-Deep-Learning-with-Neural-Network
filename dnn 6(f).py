import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Importing the dataset
dataset = pd.read_csv("C:\\Users\\Merwin S\\OneDrive\\Desktop\\Iris(1).csv")

# Separate features and target variable
X = dataset.drop(columns=['Species'])
y = dataset['Species']

# Feature Scaling (not necessary for Random Forests, but can be done)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust parameters as needed

# Perform cross-validation
cv_scores = cross_val_score(classifier, X_scaled, y, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)

# Calculate average accuracy
avg_accuracy = cv_scores.mean()
print("Average accuracy:", avg_accuracy)

# Perform cross-validated predictions
y_pred = cross_val_predict(classifier, X_scaled, y, cv=5)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y, y_pred)

# Print confusion matrix
print("Confusion matrix:")
print(conf_matrix)
