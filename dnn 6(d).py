import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("C:\\Users\\Merwin S\\OneDrive\\Desktop\\Iris(1).csv")

# Separate features and target variable
X = dataset.drop(columns=['Species'])
y = dataset['Species']

# Feature Scaling (not necessary for Decision Trees, but can be done)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Decision Tree Classifier
classifier = DecisionTreeClassifier()

# Perform cross-validation
cv_scores = cross_val_score(classifier, X_scaled, y, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Average accuracy:", cv_scores.mean())
