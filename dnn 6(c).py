import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
dataset = pd.read_csv("C:\\Users\\Merwin S\\OneDrive\\Desktop\\Iris(1).csv")

# Separate features and target variable
X = dataset.drop(columns=['Species'])
y = dataset['Species']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Logistic Regression Classifier
classifier = LogisticRegression()

# Perform cross-validation
# You can specify the number of folds with the cv parameter
# By default, StratifiedKFold is used, which preserves the class distribution in each fold
# You can also use other cross-validation strategies like KFold
# Scores are typically accuracy values, but other scoring metrics can be used
cv_scores = cross_val_score(classifier, X_scaled, y, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Average accuracy:", cv_scores.mean())
