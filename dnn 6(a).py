import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv("C:\\Users\\Merwin S\\OneDrive\\Desktop\\Iris(1).csv")

# Separate features and target variable
X = dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = dataset['Species']

# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store accuracy scores
accuracy_scores = []

# Iterate over each fold
for train_index, test_index in kf.split(X):
    # Split data into train and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the K-Nearest Neighbors (K-NN) Classification model
    classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed
    classifier.fit(X_train_scaled, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test_scaled)

    # Calculating the accuracy of the model for this fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate the average accuracy across all folds
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print("Average Accuracy (across all folds):", average_accuracy)
