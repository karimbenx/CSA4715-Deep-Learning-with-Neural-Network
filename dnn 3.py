import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv("C:/Users/Merwin S/OneDrive/Documents/WPS Cloud Files/395703879/data(1).csv")

# Preprocess the data
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
model = SVC()

# Train the model
model.fit(X_train, y_train)

# Calculate training and testing accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Calculate mean training and testing accuracy
mean_train_accuracy = cv_scores.mean()
mean_test_accuracy = model.score(X_test, y_test)

# Check for overfitting
if mean_train_accuracy > mean_test_accuracy:
    print("Model may be overfitting: Mean training accuracy ({}) is higher than mean testing accuracy ({}).".format(mean_train_accuracy, mean_test_accuracy))
else:
    print("No evidence of overfitting: Mean training accuracy ({}) is not higher than mean testing accuracy ({}).".format(mean_train_accuracy, mean_test_accuracy))
