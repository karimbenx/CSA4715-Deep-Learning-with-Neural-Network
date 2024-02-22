import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Importing the dataset
dataset = pd.read_csv("C:\\Users\\Merwin S\\OneDrive\\Desktop\\Iris(1).csv")

# Separate features and target variable
X = dataset.drop(columns=['Species'])
y = dataset['Species']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the SVM model
classifier = SVC(kernel='linear')  # You can adjust the kernel type as needed

# Perform cross-validated predictions
y_pred = cross_val_predict(classifier, X_scaled, y, cv=5)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y, y_pred)

# Print cross-validated accuracy
print("Cross-validated accuracy:", accuracy)

# Print confusion matrix
print("Confusion matrix:")
print(conf_matrix)
