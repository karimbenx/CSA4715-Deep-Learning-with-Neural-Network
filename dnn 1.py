import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generate example dataset
np.random.seed(0)
num_samples = 100
actual_labels = np.random.choice(["Dog", "Not dog"], size=num_samples)
predicted_labels = np.random.choice(["Dog", "Not dog"], size=num_samples)
print("Actual Labels: ",actual_labels)
print("Predicted Labels: ", predicted_labels)

# Calculate confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels, labels=["Dog", "Not dog"])


# Calculate evaluation metrics
accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels, pos_label="Dog")
recall = recall_score(actual_labels, predicted_labels, pos_label="Dog")
f1 = f1_score(actual_labels, predicted_labels, pos_label="Dog")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4)  # Adjust font size for better visualization
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=["Dog", "Not dog"], yticklabels=["Dog", "Not dog"])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
