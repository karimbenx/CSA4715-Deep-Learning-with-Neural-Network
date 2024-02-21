import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_custom_dataset(dataset_path):
    data = pd.read_csv(dataset_path)
    X = data.drop(['id', 'diagnosis'], axis=1)
    y = data['diagnosis']
    return X, y

def create_custom_classes(row):
    if row['radius_mean'] < 10:
        return 'class1'
    elif 10 <= row['radius_mean'] < 20:
        return 'class2'
    else:
        return 'class3'

def main():
    dataset_path = "C:/Users/Merwin S/OneDrive/Documents/WPS Cloud Files/395703879/data(1).csv"
    X, y = load_custom_dataset(dataset_path)
    X['custom_class'] = X.apply(create_custom_classes, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    clf = RandomForestClassifier(random_state=23)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='g')
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()

    # Finding accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)

if __name__ == "__main__":
    main()
