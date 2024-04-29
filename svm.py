# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def svm(userData, xTrain, yTrain):
    # Load dataset (for example, the Iris dataset)
    svm_classifier = SVC(kernel='linear', random_state=42)
    # Train the classifiergit add .
    svm_classifier.fit(xTrain, yTrain)
    return svm_classifier