# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def svm(userData, X_train, y_train, X_test, y_test):
    # Load dataset (for example, the Iris dataset)
    svm_classifier = SVC(kernel='linear', random_state=42)
    # Train the classifiergit add .
    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)

    return svm_classifier