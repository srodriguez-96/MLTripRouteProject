from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def svm_regression(X_train, y_train, X_test, y_test):
    # Initialize and train the SVR model
    svr_model = SVR(kernel='linear')  # You can specify different kernels like 'linear', 'rbf', 'poly', etc.
    svr_model.fit(X_train, y_train)

    # Predicting and calculating MSE
    y_pred = svr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    i = 0
    for pred, actual in zip(y_pred, y_test):
        print(f"{pred:.2f}\t\t{actual:.2f}")
        i += 1
        if i == 10:
            break
    return svr_model