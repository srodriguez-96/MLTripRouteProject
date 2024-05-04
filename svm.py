from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def svm_regression(X_train, y_train, X_test, y_test, data, X):
    # Initialize and train the SVR model
    svr_model = SVR(kernel='linear')  # You can specify different kernels like 'linear', 'rbf', 'poly', etc.
    svr_model.fit(X_train, y_train)

    # Predicting and calculating MSE
    y_pred = svr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error SVR: {mse}')

    #Predicting the ratings for all hotels
    data['predicted_stars'] = svr_model.predict(X)

    #Sort data by predicted ratings in descending order
    ranked_hotels = data.sort_values(by='predicted_stars', ascending=False)

    #Print the ranked hotels
    #print(ranked_hotels[['name', 'predicted_stars']])

    #Combine predictions with additional hotel details
    detailed_predictions = pd.DataFrame({
        'Hotel Name': data.loc[y_test.index, 'name'],
        'Address': data.loc[y_test.index, 'address'],
        'Actual Stars': y_test,
        'Predicted Stars': np.round(y_pred, 2)  #Rounding the predictions
    }).sort_values(by='Predicted Stars', ascending=False)  #Sort by predicted stars for ranking

    #Saving the data to a CSV file
    detailed_predictions.to_csv('svr_hotels_ranked_log.csv', sep=',', index=False)

    #i = 0
    #for pred, actual in zip(y_pred, y_test):
    #    print(f"{pred:.2f}\t\t{actual:.2f}")
    #    i += 1
    #    if i == 10:
    #        break
    return svr_model