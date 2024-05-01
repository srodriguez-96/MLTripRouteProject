import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def linearRegModel(X_train, y_train, X_test, y_test, data, userData):
    #Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)


    #Predicting and calculating MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    #Predicting the ratings for all hotels
    data['predicted_stars'] = model.predict(X)

    #Sort data by predicted ratings in descending order
    ranked_hotels = data.sort_values(by='predicted_stars', ascending=False)

    #Print the ranked hotels
    print(ranked_hotels[['name', 'predicted_stars']])

    #Combine predictions with additional hotel details
    detailed_predictions = pd.DataFrame({
        'Hotel Name': data.loc[y_test.index, 'name'],
        'Address': data.loc[y_test.index, 'address'],
        'Actual Stars': y_test,
        'Predicted Stars': np.round(y_pred, 2)  #Rounding the predictions
    }).sort_values(by='Predicted Stars', ascending=False)  #Sort by predicted stars for ranking

    #Saving the data to a CSV file
    detailed_predictions.to_csv('hotels_ranked.csv', sep=',', index=False)

    return model