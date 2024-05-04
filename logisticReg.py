import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def logisticRegModel(X_train, y_train, X_test, y_test, data, X):
    #Initialize and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)


    #Predicting and calculating log loss
    y_pred = model.predict(X_test)
    logloss = log_loss(y_test, y_pred)
    print(f'Log Loss: {logloss}')

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
    detailed_predictions.to_csv('hotels_ranked_log.csv', sep=',', index=False)

    return model