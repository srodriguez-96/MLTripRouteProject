import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def neuralNetworkModel(X_train, y_train, X_test, y_test, data, X):
    # Model Definition
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression task
    ])

    # Compile the Model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the Model
    history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

    # Evaluate the Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    #Predicting the ratings for all hotels
    data['predicted_stars'] = model.predict(X)

    #Sort data by predicted ratings in descending order
    ranked_hotels = data.sort_values(by='predicted_stars', ascending=False)

    #Print the ranked hotels
    print(ranked_hotels[['name', 'predicted_stars']])

    """#Combine predictions with additional hotel details
    detailed_predictions = pd.DataFrame({
        'Hotel Name': data.loc[y_test.index, 'name'],
        'Address': data.loc[y_test.index, 'address'],
        'Actual Stars': y_test,
        'Predicted Stars': np.round(y_pred, 2)  #Rounding the predictions
    }).sort_values(by='Predicted Stars', ascending=False)  #Sort by predicted stars for ranking"""

    #Saving the data to a CSV file
    ranked_hotels.to_csv('hotels_ranked_neural_network.csv', sep=',', index=False)

    return model
