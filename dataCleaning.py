import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def dataClean():
    #Loading the CSV file with appropriate column names
    column_names = ['business_id', 'name', 'address', 'city', 'state', 'postal_code',
                    'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'attributes', 'categories']
    #data = pd.read_csv('hotels_only.csv', header=None, names=column_names)
    data = pd.read_csv('hotels_only.csv', header=None, names = column_names)
    data.head()

    def parse_features(attributes):
        try:
            #Convert the string into a dictionary
            attributes_dict = eval(attributes)
        except:
            #if there is an error return an empty dictionary
            attributes_dict = {}
        return attributes_dict

    # Create a new column 'attributes_dict' from 'attributes' column
    data['attributes_dict'] = data['attributes'].apply(parse_features)

    # Extract individual features
    data['accepts_credit_cards'] = data['attributes_dict'].apply(lambda x: x.get('BusinessAcceptsCreditCards', 'False') == 'True')
    data['price_range'] = data['attributes_dict'].apply(lambda x: x.get('RestaurantsPriceRange2', 'unknown'))
    data['good_for_kids'] = data['attributes_dict'].apply(lambda x: x.get('GoodForKids', 'False') == 'True')
    data['outdoor_seating'] = data['attributes_dict'].apply(lambda x: x.get('OutdoorSeating', 'False') == 'True')
    data['noise_level'] = data['attributes_dict'].apply(lambda x: x.get('NoiseLevel', 'unknown'))

    #Verify the columns are added
    print(data[['name', 'accepts_credit_cards', 'price_range', 'good_for_kids', 'outdoor_seating', 'noise_level']].head())


    #Use pd.get_dummies to handle categorical data
    X = pd.get_dummies(data[['accepts_credit_cards', 'price_range', 'good_for_kids', 'outdoor_seating', 'noise_level']], drop_first=True)
    y = data['stars']

    #Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test, data, X