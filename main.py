from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import svm
import pandas as pd


def main():
    # Call functions from svm_classifier.py

    #importing csv
    # Columns to remove
    columns_to_remove = [0,2,3,4,5,6,7,9,10]  # For example, remove the 2nd and 4th columns

    # Read CSV file into a DataFrame without header
    df = pd.read_csv('hotels_only.csv', header=None)

    # Remove specified columns
    df = df.drop(columns=columns_to_remove, axis=1)

    for index, row in df.iterrows():
        value = row.iloc[2]
        print(value)
        print(type(value))
        break

    sc = StandardScaler()

    #while 1:
        #hotelName = input("Enter your name: ")
        #hp1 = input("Enter hotel parameter 1: ")
        #hp2 = input("Enter hotel parameter 2: ")
        #hp3 = input("Enter hotel parameter 3: ")

        #new_data_point = sc.transform([[hp1, hp2, hp3]])

        
        

if __name__ == "__main__":
    main()