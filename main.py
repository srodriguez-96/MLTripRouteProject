from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import svm
import dataCleaning
import linearReg
import pandas as pd


def main():
    # Call functions from svm_classifier.py
    # old dataFilter code --------------------------------------------------------------------------------------------
    #importing csv
    # Columns to remove
    #columns_to_remove = [0,2,3,4,5,6,7,9,10]  # For example, remove the 2nd and 4th columns

    # Read CSV file into a DataFrame without header
    #df = pd.read_csv('hotels_only.csv', header=None)

    # Remove specified columns
    #df = df.drop(columns=columns_to_remove, axis=1)

    # Use different models -> XG Boost, #logistic Regression, Rand Forest
    # instead of deleting features just fill with filler values
    # do the akdjklawdkl;

    #feature1 = 'BusinessAcceptsCreditCards'
    #feature2 = 'WiFi'
    #feature3 = 'RestaurantsPriceRange2'
    #dropRows = []
    #feature1List = []
    #feature2List = []
    #feature3List = []

    #for index, row in df.iterrows():
    #    value = row.iloc[2]
    #    stringDict = eval(value)
    #    
    #    if feature1 & feature2 & feature3 in stringDict:
            #if feature are in this row we add to the list to append to the df later
            #these lists will be the new features that are extracted
    #        value1 = stringDict.get(feature1)
    #        feature1List.append(value1)
    #        value2 = stringDict.get(feature2)
    #        feature2List.append(value2)
    #        value3 = stringDict.get(feature3)
    #        feature3List.append(value3)
    #    else:
            #if feature not in this row we are going to drop this row later
    #        dropRows.append(index)
            
        #delete break later
    #    break
    # --------------------------------------------------------------------------------------------


    #while 1:
        #hotelName = input("Enter your name: ")
        #hp1 = input("Enter hotel parameter 1: ")
        #hp2 = input("Enter hotel parameter 2: ")
        #hp3 = input("Enter hotel parameter 3: ")

        #new_data_point = sc.transform([[hp1, hp2, hp3]])
    
    #gets data from the dataClean function
    x_train, x_test, y_train, y_test, data, valueX = dataCleaning.dataClean()

    #calls linearReg with data
    linearModel = linearReg.linearRegModel(x_train, y_train, x_test, y_test, data, valueX)
    svmModel = svm.svm_regression(x_train, y_train, x_test, y_test)

        
        

if __name__ == "__main__":
    main()