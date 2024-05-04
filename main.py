from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import svm
import dataCleaning
import logisticReg
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

    #gets data from the dataClean function
    x_train, x_test, y_train, y_test, data, valueX = dataCleaning.dataClean()

    #calls linearReg with data
    linearModel = linearReg.linearRegModel(x_train, y_train, x_test, y_test, data, valueX)
    svrModel = svm.svm_regression(x_train, y_train, x_test, y_test, data, valueX)
    #logisticModel = logisticReg.logisticRegModel(x_train, y_train, x_test, y_test, data, valueX)

    while True:
        # Take user input for hotel details
        hotelName = input("Enter hotel name: ")
        hp1 = input("Accepts Credit Card (yes/no): ").lower() == "yes"
        hp2 = input("Price Range (cheap, moderate, expensive): ")
        hp3 = input("Good For Kids (yes/no): ").lower() == "yes"
        hp4 = input("Outdoor Seating (yes/no): ").lower() == "yes"
        hp5 = input("Noise Level (low, quite, average, loud, very loud, none): ")

        hp2_2 = hp2_3 = hp2_4 = hp2_unk = False
        hp5_l = hp5_q = hp5_n = hp5_a = hp5_uL = hp5_uq = hp5_uVL = hp5_unk = False



        
        #for Price Range
        if hp2 == 'cheap':
            hp2_2 = True
            
        elif hp2 == 'moderate':
            hp2_3 = True
            
        elif hp2 == 'expensive':
            hp2_4 = True
            
        else:
            hp2_unk = True
            
        
        #for noise level
        if hp5 == 'low':
            hp5_l = True
            
        elif hp5 == 'quite':
            hp5_q = True
            
        elif hp5 == 'none':
            hp5_n = True
            
        elif hp5 == 'average':
            hp5_a = True
            
        elif hp5 == 'low':
            hp5_uL = True
            
        elif hp5 == 'quite':
            hp5_uq = True
            
        elif hp5 == 'very loud':
            hp5_uVL = True
            
        else:
            hp5_unk = True
            

        # Prepare user input data for prediction
        user_data = [hp1,hp2_2,hp2_3, hp2_4, hp2_unk,hp3,hp4,hp5_l,hp5_q,hp5_n,hp5_a,hp5_uL,hp5_uq,hp5_uVL,hp5_unk ]
        userDataHeaders = ['accepts_credit_cards', 'good_for_kids', 'outdoor_seating', 'price_range_2', 'price_range_3', 'price_range_4', 'price_range_unknown', "noise_level_'loud'", "noise_level_'quiet'", 'noise_level_None', "noise_level_u'average'", "noise_level_u'loud'", "noise_level_u'quiet'", "noise_level_u'very_loud'", 'noise_level_unknown']
        user_data = pd.DataFrame([user_data], columns=userDataHeaders)
        # Make predictions using the linear regression model
        linearPred = linearModel.predict(user_data)
        svrPred = svrModel.predict(user_data)


        #UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names

        # Display predictions
        print("\nPredictions:")
        print(f"Linear Regression: {linearPred[0]}")
        print(f" SVR: {svrPred[0]}")
        # Ask user if they want to continue
        choice = input("\nDo you want to predict for another hotel? (yes/no): ").lower()
        if choice != 'yes':
            break


        

if __name__ == "__main__":
    main()