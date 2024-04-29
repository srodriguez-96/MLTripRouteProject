from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main():
    # Call functions from svm_classifier.py

    sc = StandardScaler()

    while 1:
        hotelName = input("Enter your name: ")
        hp1 = input("Enter hotel parameter 1: ")
        hp2 = input("Enter hotel parameter 2: ")
        hp3 = input("Enter hotel parameter 3: ")

        new_data_point = sc.transform([[hp1, hp2, hp3]])

        
        

if __name__ == "__main__":
    main()