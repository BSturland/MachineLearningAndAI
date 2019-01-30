"""
Created on Thu Oct 18 16:16:12 2018
AI & Machine Learning Course Work
           _____            __  __            _     _              _                           _             
     /\   |_   _|   ___    |  \/  |          | |   (_)            | |                         (_)            
    /  \    | |    ( _ )   | \  / | __ _  ___| |__  _ _ __   ___  | |     ___  __ _ _ __ _ __  _ _ __   __ _ 
   / /\ \   | |    / _ \/\ | |\/| |/ _` |/ __| '_ \| | '_ \ / _ \ | |    / _ \/ _` | '__| '_ \| | '_ \ / _` |
  / ____ \ _| |_  | (_>  < | |  | | (_| | (__| | | | | | | |  __/ | |___|  __/ (_| | |  | | | | | | | | (_| |
 /_/    \_\_____|  \___/\/ |_|  |_|\__,_|\___|_| |_|_|_| |_|\___| |______\___|\__,_|_|  |_| |_|_|_| |_|\__, |
                                                                                                        __/ |
@author: Suhan Ahmed, Ben Sturland, Marcin Pilarczyk                                                   |___/ 
"""
import numpy as np
import pandas as ti
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import matplotlib.pyplot as plt; plt.rcdefaults()




#Variables to compare accuracy
DTaccuracy = 0.0
RFaccuracy = 0.0
ABaccuracy = 0.0
#Load in dataset and set up Label encoder
dataset = ti.read_csv('..\iris-species\Iris.csv')
le = preprocessing.LabelEncoder()
#f, AC = plt.subplots(nrows = 2);""", sharex=False, sharey=False)"""
""""Setting pandas to show the whole dataframe"""
ti.set_option('display.max_columns', None)
ti.set_option('display.max_rows', None)
ti.set_option('display.max_colwidth', -1)


def PreProcessing():
    global X_train, X_test, y_train, y_test
    global dataset
    #Spliting data to training and test sets
    X=dataset.iloc[:,1:5] #Choose the columns in the dataset for the flower attributes
    #print(X)
    y=dataset.iloc[:,5] #Choose the species
    #print(y)
    #splitting between training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=39)
    #Cleans Dataset
    dataset = dataset.dropna()
    #Changes Species into numbers
    le.fit(dataset['Species'])
    dataset['Species'] = le.transform(dataset['Species'])
    print("Description of dataset \n")
    print(dataset.describe())

def VisulationSP():
    dataset.plot.scatter(x='SepalLengthCm',y ='SepalWidthCm',c='Species',colormap='brg')
    plt.show()


def DTREE():
    global DTaccuracy #Changes the DTaccuracy variable globally
    model=DecisionTreeClassifier() #Set the current model as a Decision Tree
    model.fit(X_train, y_train) #Fit the train data to the model
    y_pred=model.predict(X_test) #Test the model
    DTaccuracy=accuracy_score(y_test, y_pred) #Generate accuracy score
    #Print accuracy score and classification report
    print('\n Decision tree accuracy = %s' % DTaccuracy + '\n' + '\n' + classification_report(y_test, y_pred))


def RDMFRRST():
    global RFaccuracy #Changes the RFaccuracy variable globally
    model=RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=50) #Set the model to a Random Forest algorithm and set parameters to modify the algorithm
    model.fit(X_train, y_train) #Fit the train data
    y_pred=model.predict(X_test) #Test the model using test data and store it in y_pred
    RFaccuracy=accuracy_score(y_test,y_pred) #Generate accuracy score
    #Print accuracy score and classification report
    print ('Random Forest accuracy = %s' % RFaccuracy + '\n' + '\n' + classification_report(y_test, y_pred))
    

def ABOOST():
    global ABaccuracy
    #adaboost
    model=AdaBoostClassifier(learning_rate=1.9)
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    ABaccuracy =accuracy_score(y_test, y_pred)
    print('Adaboost accuracy = %s' % ABaccuracy +'\n' + '\n' + classification_report(y_test, y_pred))




def CreateBars():
    objects = ('Decision Tree', 'Random Forest', 'Ada Boost') #Define names for bar chart
    y_pos = np.arange(len(objects))
    performance = [DTaccuracy,RFaccuracy,ABaccuracy] #Obtain accuracy of all algorithms
    plt.bar(y_pos, performance, align='center', alpha=1.0) #Set the attributes for the bar chart
    plt.xticks(y_pos, objects)
    """AC[1].set_xticks(y_pos, objects)"""
    plt.ylabel('Accuracy')
    plt.xlabel('\n' + 'Algorithms')
    plt.title('A Graph To Show Algorithms And Their Accuracy')
    plt.show()    


def Main():
    PreProcessing()
    VisulationSP()
    DTREE()
    RDMFRRST()
    ABOOST()
    CreateBars()
 
    
Main()