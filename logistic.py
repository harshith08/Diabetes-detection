import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random as rd

class LogisticRegression:

    '''
    This function is the error representation of the sigmoid curve
    '''
    def __init__(self):
        self.rows = 0 

    def trainTestSplit(self,df,split):
        percentage = int(split*self.rows)
        train = rd.sample(list(df.index),percentage)
        self.train_data = []
        self.test_data= []
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        for i in range(len(list(df.index))):
            if i in train:
                self.train_data.append(self.data[i])
            else:
                self.test_data.append(self.data[i])
        
        for i in range(len(self.test_data)):
            X_test.append(self.test_data[i][:8])
            y_test.append(self.test_data[i][8])

        for i in range(len(self.train_data)):
            X_train.append(self.train_data[i][:8])
            y_train.append(self.train_data[i][8])

        return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)


    def sigmoid(self,x):
        return 1/(1+np.exp(x))

    def costfunction(self):
        pass

    
    def logisticRegression(self,X_train,y_train,learning_rate,epochs):
        
        weights = np.zeros(shape=(len(X_train),1))
        b = 0 # intercept
        all_costs = []
        for i in range(epochs):
            y = np.dot(weights.T,X_train) + b
            y_hat = ob1.sigmoid(y)

            cost = -(1/len(X_train[0]))*np.sum((y_train*np.log(y_hat)) + ((1-y_train)*np.log(1-y_hat)))

            weights_dericative  = (1/len(X_train[0]))*np.dot(y_hat-y_train,X_train.T)
            b_derivative = (1/len(X_train[0]))*np.sum(y_hat - y_train)

            #updating the weights and intercept
            weights = weights - learning_rate*weights_dericative.T
            b = b - learning_rate*b_derivative

            all_costs.append(cost)

            if (i%(epochs/10)) == 0:
                print("Cost for ",i," epochs = ",cost)

        return weights,b,all_costs
        # return 1,2,3

    def fit(self,X_test,y_test,weights,b):

        y = np.dot(weights.T,X_test) + b
        y_hat = ob1.sigmoid(y)

        y_hat = y_hat > 0.5

        y_hat = np.array(y_hat, type = int)

        accuracy = (1 - np.sum(np.abs(y_hat - y_test))/len(y_test[0]))

        print(accuracy*100,"%")

    def main(self):
        df = pd.read_csv("diabetes.csv")
        self.data = np.array(df[df.columns])
        self.rows = len(self.data)
        X_train,X_test,y_train,y_test = ob1.trainTestSplit(df,split=0.70)
        
        #data reshaping 
        X_train = X_train.T
        y_train = y_train.reshape(1,len(X_train[0]))
        X_test = X_test.T
        y_test = y_test.reshape(1,len(X_test[0]))

        w,b,c = ob1.logisticRegression(X_train,y_train,learning_rate=0.0005,epochs=10000)
        ob1.fit(X_test,y_test,w,b)

        
        
        
if __name__ == "__main__":
    ob1 = LogisticRegression()
    ob1.main()
