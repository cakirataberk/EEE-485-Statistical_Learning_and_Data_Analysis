import numpy as np  
import pandas as pd 
from timeit import default_timer as timer

start = timer()

def TrainTestSplit(X,y,raw_data,split = 0.8):
    X_train = X[:int(raw_data.shape[0]*0.8)]
    X_test = X[int(raw_data.shape[0]*0.8):]
    y_train= y[:int(raw_data.shape[0]*0.8)]
    y_test = y[int(raw_data.shape[0]*0.8):]
    return X_train, X_test, y_train, y_test

def ValidationSplit(X_train, y_train, split = 0.8):
    X_train = X_train[:int(X_train.shape[0]*0.8)]
    X_val = X_train[int(X_train.shape[0]*0.8):]    
    y_train = y_train[:int(y_train.shape[0]*0.8)]
    y_val = y_train[int(y_train.shape[0]*0.8):]  
    return X_train, X_val , y_train, y_val

raw_data = pd.read_excel("raw_data.xlsx")    

#Splitting data to predictors and response
y = raw_data["Team1Win"]
X = raw_data.drop(["Visitor/Neutral","PTS","Home/Neutral",
                   "PTS.1","Team1Win","Team1AST RATIO", "Team2AST RATIO", "Team1PIE", "Team2PIE"], axis = 1)

X_train, X_test, y_train, y_test = TrainTestSplit(X,y,raw_data)
X_train, X_val, y_train, y_val = ValidationSplit(X_train,y_train)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
X_val = X_val.values
y_val = y_val.values

def find_min_max(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = []
        for row in dataset:
            col_values.append(row[i])
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax
#normalize dataset
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
# dataset normalization       
normalize_dataset(X_train,find_min_max(X_train))
normalize_dataset(X_test,find_min_max(X_test))
normalize_dataset(X_val,find_min_max(X_val))

def fit(X,y,iteration,learning_rate,lambda_param):
    n_samples, n_features = X.shape
    y_hat = np.where(y <= 0, -1, 1)
    w = np.zeros(n_features)
    b = 0
    for _ in range(iteration):
        i = 0
        for x in X: 
            condition = y_hat[i]*(np.dot(x,w)-b) >= 1
            if condition:
                w -= learning_rate*(2*lambda_param*w)
            else:
                w -= learning_rate*(2*lambda_param*w - np.dot(x,y_hat[i]))
                b -= learning_rate * y_hat[i]
            i+=1
    return w,b
    
def predict (X,w,b):
    predict = np.dot(X,w) - b
    return np.sign(predict)

def calculate_accuracy(predicted_labels,test_labels):
    tp = 0.0 #true positive count 
    tn = 0.0 #true negative count
    fp = 0.0 #false positive count
    fn = 0.0 #false negative count
    for i in range(len(predicted_labels)):
        prediction = predicted_labels[i]
        actual = test_labels[i]
        if ((prediction == 1) & (actual == 1)):
            tp+=1
        elif ((prediction == 1) & (actual == 0)):
            fp+=1
        elif ((prediction == 0) & (actual == 0)):
            tn+=1
        elif ((prediction == 0) & (actual == 1)):
            fn+=1
    accuracy = ((tp+tn)/(tp+tn+fp+fn))*100
    return accuracy,tp,tn,fp,fn

def performance_metrics(tp,tn,fp,fn):
    precision = tp/(tp+fp+0.0000001)
    recall = tp/(tp+fn+0.0000001)
    NPV = tn/(tn+fn+0.0000001)
    FPR = fp/(fp+tn+0.0000001)
    FDR = fp/(fp+tp+0.0000001)
    F1_score = 2*precision*recall/(precision+recall+0.0000001)
    F2_score = 5*precision*recall/(4*precision+recall+0.0000001)
    print("Precision: ",precision)
    print("Recall: ",recall)
    print("NPV: ",NPV)
    print("FPR: ",FPR)
    print("FDR: ",FDR)
    print("F1 score: ",F1_score)
    print("F2_score: ",F2_score)

################# Validation ##################################################
#iteration = [200, 1000, 2000]
#learning_rate = [1e-3,1e-4,1e-5]
#lambda_par = [1e-2,1e-3]
#accuracy_max = 0
#perf_metrics = []
#for iterations in iteration:
#    for learning_rates in learning_rate:
#        for lambda_pars in lambda_par:
#            
#            w,b = fit(X_train,y_train,iterations,learning_rates,lambda_pars)
#            predictions = predict(X_test,w,b)
#            
#            true_predictions = []
#            for prediction in predictions:
#                # print(prediction)
#                if prediction == -1:
#                    true_predictions.append(0)
#                else:
#                    true_predictions.append(1)
#            print("When iterations is equal to {} and learning rate is equal to {} and lambda is equal to {}".format(iterations,learning_rates,lambda_pars))
#            accuracy,tp,tn,fp,fn = calculate_accuracy(true_predictions,y_test)
#            print("Accuracy: %"+str(accuracy))
#            print("tp: "+str(tp))
#            print("tn: "+str(tn))
#            print("fp: "+str(fp))
#            print("fn: "+str(fn))
#            performance_metrics(tp,tn,fp,fn)
#            if accuracy_max < accuracy:    
#                accuracy_max = accuracy
#                params = [iterations, learning_rates, lambda_pars]
#            perf_metrics.append([iterations,learning_rates,lambda_pars,accuracy])
################# Validation ##################################################

w,b = fit(X_train,y_train,1000,1e-4,1e-2)
predictions = predict(X_test,w,b)

true_predictions = []
for prediction in predictions:
    # print(prediction)
    if prediction == -1:
        true_predictions.append(0)
    else:
        true_predictions.append(1)
        
accuracy,tp,tn,fp,fn = calculate_accuracy(true_predictions,y_test)
print("When iterations is equal to 1000 and learning rate is equal to 0.0001 and lambda is equal to 0.01")
print("Accuracy: %"+str(accuracy))
print("tp: "+str(tp))
print("tn: "+str(tn))
print("fp: "+str(fp))
print("fn: "+str(fn))
performance_metrics(tp,tn,fp,fn)

end = timer()
print("Time elapsed: ",end - start)

########################### Plot ROC Curve #######################
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
lr_fpr, lr_tpr, _ = roc_curve(y_test, true_predictions)
plt.plot(lr_fpr, lr_tpr, marker='.', label='SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC")
plt.legend()
plt.show()

