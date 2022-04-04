#import libraries
import pandas as pd
import numpy as np
from timeit import default_timer as timer

start = timer()

def TrainTestSplit(X, y, raw_data, split = 0.8):
    X_train = X[:int(raw_data.shape[0]*0.8)]
    X_test = X[int(raw_data.shape[0]*0.8):]
    y_train = y[:int(raw_data.shape[0]*0.8)]
    y_test = y[int(raw_data.shape[0]*0.8):]
    return X_train, X_test, y_train, y_test
 
def ValidationSplit(X_train, y_train, split = 0.8):
    X_train = X_train[:int(X_train.shape[0]*0.8)]
    X_val = X_train[int(X_train.shape[0]*0.8):]    
    y_train = y_train[:int(y_train.shape[0]*0.8)]
    y_val = y_train[int(y_train.shape[0]*0.8):]  
    return X_train, X_val, y_train, y_val
    
raw_data = pd.read_excel("raw_data.xlsx")    

#Splitting data to predictors and response
y = raw_data["Team1Win"]
X = raw_data.loc[:, raw_data.columns != "Team1Win"]

X_train, X_test, y_train, y_test = TrainTestSplit(X,y,raw_data)
X_train, X_val, y_train, y_val = ValidationSplit(X_train,y_train)

y_test = y_test.to_frame()
y_train = y_train.to_frame()
y_val = y_val.to_frame()

y_val['index_col'] = y_val.index
y_test['index_col'] = y_test.index
y_train['index_col'] = y_train.index

X_train['index_col'] = X_train.index
X_test['index_col'] = X_test.index
X_val['index_col'] = X_val.index

df_train = pd.merge(X_train, y_train,how="left")
df_val = pd.merge(X_val, y_val,how="left")
df_test = pd.merge(X_test, y_test,how="left")

df_train = df_train.iloc[:,5:]
df_val = df_val.iloc[:,5:]
df_test = df_test.iloc[:,5:]

df_train = df_train.drop(["Team1AST RATIO", "Team2AST RATIO", "Team1PIE", "Team2PIE"], axis = 1)
df_val = df_val.drop(["Team1AST RATIO", "Team2AST RATIO", "Team1PIE", "Team2PIE"], axis = 1)
df_test = df_test.drop(["Team1AST RATIO", "Team2AST RATIO", "Team1PIE", "Team2PIE"], axis = 1)

df_test = df_test.drop('index_col',1).values
df_train = df_train.drop('index_col',1).values
df_val = df_val.drop('index_col',1).values

y_test = y_test.drop('index_col',1).values
y_train = y_train.drop('index_col',1).values
y_val = y_val.drop('index_col',1).values

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
normalize_dataset(df_train,find_min_max(df_train))
normalize_dataset(df_test,find_min_max(df_test))
normalize_dataset(df_val,find_min_max(df_val))

def sigmoid(z):
    sig = (1/(1+np.exp(-z)))
    return sig

def model_fit_full(features, labels, rate,iterations):
    weights = np.zeros(len(features[0])+1) 
    beta0 = weights[0]
    betai = weights[1:len(weights)]
    for iteration in range(iterations):
        z = beta0 + np.dot(features, betai)
        predicted_labels = sigmoid(z)
        error = np.subtract(labels,predicted_labels)[:,0]
        gradient = np.dot(features.T, error)
        beta0 += rate * np.sum(error)
        betai += rate * gradient
    predicted_weights = np.append(beta0, betai)
    return predicted_weights

def predict(test_features,weights):
    beta0 = weights[0]
    betai = weights[1:len(weights)]
    predicted_labels = []
    for c in range(len(test_features)):
        score = beta0 + np.sum(np.dot(test_features[c], betai))
        prob_true = sigmoid(score)
        prob_false = 1-sigmoid(score)
        if prob_false >= prob_true:
            prediction = 0
        else:
            prediction = 1
        predicted_labels.append(prediction)
    return predicted_labels

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
iteration = [200, 1000, 2000]
learning_rates = [1e-7,1e-6,1e-5]
accuracy_max = 0
perf_metrics = []
for iterations in iteration:
    for rate in learning_rates:
        weights = model_fit_full(df_train, y_train,rate,iterations)
        predicted_labels = predict(df_test,weights)
        accuracy,tp,tn,fp,fn = calculate_accuracy(predicted_labels,y_test)
        print("When learning rate is equal to {} and iteration is equal to {}".format(rate,iterations))
        print("Accuracy: %"+str(accuracy))
        print("tp: "+str(tp))
        print("tn: "+str(tn))
        print("fp: "+str(fp))
        print("fn: "+str(fn))
        performance_metrics(tp,tn,fp,fn)
        if accuracy_max < accuracy:    
            accuracy_max = accuracy
            params = [rate, iterations]
        perf_metrics.append([iterations,rate,accuracy])
################# Validation ##################################################
    
print("Performance metrics for full gradient ascent when learning rate is equal to 1e-6 and iteration is equal to 2000")
weights = model_fit_full(df_train, y_train,1e-6,2000)
predicted_labels = predict(df_test,weights)
accuracy,tp,tn,fp,fn = calculate_accuracy(predicted_labels,y_test)
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
lr_fpr, lr_tpr, _ = roc_curve(y_test, predicted_labels)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC")
plt.legend()
plt.show()





