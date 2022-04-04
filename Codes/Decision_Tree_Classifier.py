import pandas as pd
import numpy as np
from timeit import default_timer as timer

start = timer()

def entropy(data): 
    labels = np.unique(data)
    entropy = 0
    for label in labels:
        p = len(data[data == label]) / len(data)
        entropy += -p * np.log2(p)
    return entropy

def gini_index(data):    
    labels = np.unique(data)
    gini = 0
    for label in labels:
        p = len(data[data == label]) / len(data)
        gini += p**2
    gini = 1 - gini
    return gini

def information_gain(parent, l_child, r_child):
    weight_l = len(l_child) / len(parent)
    weight_r = len(r_child) / len(parent)
    gain = gini_index(parent) - (weight_l * gini_index(l_child) + weight_r * gini_index(r_child))
    return gain

def split(dataset, feature_index, threshold):    
    dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
    dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
    return dataset_left, dataset_right

def get_best_split(dataset, num_samples, num_features):
    best_split = {}
    max_info_gain = -1000000000
    for feature_index in range(num_features):
        feature_values = dataset[:, feature_index]
        candidate_thresholds = np.unique(feature_values)   
        for threshold in candidate_thresholds:
            dataset_left, dataset_right = split(dataset, feature_index, threshold)
            if len(dataset_left) > 0 and len(dataset_right) > 0:
                y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                curr_info_gain = information_gain(y, left_y, right_y)
                if curr_info_gain > max_info_gain:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["dataset_left"] = dataset_left
                    best_split["dataset_right"] = dataset_right
                    best_split["info_gain"] = curr_info_gain
                    max_info_gain = curr_info_gain                 
    return best_split
        
def calculate_leaf_value(Y):
    Y = list(Y)
    return max(Y, key=Y.count)     
        
def build_tree(dataset, curr_depth=0, min_samples_split = 2, max_depth= 2):
    X, Y = dataset[:,:-1], dataset[:,-1]
    num_samples, num_features = np.shape(X)
    if num_samples >= min_samples_split and curr_depth <= max_depth:
        best_split = get_best_split(dataset, num_samples, num_features)
        if best_split["info_gain"]>0:
            left_subtree = build_tree(best_split["dataset_left"], curr_depth+1, min_samples_split, max_depth)
            right_subtree = build_tree(best_split["dataset_right"], curr_depth+1, min_samples_split, max_depth)
            node_obj = Node(best_split["feature_index"], best_split["threshold"], 
                        left_subtree, right_subtree, best_split["info_gain"])
            return node_obj   
    leaf_value = calculate_leaf_value(Y)
    node_obj = Node(value=leaf_value)
    return node_obj
    
def fit(X, Y, min_sample_split=2, mx_depth=2):        
    dataset = np.concatenate((X, Y), axis=1)
    tree = build_tree(dataset,min_samples_split = min_sample_split, max_depth = mx_depth)    
    return tree
        
def predict(test_set, root):        
    prediction_list = [make_prediction(test_instance, root) for test_instance in test_set]
    return prediction_list
    
def make_prediction(test_instance, tree):        
    if tree.value != None: 
        return tree.value
    feature_val = test_instance[tree.feature_index]
    if feature_val <= tree.threshold:
        return make_prediction(test_instance, tree.left)
    else:
        return make_prediction(test_instance, tree.right)
    
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

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

def TrainTestSplit(X, y, raw_data, split = 0.8):
    
    X_train = X[:int(raw_data.shape[0]*0.8)]
    X_test = X[int(raw_data.shape[0]*0.8):]
    y_train= y[:int(raw_data.shape[0]*0.8)].values.reshape(-1, 1)
    y_test = y[int(raw_data.shape[0]*0.8):].values.reshape(-1, 1)
    return X_train, X_test, y_train, y_test

def ValidationSplit(X_train, y_train, split = 0.8):
    X_train = X_train[:int(X_train.shape[0]*0.8)]
    X_val = X_train[int(X_train.shape[0]*0.8):]    
    y_train = y_train[:int(y_train.shape[0]*0.8)]
    y_val = y_train[int(y_train.shape[0]*0.8):]  
    return X_train, X_val , y_train, y_val


raw_data = pd.read_excel("raw_data.xlsx")    

y = raw_data["Team1Win"]
X = raw_data.drop(["Visitor/Neutral","PTS","Home/Neutral",
                   "PTS.1","Team1Win","Team1AST RATIO", "Team2AST RATIO", "Team1PIE", "Team2PIE"], axis = 1)

X_train, X_test, y_train, y_test = TrainTestSplit(X,y,raw_data)
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train, X_val, y_train, y_val = ValidationSplit(X_train,y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)



################# Validation ##################################################
mx_depth = [2,3,4,5]
min_sample_split = [10,100,1000,10000]
accuracy_max = 0
perf_metrics = []
for depth in mx_depth:
    for min_split in min_sample_split:
        decision_tree = fit(X_train, y_train, min_split, depth)
        prediction = predict(X_val, decision_tree)
        accuracy,tp,tn,fp,fn = calculate_accuracy(prediction, y_val)
        print("When max depth is equal to {} and minimum sample split is equal to {} ".format(depth,min_split))
        print("Accuracy: %"+str(accuracy))
        print("tp: "+str(tp))
        print("tn: "+str(tn))
        print("fp: "+str(fp))
        print("fn: "+str(fn))
        performance_metrics(tp,tn,fp,fn)
        if accuracy_max < accuracy:    
            accuracy_max = accuracy
            params = [min_split, depth]
        perf_metrics.append([depth,min_split,accuracy])
################# Validation ##################################################

decision_tree = fit(X_train, y_train, 10, 5)
prediction = predict(X_test, decision_tree)

accuracy,tp,tn,fp,fn = calculate_accuracy(prediction,y_test)
print("When max depth is equal to 5 and minimum sample split is equal to 10 ")
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
lr_fpr, lr_tpr, _ = roc_curve(y_test, prediction)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC")
plt.legend()
plt.show()



