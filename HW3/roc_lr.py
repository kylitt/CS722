import numpy as np
import matplotlib.pyplot as plt

# given an array of prdeicted labels and the ground truth:
def positive_rates(pred,truth):
    # combine the arrays.
    roc = np.array([pred,truth]).T
    
    # sort the array based on the predicted labels, largest to smallest.
    sorted_roc = roc[np.argsort(roc[:,0])]
    
    fpDat = []
    tpDat = []
    for i in sorted_roc:
        # initialize the threshold and the confusion matrix
        threshold = i[0]
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        # starting at the largest value of our sorted array: 
        for j in sorted_roc:
            if j[0] >= threshold:
                if j[1] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if j[1] == 1:
                    FN += 1
                else:
                    TN += 1
        # calculate the False Positive Rate
        fpr = FP / (FP + TN)
        fpDat.append(fpr) #FPR
        
        # calculate the True Positive Rate
        tpr = TP / (TP + FN)
        tpDat.append(tpr) #TPR  
    return fpDat, tpDat

# given an array hold fpr and an array holding tpr
def plot_roc(fpr, tpr):
    plt.plot(fpr,tpr, marker='.', label='Logistic')
    plt.plot([0,1],[0,1], linestyle='--', label='No Skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve")
    plt.legend()
    plt.show() 