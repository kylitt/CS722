from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from train_lr import train, test
from roc_lr import positive_rates, plot_roc

# Load data
data = load_breast_cancer()
X = data.data

# I chose to normalize the data using sklearn. I was having issues with overflow.
norm_X = preprocessing.normalize(X)
label = data.target

# split the data into train and test.
X_train, X_test, y_train, y_test = train_test_split(norm_X, label, test_size=0.3, random_state=24)

# randomly initialize the weights.
w=np.random.random(X.shape[1])

# generate weights
trained_weight= train(X_train,y_train,w)

# make a prediction with the weights
pred = test(X_test,y_test,trained_weight)

# find the Fale Positive and True Positive Rates
lr_fpr, lr_tpr = positive_rates(pred,y_test)

# plot the ROC curve
plot_roc(lr_fpr, lr_tpr)