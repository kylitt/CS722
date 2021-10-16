import numpy as np
Learning_Rate=0.01
num_iterations=100000
target_cost = 0.5
X=0
y=0
w=0

# given an initialized training set and weights:
def train(X,y,w):
    N = len(X)
    for i in range(num_iterations):
        # calculate all of the features times their weights
        Z = np.dot(X,w)

        # apply the sigmoid function to all Z
        y_pred = 1/(1+np.exp(-Z))
     
        # calcualte the average cost
        cost = (np.sum(np.log( 1 + np.exp(Z)) - y*Z)) / N
        
        # generate the change in weights
        dw = np.dot(X.T, (y_pred-y)) / N
        
        # apply the change in weights
        w = w - Learning_Rate*dw
        
        # if the cost gets to a predefined target, break early
        if cost < target_cost:
            print(cost)
            print(i)
            break
    return w

# given an initialized test set and their weights
def test(X,y,w):
    # calculate all of the features times their weights
    Z = np.dot(X,w)
    
    # apply the sigmoid function to all Z
    y_pred = 1/(1+np.exp(-Z))

    # using numpy creates a (1) dimensional array. I need a (0) dimensional array to match the label data.
    y_pred=np.reshape(y_pred.T,-1)
    return y_pred