import pandas as pd
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np

def NNet(X,y,test):



    #rows = int(X.shape[0])
    cols = int(X.shape[1])

    net = NeuralNet(
                   layers = [
                               ('input',layers.InputLayer),
                               ('hidden1',layers.DenseLayer),
                               ('dropout1',layers.DropoutLayer),
                               ('hidden2',layers.DenseLayer),
                               ('dropout2',layers.DropoutLayer),
                               ('hidden3',layers.DenseLayer),
                               ('dropout3',layers.DropoutLayer),
                               ('hidden4',layers.DenseLayer),
                               #('dropout4',layers.DropoutLayer),
                               ('output',layers.DenseLayer),
                            ],
                            input_shape = (None,cols),
                            hidden1_num_units = 800,
                            dropout1_p = 0.4,
                            hidden2_num_units = 500,
                            dropout2_p = 0.3,
                            hidden3_num_units = 300,
                            dropout3_p = 0.3,
                            hidden4_num_units = 200,
                            #dropout4_p = 0.2,


                            output_num_units = len(np.unique(y)),
                            output_nonlinearity = lasagne.nonlinearities.softmax,

                             update=nesterov_momentum,
                             update_learning_rate=0.0001,
                             update_momentum=0.9,
                             max_epochs = 300,
                             verbose = 1,
                )

  #  net.load_params_from('w3')
  #  details = net.get_all_params()
  #  oldw = net.get_all_params_values()
    '''
    skf = cross_validation.StratifiedKFold(y,n_folds = 7)
    blend_train = np.zeros(X.shape[0])
    prediction = []
    blend_test_j = np.zeros((test.shape[0], len(skf)))

    for i,(train_index,cv_index) in enumerate(skf):
            print "Fold:",i
            X_train = X[train_index,]
            y_train = y[train_index]
            X_cv = X[cv_index,]
            #y_cv = y[cv_index]
            net.fit(X_train,y_train)

            blend_train[cv_index] = net.predict_proba(X_cv)[:,1]
            blend_test_j[:,i] = net.predict_proba(test)[:,1]
    prediction = blend_test_j.mean(1)
    '''
    net.fit(X,y)
    prediction = net.predict_proba(test)
    return prediction
