import keras_metrics as km
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc,classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

scores = ['f1','precision', 'recall']
scoring = {'f1':'f1', 'precision':'precision','recall':'recall'}
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh','sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)

def create_model(activation='relu'):
    model = Sequential()
    model.add(Dense(4, input_dim=6, activation=activation))
    model.add(Dense(4, activation=activation))
    model.add(Dense(2, activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

#def create_model(activation='relu'):
	# create model
#paremetros buscados

#model = Sequential()
#model.add(Dense(4, input_dim=6, activation='tanh'))
#model.add(Dense(4, activation='tanh'))
#model.add(Dense(2, activation='tanh'))
 #model.add(Dense(6,  kernel_initializer='uniform',activation=activation))
#model.add(Dense(1, activation='tanh'))
 # Compile model
#model.compile(loss='binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
#model.save('model.h5')
 #model.summary()
# return model
seed = 7
np.random.seed(seed)
# load dataset
#data = numpy.loadtxt('interface_data.csv')
#np.savetxt(file_name, array, delimiter=",", header='x,y,z, data from monte carlo simulation')
data = np.loadtxt('interface_data_sola.csv',delimiter=",", skiprows=1)
X=data[:,0:6]
Y=data[:,7]
print(X)
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X ,Y,test_size=0.30)
X_train = preprocessing.scale(x_train)
X_test = preprocessing.scale(x_test)

for score in scores:
    mlp = KerasClassifier(create_model, epochs=300, batch_size=32, verbose=0)
    grid = GridSearchCV(mlp, param_grid=param_grid, n_jobs=-1, cv=5,scoring= '%s_macro' % score)
    grid_result = grid.fit(X_train,y_train)

    print("Best parameters set found on development set:")
    print()
    print(grid_result.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    #print("Detailed classification report:")
    #print()
    #y_pred =grid_result.predict_classes(X_test)
    #y_pred =grid.predict_classes(X_test)
    #print(confusion_matrix(y_test , y_pred))
    #print(classification_report(y_test , y_pred))
    #print()
