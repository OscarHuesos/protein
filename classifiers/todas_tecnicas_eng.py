import keras_metrics as km
import pandas as pd
import numpy as np
import re
import seaborn as sn
from io import StringIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import sparse_categorical_crossentropy
from sklearn.metrics import matthews_corrcoef
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.metrics import plot_roc_curve,roc_curve, auc,classification_report, confusion_matrix, f1_score,plot_confusion_matrix,roc_auc_score
from matplotlib import pyplot

#def create_baseline():
model = Sequential()
model.add(Dense(8, input_dim=6, activation='tanh'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.save('model.h5')
model.summary()

random_state = np.random.RandomState(0)
seed = 7
np.random.seed(seed)

#Execute 10 times
data=np.genfromtxt("interface_data_sola.csv", delimiter=',', skip_header=1)
print("data")
print(data)
X=data[:,0:6]
Y=data[:,7]
print(X)
print(Y)
print(Y.astype)
# Binarize the output
y = label_binarize(Y, classes=[0, 1])
nc = len(y)
x_train, x_test, y_train, y_test = train_test_split(X ,y,test_size=0.30)
#scaling
X_train = preprocessing.scale(x_train)
X_test = preprocessing.scale(x_test)
mlp=model.fit(X_train, y_train, epochs=300, batch_size=32, verbose=0)
print("Reporte de clasifocacion keras:")
print()
y_mlp = model.predict_classes(X_test)
pred_mlp= model.predict(X_test)
print(confusion_matrix(y_test , y_mlp))
print(classification_report(y_test , y_mlp))
print()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, pred_mlp)
auc_keras = auc(fpr_keras, tpr_keras)
MCC_mlp=matthews_corrcoef(y_test, y_mlp)
print("MCC mlp:")
print(MCC_mlp)

log=LogisticRegression(random_state=random_state,C=100,penalty='l2',solver='liblinear',tol=0.01)
log_result=log.fit(x_train,y_train)
print("Reporte de clasifocacion:")
print()
y_log =log_result.predict_proba(x_test)
y_log2 =log_result.predict(x_test)

fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_log2,pos_label=1)
auc_log = auc(fpr_log, tpr_log)
MCC_log=matthews_corrcoef(y_test, y_log2)
print("MCC log:")
print(MCC_log)

print(confusion_matrix(y_test , y_log2))
print(classification_report(y_test , y_log2 ))

rfc=RandomForestClassifier(n_estimators=500,criterion='entropy',oob_score=True)
rfc_result=rfc.fit(x_train,y_train)
#scores = ['f1','precision','recall']
print("Reporte de clasifocacion forest:")
print()
#y_rfc =rfc_result.predict_classes(x_test)
y_rfc =rfc_result.predict(x_test)
print(confusion_matrix(y_test , y_rfc))
print(classification_report(y_test , y_rfc ))
print()
OOB=rfc_result.oob_score_
print("OOB error:")
print(OOB)
IMP=rfc_result.feature_importances_.argsort()
print("Mejores caracteristicas")
print(IMP)

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_test, y_rfc,pos_label=1)
auc_forest = auc(fpr_forest, tpr_forest)
MCC_forest=matthews_corrcoef(y_test, y_rfc)
print("MCC forest:")
print(MCC_forest)

pyplot.figure()
cf_matrix=confusion_matrix(y_test, y_rfc)
group_names = ['TN','FP','FN','TP']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}" for v1,v2 in zip(group_names,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sn.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',cbar=False)
#pyplot.title('Random forest classifier')
pyplot.xlabel('Predicted values')
pyplot.ylabel('True values')
ax= pyplot.subplot()
ax.set_yticks([0, 1])

ax.xaxis.set_ticklabels(['Non-hot spot/non-hot region','Hot spot/hot region'])
ax.yaxis.set_ticklabels(['Non-hot spot/non-hot region','Hot spot/hot region'])
ax.yaxis.set_ticks_position('both')
pyplot.show()

svm=SVC(kernel='rbf',gamma=0.01,C=10,coef0=0,degree=1)
svm_result=svm.fit(x_train,y_train)
print("Reporte de clasifocacion:")
print()
y_svm =svm_result.predict(x_test)
print(classification_report(y_test , y_svm ))
print(confusion_matrix(y_test , y_svm))
print()

fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_svm,pos_label=1)
auc_svm = auc(fpr_svm, tpr_svm)
MCC_svm=matthews_corrcoef(y_test, y_svm)
print("MCC svm:")
print(MCC_svm)

naive=GaussianNB(var_smoothing= 1.0)
naive_result=naive.fit(x_train,y_train)
print("Reporte de clasifocacion gussian:")
print()
y_naive = naive_result.predict(x_test)
print(classification_report(y_test, y_naive ))
print(confusion_matrix(y_test ,  y_naive))

fpr_naive, tpr_naive, thresholds_naive= roc_curve(y_test, y_naive)
auc_naive = auc(fpr_naive, tpr_naive)
MCC_naive=matthews_corrcoef(y_test, y_naive)
print("MCC naive:")
print(MCC_naive)

pyplot.figure()
pyplot.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
pyplot.plot(fpr_keras, tpr_keras, label='Neuronal Networks Classifier (area = {:.3f})'.format(auc_keras))
#pyplot.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
ax = pyplot.gca()
smv_plot= plot_roc_curve(svm_result, x_test, y_test,ax=ax,label='Support Vector Classifier (area = {:.3f})'.format(auc_svm))
forest_plot= plot_roc_curve(rfc_result, x_test, y_test,ax=ax,label='Random Forest Classifier (area = {:.3f})'.format(auc_forest))
log_plot= plot_roc_curve(log_result, x_test, y_test,ax=ax,label='Logistic Regression Classifier (area = {:.3f})'.format(auc_log))
pyplot.grid(True)
pyplot.title('Performance of the HotSpot Classifiers of Interface Residues')
pyplot.show()
