import keras_metrics as km
import pandas as pd
import numpy as np
import re
import seaborn as sn
from io import StringIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import sparse_categorical_crossentropy
#from tensorflow.keras.optimizers import Adam
#from kerastuner.tuners import RandomSearch
from sklearn.metrics import matthews_corrcoef
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,SVR
from sklearn.ensemble import BaggingClassifier
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.model_selection import RepeatedKFold, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import plot_roc_curve,roc_curve, classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import  make_scorer,roc_auc_score, auc
import matplotlib.pylab as plt
from scipy import interp
from matplotlib import pyplot
from statistics import stdev


data=np.genfromtxt("interface_data_sola.csv", delimiter=',', skip_header=1)
#print("data")
#print(data)
X=data[:,0:6]
Y=data[:,7]
print(X)
print(Y)
print(Y.astype)
y = label_binarize(Y, classes=[0, 1])
yy=np.ravel(y)
nc = len(yy)
print("y")
print(yy)
micros=[]
micros_bag=[]
estimators=[]
recalls=[]
recalls_bag=[]
MCC=[]
MCC_bag=[]
AUCS=[]
AUCS_bag=[]
fig_nn = plt.figure(figsize=[12,12])
ax1 = fig_nn.add_subplot(111,aspect = 'auto')
mean_fpr_nn = np.linspace(0,1,100)
cont=0
trees=0
depth=100
x_train, x_test, y_train, y_test = train_test_split(X ,y,
test_size=0.30, random_state=42)
for i in range(1,11):
 trees=i*100
 print("trees es")
 print(trees)
 estimators.append(trees)
 #for j in range(1,6,1):
 rfc=RandomForestClassifier(n_estimators= trees,
 criterion='entropy', oob_score=True, max_depth=depth)
 bag=BaggingClassifier(rfc,
 n_estimators=150, random_state=0)
 bag_results=bag.fit(x_train, y_train)
 rfc_result=rfc.fit(x_train, y_train)
  #scores = ['f1','precision','recall']
 print("Reporte de clasifocacion forest:")
 print()
 y_rfc =rfc_result.predict(x_test)
 print(confusion_matrix(y_test , y_rfc))
 print(classification_report(y_test , y_rfc ))
 print()
 acc=accuracy_score(y_test , y_rfc)
 print("accuaracy")
 print(acc)
 prec=precision_score(y_test , y_rfc)
 print("precision")
 print(prec)
 rec=recall_score(y_test , y_rfc)
 print("recall")
 print(rec)
 funo=f1_score(y_test , y_rfc)
 print("f1 score")
 print(funo)
 print("Reporte de clasificacion forest bag:")
 print()
 y_rfc_bag =bag_results.predict(x_test)
 print(confusion_matrix(y_test , y_rfc_bag))
 print(classification_report(y_test , y_rfc_bag ))
 print()
 acc_bag=accuracy_score(y_test , y_rfc_bag)
 print("accuaracy bag")
 print(acc_bag)
 prec_bag=precision_score(y_test , y_rfc_bag)
 print("precision bag")
 print(prec_bag)
 rec_bag=recall_score(y_test , y_rfc_bag)
 print("recall bag")
 print(rec_bag)
 funo_bag=f1_score(y_test , y_rfc_bag)
 micro_f1 = f1_score(y_test , y_rfc, average='micro')
 micro_f1_bag =  f1_score(y_test , y_rfc_bag, average='micro')
 print("f1 score micro")
 print(micro_f1)
 print("f1 score micro bag")
 print(micro_f1_bag)
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
 fpr_forest_bag, tpr_forest_bag, thresholds_forest_bag = roc_curve(y_test,
 y_rfc_bag, pos_label=1)
 auc_forest_bag = auc(fpr_forest_bag, tpr_forest_bag)
 MCC_forest_bag=matthews_corrcoef(y_test, y_rfc_bag)
 print("MCC forest BAG:")
 print(MCC_forest_bag)
 # if depth==100:
 micros.append(micro_f1)
 micros_bag.append(micro_f1_bag)
 recalls.append(rec)
 recalls_bag.append(rec_bag)
 MCC.append(MCC_forest)
 MCC_bag.append(MCC_forest_bag)
 AUCS.append(auc_forest)
 AUCS_bag.append(auc_forest_bag)

x_axis = [i for i in range(1,11)]

plt.plot(estimators, micros, marker='o', color='red',
         label='RF')
#plt.plot(mean_fpr_svm, mean_tpr_svm, color='green',
#         label='micro f1 (AUC = %0.2f )' % (mean_auc_svm))
#         lw=2, alpha=1)
plt.plot(estimators, micros_bag, marker='o', color='blue',
          label='Bagging Classifier with RF' )
plt.xlabel('Estimators',fontsize = 16)
plt.ylabel('Average Micro F1-Score',fontsize = 16)
#plt.title('Random Forest using Depth=100',fontsize = 16)
plt.legend(loc="right",fontsize = 16 ,ncol=1)
plt.show()

plt.plot(estimators, MCC, marker='o', color='red',
 label=' RF' )
plt.plot(estimators, MCC_bag, marker='o', color='blue',
          label='Bagging Classifier with RF' )
plt.xlabel('Estimators',fontsize = 16)
plt.ylabel('MCC Rate',fontsize = 16)
#plt.title('Random Forest using Depth=100',fontsize = 16)
plt.legend(loc="right",fontsize = 16 ,ncol=1)
plt.show()
