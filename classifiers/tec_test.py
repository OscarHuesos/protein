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

model = Sequential()
model.add(Dense(8, input_dim=6, activation='tanh'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer='adam',
metrics=['accuracy'])
model.summary()

def media(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t
    avg = sum_num / len(num)
    return avg

train_data=np.genfromtxt("Train_data_apart.csv", delimiter=',', skip_header=1)
test_data=np.genfromtxt("Test_data_apart.csv", delimiter=',', skip_header=1)
#print("data")
#print(data)
X_train=train_data[:,0:6]
Y_train=train_data[:,7]
print(X_train)
print(Y_train)
print(Y_train.astype)
y_train = label_binarize(Y_train, classes=[0, 1])
yy_train=np.ravel(y_train)
#nc = len(yy)
#print("y")
#print(yy)
X_test=test_data[:,0:6]
Y_test=test_data[:,7]
print(X_test)
print(Y_test)
print(Y_test.astype)
y_test = label_binarize(Y_test, classes=[0, 1])
yy_test=np.ravel(y_test)

X_train_redes = preprocessing.scale(X_train)
X_test_redes = preprocessing.scale(X_test)
#X_test = preprocessing.scale(x_test)
tprs_nn = []
aucs_nn = []
mean_fpr_nn = np.linspace(0,1,100)
# plot arrows
#fig1 = plt.figure(figsize=[12,12])
fig_nn = plt.figure(figsize=[12,12])
ax1 = fig_nn.add_subplot(111,aspect = 'auto')

scores_model_nn_train = []
scores_model_nn_test = []
scores_nn_acc=[]
scores_nn_prec_0=[]
scores_nn_prec_1=[]
scores_nn_rec_0=[]
scores_nn_rec_1=[]
scores_nn_f1_0=[]
scores_nn_f1_1=[]
scores_micros_f1_nn=[]
scores_nn_mcc=[]
cont=0

for i in range(1,11):
 #model.fit(X_train_redes , yy_train, epochs=300, batch_size=32, verbose=0)
 model.fit(X_train_redes , yy_train, epochs=500, batch_size=500, verbose=0)
 scores_nn_train=model.evaluate(X_train_redes , yy_train, verbose=0)
 scores_nn_test=model.evaluate(X_test_redes , yy_test, verbose=0)
 scores_model_nn_train.append(scores_nn_train)
 scores_model_nn_test.append(scores_nn_test)
 y_mlp = model.predict_classes(X_test_redes)
 y_mlp_proba = model.predict_proba(X_test_redes)
#y_test_pred_model = model.predict(x_train[test])
 acc_mlp = accuracy_score(yy_test , y_mlp)
 print("accuaracy NN")
 print(acc_mlp)
 scores_nn_acc.append(acc_mlp)
 prec_mlp=precision_score(yy_test , y_mlp, average=None)
 print("precision NN")
 print(prec_mlp)
 scores_nn_prec_0.append(prec_mlp[0])
 scores_nn_prec_1.append(prec_mlp[1])
 rec_mlp= recall_score(yy_test , y_mlp, average=None)
 print("recall NN")
 print(rec_mlp)
 scores_nn_rec_0.append(rec_mlp[0])
 scores_nn_rec_1.append(rec_mlp[1])
 funo_mlp = f1_score(yy_test , y_mlp, average=None)
 print("f1 score NN")
 print(funo_mlp)
 scores_nn_f1_0.append(funo_mlp[0])
 scores_nn_f1_1.append(funo_mlp[1])
 fpr_mlp, tpr_mlp, threshold = roc_curve(yy_test , y_mlp_proba)
 print("fpr NN")
 print(fpr_mlp)
 print("tpr NN")
 print(tpr_mlp)
 print("Reporte de clasificacion NN")
 print()
 # y_rfc =rfc_result.predict(x_test)
 print(confusion_matrix(yy_test , y_mlp))
 print(classification_report(yy_test , y_mlp ))
 micro_f1_NN = f1_score(yy_test , y_mlp, average='micro')
 print("f1 score micro")
 print(micro_f1_NN)
 scores_micros_f1_nn.append(micro_f1_NN)
 auc_NN = auc(fpr_mlp, tpr_mlp)
 MCC_NN=matthews_corrcoef(yy_test, y_mlp)
 print("MCC NN:")
 print(MCC_NN)
 scores_nn_mcc.append(MCC_NN)
 tprs_nn.append(interp(mean_fpr_nn , fpr_mlp, tpr_mlp))
 roc_auc_nn = auc(fpr_mlp, tpr_mlp)
 aucs_nn.append(roc_auc_nn)
 plt.plot(fpr_mlp, tpr_mlp, lw=2, alpha=0.3,
 label='ROC fold %d (AUC = %0.2f)' % (cont, roc_auc_nn))
 #nn_plot= plot_roc_curve(log, X[test], Y[test] ,ax=ax1,
 #alpha=0.3,label='Fold %d (AUC = %0.2f)' % (cont, roc_auc_log))
 print("cont:")
 print(cont)
 cont=cont+1

plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
mean_tpr_nn = np.mean(tprs_nn, axis=0)
print("mean trp nn")
print(mean_tpr_nn)
print("mean fpr nn")
print(mean_fpr_nn)
mean_auc_nn = auc(mean_fpr_nn, mean_tpr_nn)
plt.plot(mean_fpr_nn, mean_tpr_nn, color='blue',
         label='Mean Neuronal Network (AUC = %0.2f )' % (mean_auc_nn),
         lw=2, alpha=1)
plt.xlabel('False Positive Rate',fontsize = 10)
plt.ylabel('True Positive Rate',fontsize = 10)
plt.title('Neuronal Network ROC curve',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=2)
#plt.text(0.32,0.7,'More accurate area',fontsize = 12)
#plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()

##########################################################
tprs_rfc = []
aucs_rfc = []

true_pos= []
true_neg= []
false_pos= []
false_neg=[]

rate_true_pos= []
rate_true_neg= []
rate_false_pos= []
rate_false_neg=[]

mean_fpr_rfc = np.linspace(0,1,100)
fig_rfc = plt.figure(figsize=[12,12])
ax2 = fig_rfc.add_subplot(111,aspect = 'auto')

scores_rfc= []
#scores_rfc_train = []
#scores_rfc_test = []
scores_rfc_acc=[]
scores_rfc_prec_0=[]
scores_rfc_prec_1=[]
scores_rfc_rec_0=[]
scores_rfc_rec_1=[]
scores_rfc_f1_0=[]
scores_rfc_f1_1=[]
scores_micros_f1_rfc=[]
scores_rfc_mcc=[]
cont=0
rfc=RandomForestClassifier(n_estimators=1000,
criterion='entropy', max_depth=100 , oob_score=True)

for i in range(1,11):
 rfc_result=rfc.fit(X_train , yy_train)
 #scores_rfc_train=rfc_result.evaluate(X_train, yy_train, verbose=0)
 #scores_rfc_test=rfc_result.evaluate(X_test, yy_test, verbose=0)
 #scores_rfc_train.append(scores_rfc_train)
 #scores_rfc_test.append(scores_rfc_test)
 y_rfc =  rfc.predict(X_test)
 y_rfc_proba = rfc.predict_proba(X_test)
 acc_rfc=accuracy_score(yy_test , y_rfc)
 print("accuaracy rfc")
 print(acc_rfc)
 scores_rfc_acc.append(acc_rfc)
 prec_rfc=precision_score(yy_test , y_rfc, average=None)
 print("precision rfc")
 print(prec_rfc)
 scores_rfc_prec_0.append(prec_rfc[0])
 scores_rfc_prec_1.append(prec_rfc[1])
 rec_rfc=recall_score(yy_test , y_rfc, average=None)
 print("recall rfc")
 print(rec_rfc)
 scores_rfc_rec_0.append(rec_rfc[0])
 scores_rfc_rec_1.append(rec_rfc[1])
 funo_rfc=f1_score(yy_test , y_rfc, average=None)
 print("f1 score rfc")
 print(funo_rfc)
 scores_rfc_f1_0.append(funo_rfc[0])
 scores_rfc_f1_1.append(funo_rfc[1])
 fpr_rfc, tpr_rfc, threshold = roc_curve(yy_test , y_rfc_proba[:, 1])
 print("fpr rfc")
 print(fpr_rfc)
 print("tpr rfc")
 print(tpr_rfc)
 print("Reporte de clasifocacion RF")
 print()
 # y_rfc =rfc_result.predict(x_test)
 print(confusion_matrix(yy_test , y_rfc))
 TN, FP, FN, TP = confusion_matrix(yy_test , y_rfc).ravel()
 print("size test")
 print(yy_test.size)
 print("true pos")
 print(TP)
 print("true neg")
 print(TN)
 print("false positive")
 print(FP)
 print("false neg")
 print(FN)
 true_pos.append(TP)
 true_neg.append(TN)
 false_pos.append(FP)
 false_neg.append(FN)
 rate_TP=(TP/yy_test.size)*100
 rate_TN=(TN/yy_test.size)*100
 rate_FP=(FP/yy_test.size)*100
 rate_FN=(FN/yy_test.size)*100
 print("rate true pos")
 print(rate_TP)
 print("rate true neg")
 print(rate_TN)
 print("rate false positive")
 print(rate_FP)
 print("rate false neg")
 print(rate_FN)
 rate_true_pos.append(rate_TP)
 rate_true_neg.append(rate_TN)
 rate_false_pos.append(rate_FP)
 rate_false_neg.append(rate_FN)


 print(classification_report(yy_test , y_rfc ))
 micro_f1_rfc = f1_score(yy_test , y_rfc, average='micro')
 print("f1 score micro rfc")
 print(micro_f1_rfc)
 scores_micros_f1_rfc.append(micro_f1_rfc)
 auc_forest = auc(fpr_rfc, tpr_rfc)
 MCC_forest = matthews_corrcoef(yy_test, y_rfc)
 print("MCC forest:")
 print(MCC_forest)
 scores_rfc_mcc.append(MCC_forest)
 tprs_rfc.append(interp(mean_fpr_rfc , fpr_rfc, tpr_rfc))
 roc_auc_rfc = auc(fpr_rfc, tpr_rfc)
 aucs_rfc.append(roc_auc_rfc)
 plt.plot(fpr_rfc, tpr_rfc, lw=2, alpha=0.3,
 label='ROC fold %d (AUC = %0.2f)' % (cont, roc_auc_rfc))
 print("cont:")
 print(cont)
 cont=cont+1

plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
mean_tpr_rfc = np.mean(tprs_rfc, axis=0)
print("mean trp RF")
print(mean_tpr_rfc)
print("mean fpr RF")
print(mean_fpr_rfc)
mean_auc_rfc = auc(mean_fpr_rfc, mean_tpr_rfc)
plt.plot(mean_fpr_rfc, mean_tpr_rfc, color='blue',
        label='Mean Random Forest (AUC = %0.2f )' % (mean_auc_rfc),
        lw=2, alpha=1)
plt.xlabel('False Positive Rate',fontsize = 10)
plt.ylabel('True Positive Rate',fontsize = 10)
plt.title('Random Forest ROC curve',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=2)
#plt.text(0.32,0.7,'More accurate area',fontsize = 12)
#plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()
#nn_plot= plot_roc_curve(log, X[test], Y[test] ,ax=ax1,
#alpha=0.3,label='Fold %d (AUC = %0.2f)' % (cont, roc_auc_log))

#####################################################################
tprs_svm = []
aucs_svm = []
mean_fpr_svm = np.linspace(0,1,100)

fig_svm = plt.figure(figsize=[12,12])
ax3 = fig_svm.add_subplot(111,aspect = 'auto')

scores_svm= []
#scores_svm_train = []
#scores_svm_test = []
scores_svm_acc=[]
scores_svm_prec_0=[]
scores_svm_prec_1=[]
scores_svm_rec_0=[]
scores_svm_rec_1=[]
scores_svm_f1_0=[]
scores_svm_f1_1=[]
scores_micros_f1_svm=[]
scores_svm_mcc=[]
cont=0

svm=SVC(kernel='rbf', gamma=0.01, C=10,
coef0=0, degree=1, probability=True)

for i in range(1,11):
 svm_result=svm.fit(X_train, yy_train)
 #scores_svm_train=svm_result.evaluate(X_train, yy_train, verbose=0)
 #scores_svm_test=svm_result.evaluate(X_test, yy_test, verbose=0)
 #scores_svm_train.append(scores_svm_train)
 #scores_svm_test.append(scores_svm_test)
 y_svm = svm.predict(X_test)
 y_svm_proba = svm.predict_proba(X_test)
 acc_svm=accuracy_score(yy_test , y_svm)
 print("accuaracy SVM")
 print(acc_svm)
 scores_svm_acc.append(acc_svm)
 prec_svm=precision_score(yy_test , y_svm, average=None)
 print("precision SVM")
 print(prec_svm)
 scores_svm_prec_0.append(prec_svm[0])
 scores_svm_prec_1.append(prec_svm[1])
 rec_svm=recall_score(yy_test , y_svm, average=None)
 print("recall svm")
 print(rec_svm)
 scores_svm_rec_0.append(rec_svm[0])
 scores_svm_rec_1.append(rec_svm[1])
 funo_svm=f1_score(yy_test , y_svm, average=None)
 print("f1 score SVM")
 print(funo_svm)
 scores_svm_f1_0.append(funo_svm[0])
 scores_svm_f1_1.append(funo_svm[1])
 fpr_svm, tpr_svm, threshold = roc_curve(yy_test , y_svm_proba[:, 1])
 print("fpr svm")
 print(fpr_svm)
 print("tpr svm")
 print(tpr_svm)
 print("Reporte de clasifocacion SVM")
 print()
 print(classification_report(yy_test , y_svm ))
 print(confusion_matrix(yy_test , y_svm))
 print()
 micro_f1_svm = f1_score(yy_test , y_svm, average='micro')
 print("f1 score micro SVM")
 print(micro_f1_svm)
 scores_micros_f1_svm.append(micro_f1_svm)
 auc_svm = auc(fpr_svm, tpr_svm)
 MCC_svm = matthews_corrcoef(yy_test, y_svm)
 print("MCC SVM:")
 print(MCC_svm)
 scores_svm_mcc.append(MCC_svm)
 tprs_svm.append(interp(mean_fpr_svm , fpr_svm, tpr_svm))
 roc_auc_svm = auc(fpr_svm, tpr_svm)
 aucs_svm.append(roc_auc_svm)
 #svm_plot= plot_roc_curve(svm, X[test], Y[test] ,ax=ax4,
 #alpha=0.3,label='Fold %d (AUC = %0.2f)' % (cont, roc_auc_svm))
 plt.plot(fpr_svm, tpr_svm, lw=2, alpha=0.3,
 label='ROC fold %d (AUC = %0.2f)' % (cont, roc_auc_svm))
 print("cont:")
 print(cont)
 cont=cont+1

plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
mean_tpr_svm = np.mean(tprs_svm, axis=0)
print("mean trp svm")
print(mean_tpr_svm)
print("mean fpr svm")
print(mean_fpr_svm)
mean_auc_svm = auc(mean_fpr_svm, mean_tpr_svm)
plt.plot(mean_fpr_svm, mean_tpr_svm, color='blue',
         label='Mean Support Vector Machine (AUC = %0.2f )' % (mean_auc_svm),
         lw=2, alpha=1)
plt.xlabel('False Positive Rate',fontsize = 10)
plt.ylabel('True Positive Rate',fontsize = 10)
plt.title('Support Vector Machine ROC curve',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=2)
#plt.text(0.32,0.7,'More accurate area',fontsize = 12)
#plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()
#y_mlp_proba = model.predict_proba(X_train_redes)
  # micros_bag.append(micro_f1_bag)
  #OOB=rfc_result.oob_score_
  #print("OOB error:")
 # print(OOB)
 # IMP=rfc_result.feature_importances_.argsort()
  #print("Mejores caracteristicas")
  #print(IMP)
 # fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_test, y_rfc,pos_label=1)

fig_total = plt.figure(figsize=[12,12])
ax4 = fig_total.add_subplot(111,aspect = 'auto')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.plot(mean_fpr_nn, mean_tpr_nn, color='blue',
         label='Mean Neuronal Network (AUC = %0.2f )' % (mean_auc_nn),
         lw=2, alpha=1)
plt.plot(mean_fpr_rfc, mean_tpr_rfc, color='red',
         label='Mean Random Forest (AUC = %0.2f )' % (mean_auc_rfc),
         lw=2, alpha=1)
plt.plot(mean_fpr_svm, mean_tpr_svm, color='green',
         label='Mean Support Vector Machine (AUC = %0.2f )' % (mean_auc_svm),
         lw=2, alpha=1)
plt.xlabel('False Positive Rate',fontsize = 15)
plt.ylabel('True Positive Rate',fontsize = 15)
#plt.title('Mean of ROC curve (10 times using all data (70% train and 30% test))'
#,fontsize = 18)
plt.legend(loc="lower right",fontsize = 15, ncol=1)
#plt.text(0.32,0.7,'More accurate area',fontsize = 12)
#plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()
#plt.plot(fpr_svm, tpr_svm, lw=2, alpha=0.3,
#label='ROC fold %d (AUC = %0.2f)' % (cont, roc_auc_svm))

print("Reporte de clasificacion NN:")
print()
print("scores model NN ")
print('accuracy NN:')
print(scores_nn_acc)
print('NN train eval:')
print(scores_model_nn_train)
print('NN test eval:')
print(scores_model_nn_test)
print('precision 0 NN:')
print(scores_nn_prec_0)
print('precision 1 NN:')
print(scores_nn_prec_1)
print('recall 0 NN:')
print(scores_nn_rec_0)
print('recall_1 NN:')
print(scores_nn_rec_1)
print('f1_0 NN:')
print(scores_nn_f1_0)
print('f1_1 NN:')
print(scores_nn_f1_1)
print('micro f1:')
print(scores_micros_f1_nn)
print('NN MCC:')
print(scores_nn_mcc)

print("Means:")
print('accuracy:', media(scores_nn_acc))
#print('NN train eval:', media(scores_model_nn_train))
#print('NN test eval:', media(scores_model_nn_test))
print('precision 0:', media(scores_nn_prec_0))
print('precision 1:', media(scores_nn_prec_1))
print('recall 0:', media(scores_nn_rec_0))
print('recall 1:', media(scores_nn_rec_1))
print('f1 0:', media(scores_nn_f1_0))
print('f1 1:', media(scores_nn_f1_1))
print('micro f1:', media(scores_micros_f1_nn))
print('NN MCC:', media(scores_nn_mcc))
#print('recall:', media(scores_nn_rec) )
#print('f1:', media(scores_nn)   )
print("Std Dev:")
print('accuracy:',  stdev(scores_nn_acc))
#print('NN train eval:', stdev(scores_model_nn_train))
#print('NN test eval:', stdev(scores_model_nn_test))
print('precision 0:', stdev(scores_nn_prec_0))
print('precision 1:', stdev(scores_nn_prec_1))
print('recall 0:', stdev(scores_nn_rec_0))
print('recall 1:', stdev(scores_nn_rec_1))
print('f1 0:', stdev(scores_nn_f1_0))
print('f1 1:', stdev(scores_nn_f1_1))
print('micro f1:', stdev(scores_micros_f1_nn))
print('NN MCC:', stdev(scores_nn_mcc))

print("Reporte de clasificacion RF:")
print()
print("scores model RF ")
print('accuracy RF:')
print(scores_rfc_acc)
print('precision 0 RF:')
print(scores_rfc_prec_0)
print('precision 1 RF:')
print(scores_rfc_prec_1)
print('recall 0 RF:')
print(scores_rfc_rec_0)
print('recall 1 RF:')
print(scores_rfc_rec_1)
print('F1 0 RF:')
print(scores_rfc_f1_0)
print('F1 1 RF:')
print(scores_rfc_f1_1)
print('micro f1 RF:')
print(scores_micros_f1_rfc)
print('MCC RF')
print(scores_rfc_mcc)

print("Means RF:")
print('accuracy:', media(scores_rfc_acc))
print('precision 0:', media(scores_rfc_prec_0))
print('precision 1:', media(scores_rfc_prec_1))
print('recall 0:', media(scores_rfc_rec_0))
print('recall 1:', media(scores_rfc_rec_1))
print('f1 0:', media(scores_rfc_f1_0))
print('f1 1:', media(scores_rfc_f1_1))
print('micro f1 RF:', media(scores_micros_f1_rfc))
print('RF MCC:', media(scores_rfc_mcc))

print("Std Dev:")
print('accuracy:',  stdev(scores_rfc_acc))
print('precision 0:', stdev(scores_rfc_prec_0))
print('precision 1:', stdev(scores_rfc_prec_1))
print('recall 0:', stdev(scores_rfc_rec_0))
print('recall 1:', stdev(scores_rfc_rec_1))
print('f1 0:', stdev(scores_rfc_f1_0))
print('f1 1:', stdev(scores_rfc_f1_1))
print('micro f1 RF:', stdev(scores_micros_f1_rfc))
print('RF MCC:', stdev(scores_rfc_mcc))

TP_med=media(rate_true_pos)
TN_med=media(rate_true_neg)
FP_med=media(rate_false_pos)
FN_med=media(rate_false_neg)

plt.figure()
array=np.array([[TN_med, FP_med], [FN_med, TP_med]])
#group_names = ['TN','FP','FN','TP']
df_cm = pd.DataFrame(array, range(2), range(2))
text = np.asarray([['TN%', 'FP%'], ['FN%', 'TP%']])


labels = (np.asarray(["{0}\n{1:.2f}".format(text, array) for text, array in zip(text.flatten(), array.flatten())])).reshape(2,2)
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=labels, annot_kws={"size": 16},fmt='',cbar=False, cmap='Blues') # font size
#plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
ax= plt.subplot()
ax.set_yticks([0, 1])
ax.tick_params(labelsize=10)
ax.xaxis.set_ticklabels(['Non-hot spot/non-hot region','Hot spot/hot region'])
ax.yaxis.set_ticklabels(['Non-hot spot/non-hot region','Hot spot/hot region'])
#ax.xaxis.set_ticklabels(['Non-hot spot/non-hot region','Hot spot/hot region'])
#ax.yaxis.set_ticklabels(['Non-hot spot/non-hot region','Hot spot/hot region'])
ax.yaxis.set_ticks_position('both')
plt.show()


print("Reporte de clasificacion SVM:")
print()
print("scores model SVM ")
print('accuracy SVM:')
print(scores_svm_acc)
print('precision 0 SVM:')
print(scores_svm_prec_0)
print('precision 1 SVM:')
print(scores_svm_prec_1)
print('recall 0 SVM:')
print(scores_svm_rec_0)
print('recall 1 SVM:')
print(scores_svm_rec_1)
print('F1 0 SVM:')
print(scores_svm_f1_0)
print('F1 1 SVM:')
print(scores_svm_f1_1)
print('micro f1 SVM:')
print(scores_micros_f1_svm)
print('MCC SVM')
print(scores_svm_mcc)

print("Means SVM:")
print('accuracy:', media(scores_svm_acc))
print('precision 0:', media(scores_svm_prec_0))
print('precision 1:', media(scores_svm_prec_1))
print('recall 0:', media(scores_svm_rec_0))
print('recall 1:', media(scores_svm_rec_1))
print('f1 0:', media(scores_svm_f1_0))
print('f1 1:', media(scores_svm_f1_1))
print('micro f1 SVM:', media(scores_micros_f1_svm))
print('SVM MCC:', media(scores_svm_mcc))

print("Std Dev SVM:")
print('accuracy:',  stdev(scores_svm_acc))
print('precision 0:', stdev(scores_svm_prec_0))
print('precision 1:', stdev(scores_svm_prec_1))
print('recall 0:', stdev(scores_svm_rec_0))
print('recall 1:', stdev(scores_svm_rec_1))
print('f1 0:', stdev(scores_svm_f1_0))
print('f1 1:', stdev(scores_svm_f1_1))
print('micro f1 SVM:', stdev(scores_micros_f1_svm))
print('SVM MCC:', stdev(scores_svm_mcc))
