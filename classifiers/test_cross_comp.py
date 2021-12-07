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

def media(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t
    avg = sum_num / len(num)
    return avg

#Ejecutar programa 10 veces
data=np.genfromtxt("interface_data_sola.csv", delimiter=',', skip_header=1)
print("data")
print(data)
#scoring = {'accuracy' : make_scorer(accuracy_score),
#           'precision' : make_scorer(precision_score),
#           'recall' : make_scorer(recall_score),
#           'f1_score' : make_scorer(f1_score)}
#cv= KFold(n_folds=1)
#cv.get_n_splits(X)
score_funcs = ['accuracy','precision','recall','f1']
#x_train, x_test, y_train, y_test = train_test_split(X ,yy,
#test_size=0.30,random_state=42)
Train, Test = train_test_split(data,
test_size=0.30, random_state=42)
cv = RepeatedKFold(n_splits=10, random_state=42)
#cv = KFold(n_splits=10, random_state=42)
#cv = StratifiedKFold(n_splits=5,shuffle=False)
print(Train)
Train_pandas = pd.DataFrame(Train,
        #         index=['day1', 'day2', 'day3', 'day4'],
                 columns=['BfactorProt','area','hidrofobicidad',
                 	'indice_prevalencia','conservation_score',
                    'energia_dft','interface','hotspot'])
print(Test)
Test_pandas = pd.DataFrame(Test,
        #         index=['day1', 'day2', 'day3', 'day4'],
                 columns=['BfactorProt','area','hidrofobicidad',
                 	'indice_prevalencia','conservation_score',
                    'energia_dft','interface','hotspot'])

Test_pandas.to_csv('Test_data_apart.csv',index=False)
Train_pandas.to_csv('Train_data_apart.csv',index=False)



X=Train[:,0:6]
Y=Train[:,7]
print(X)
print(Y)
print(Y.astype)
y = label_binarize(Y, classes=[0, 1])
yy=np.ravel(y)
nc = len(yy)
print("y")
print(yy)
print("cv")
print(cv)

log=LogisticRegression(random_state=42,C=100,penalty='l2',solver='liblinear',tol=0.01)
#for train_index, test_index in cv.split(X):
#scores_log = cross_validate(log, x_train, y_train, scoring=score_funcs, cv=cv)
#log_result=log.fit(x_train,y_train)
#print("Reporte de clasificacion log directo:")
#print(scores_log )
#print('mean')
#print('accuaracy: ', scores_log['test_accuracy'].mean())
#print('precision: ', scores_log['test_precision'].mean())
#print('recall: ', scores_log['test_recall'].mean())
#print('f1: ', scores_log['test_f1'].mean())
#print('std')
#print('accuaracy: ', scores_log['test_accuracy'].std())
#print('precision: ', scores_log['test_precision'].std())
#print('recall: ', scores_log['test_recall'].std())
#print('f1: ' ,scores_log['test_f1'].std())
#print('recall:', scores_log.mean() )
#print('f1:', media(scores_nn)   )
#mean
#accuaracy:  0.7624804404520428
#precision:  0.6375977471196541
#recall:  0.6283840684224183
#f1:  0.6310604079363624
#std
#accuaracy:  0.028962879994800905
#precision:  0.0596125778819534
#recall:  0.05855323938681276
#f1:  0.04809879668584937
tprs_log = []
aucs_log = []
mean_fpr_log = np.linspace(0,1,100)
# plot arrows
#fig1 = plt.figure(figsize=[12,12])
fig_log = plt.figure(figsize=[12,12])
ax1 = fig_log.add_subplot(111,aspect = 'auto')
scores_log2= []
scores_log_acc=[]
scores_log_prec_0=[]
scores_log_prec_1=[]
scores_log_rec_0=[]
scores_log_rec_1=[]
scores_log_f1_0=[]
scores_log_f1_1=[]
cont=0

#ax1 = fig_log.add_subplot(111,aspect = 'equal')
#ax1.add_patch(patches.Arrow(0.45,0.5,-0.25,0.25,width=0.3,color='green',alpha = 0.5))
#ax1.add_patch(patches.Arrow(0.5,0.45,0.25,-0.25,width=0.3,color='red',alpha = 0.5))
#for train, test in cv.split(x_train, y_train):
for train, test in cv.split(X,Y):
 log.fit(X[train],Y[train])
 #scores_log2=m.evaluate(x_train[test],y_train[test], verbose=0)
 #scores_model_nn.append(scores_nn)
 y_log = log.predict(X[test])
 y_log_proba = log.predict_proba(X[test])
 #y_test_pred_model = model.predict(x_train[test])
 acc=accuracy_score(Y[test] , y_log)
 scores_log_acc.append(acc)
 prec=precision_score(Y[test] , y_log, average=None)
 scores_log_prec_0.append(prec[0])
 scores_log_prec_1.append(prec[1])
 rec=recall_score(Y[test] , y_log, average=None)
 scores_log_rec_0.append(rec[0])
 scores_log_rec_1.append(rec[1])
 funo=f1_score(Y[test] , y_log, average=None)
 scores_log_f1_0.append(funo[0])
 scores_log_f1_1.append(funo[1])
 #print(classification_report(y_train[test] , y_log))
 print("cont:")
 print(cont)
 #print("y log")
 #print(y_log)
 #print("y trian")
 #print(y_train[test])
 fpr, tpr, threshold = roc_curve(Y[test] , y_log_proba[:, 1])
 print("fpr")
 print(fpr)
 print("tpr")
 print(tpr)
 #prediction = clf.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
 #fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
 tprs_log.append(interp(mean_fpr_log , fpr, tpr))
 roc_auc_log = auc(fpr, tpr)
 MCC_log=matthews_corrcoef(Y[test] , y_log)
 print("MCC log:")
 print(MCC_log)
 aucs_log.append(roc_auc_log)
 log_plot= plot_roc_curve(log, X[test], Y[test] ,ax=ax1,
 alpha=0.3,label='Fold %d (AUC = %0.2f)' % (cont, roc_auc_log))
 #print("log plot")
 #print(log_plot)
 #plt.plot(fpr, tpr, alpha=0.3,
 #label='ROC fold %d (AUC = %0.2f)' % (cont, roc_auc_log))
 cont=cont+1


print("Reporte de clasificacion log:")
print()
print("scores log")
print('accuracy:')
print(scores_log_acc)
print('precision 0:')
print(scores_log_prec_0)
print('precision 1:')
print(scores_log_prec_1)
print('recall 0:')
print(scores_log_rec_0)
print('recall_1:')
print(scores_log_rec_1)
print('f1_0:')
print(scores_log_f1_0)
print('f1_1:')
print(scores_log_f1_1)

print("means:")
print('accuracy:', media(scores_log_acc))
print('precision 0:', media(scores_log_prec_0))
print('precision 1:', media(scores_log_prec_1))
print('recall 0:', media(scores_log_rec_0))
print('recall 1:', media(scores_log_rec_1))
print('f1 0:', media(scores_log_f1_0))
print('f1 1:', media(scores_log_f1_1))
#print('recall:', media(scores_nn_rec) )
#print('f1:', media(scores_nn)   )
print("std dev:")
print('accuracy:',  stdev(scores_log_acc))
print('precision 0:', stdev(scores_log_prec_0))
print('precision 1:', stdev(scores_log_prec_1))
print('recall 0:', stdev(scores_log_rec_0))
print('recall 1:', stdev(scores_log_rec_1))
print('f1 0:', stdev(scores_log_f1_0))
print('f1 1:', stdev(scores_log_f1_1))

#y_log_cross=cross_val_predict(log, X, y, scoring='accuracy', cv=cv)
#y_log =log_result.predict_proba(x_test)
#y_log2 =log_result.predict(x_test)
#yy=y_log2.flatten()
#fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_log2,pos_label=1)
#auc_log = auc(fpr_log, tpr_log)
#MCC_log=matthews_corrcoef(y_test, y_log2)
#print("MCC log:")
#print(MCC_log)
#pyplot.figure()
#print(fpr_log)
#print(y_log2)
#print(yy)
#print(confusion_matrix(y_test , y_log2))
#print(classification_report(y_test , y_log2 ))
#LogisticRegression
#means:
#accuracy: 0.7624804404520428
#precision: 0.6375977471196541
#recall: 0.6283840684224183
#f1: 0.6310604079363624
#
#accuracy: 0.7586001642036124
#precision: 0.6329170008115425
#recall: 0.5675983860001151
#f1: 0.5962495105188933
#means:
#accuracy: 0.7514022505553948
#precision: 0.6175190063432541
#recall: 0.5790343780373711
#f1: 0.5954407218201947

#accuracy: 0.7624804404520428
#precision 0: 0.8219075863441693
#precision 1: 0.6375977471196542
#recall 0: 0.8272341931555718
#recall 1: 0.6283840684224183
#f1 0: 0.8239995611941938
#f1 1: 0.6310604079363624
#std dev:
#accuracy: 0.02910878963358944
#precision 0: 0.03163921575227002
#precision 1: 0.05991289503644807
#recall 0: 0.03296022413250202
#recall 1: 0.058848219789671946
#f1 0: 0.02418557847551069
#f1 1: 0.048341109537743086

#means:
#accuracy: 0.7624804404520428
#precision 0: 0.8219075863441693
#precision 1: 0.6375977471196542
#recall 0: 0.8272341931555718
#recall 1: 0.6283840684224183
#f1 0: 0.8239995611941938
#f1 1: 0.6310604079363624
#std dev:
#accuracy: 0.02910878963358944
#precision 0: 0.03163921575227002
#precision 1: 0.05991289503644807
#recall 0: 0.03296022413250202
#recall 1: 0.058848219789671946
#f1 0: 0.02418557847551069
#f1 1: 0.048341109537743086

plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
mean_tpr_log = np.mean(tprs_log, axis=0)
print("mean trp log")
print(mean_tpr_log)
print("mean fpr log")
print(mean_fpr_log)
mean_auc_log = auc(mean_fpr_log, mean_tpr_log)
plt.plot(mean_fpr_log, mean_tpr_log, color='blue',
         label='Mean Log (AUC = %0.2f )' % (mean_auc_log),
         lw=2, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Logistic')
plt.legend(loc="lower right",fontsize = 5,ncol=5)
#plt.text(0.32,0.7,'More accurate area',fontsize = 12)
#plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()

############################################################
X_train = preprocessing.scale(X)
#X_test = preprocessing.scale(x_test)
tprs_nn = []
aucs_nn = []
mean_fpr_nn = np.linspace(0,1,100)
# plot arrows
#fig1 = plt.figure(figsize=[12,12])
fig_nn = plt.figure(figsize=[12,12])
ax2 = fig_nn.add_subplot(111,aspect = 'auto')
scores_model= []
scores_model_nn= []
scores_nn_acc=[]
scores_nn_prec_0=[]
scores_nn_prec_1=[]
scores_nn_rec_0=[]
scores_nn_rec_1=[]
scores_nn_f1_0=[]
scores_nn_f1_1=[]
cont=0

model = Sequential()
model.add(Dense(8, input_dim=6, activation='tanh'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

for train, test in cv.split(X_train, Y):
 #model.save('model.h5')
 model.fit(X_train[train],Y[train], epochs=500, batch_size=500, verbose=0)
 scores_nn=model.evaluate(X_train[test], Y[test], verbose=0)
 scores_model_nn.append(scores_nn)
 y_mlp = model.predict_classes(X_train[test])
 y_mlp_proba = model.predict_proba(X_train[test])
 #y_test_pred_model = model.predict(x_train[test])
 acc=accuracy_score(Y[test] , y_mlp)
 scores_nn_acc.append(acc)
 prec=precision_score(Y[test] , y_mlp, average=None)
 scores_nn_prec_0.append(prec[0])
 scores_nn_prec_1.append(prec[1])
 rec=recall_score(Y[test] , y_mlp, average=None)
 scores_nn_rec_0.append(rec[0])
 scores_nn_rec_1.append(rec[1])
 funo=f1_score(Y[test] , y_mlp, average=None)
 scores_nn_f1_0.append(funo[0])
 scores_nn_f1_1.append(funo[1])
 fpr, tpr, threshold = roc_curve(Y[test] , y_mlp_proba)
 print("fpr")
 print(fpr)
 print("tpr")
 print(tpr)
 #print(classification_report(y_train[test] , y_mlp))
 print("cont:")
 print(cont)
 tprs_nn.append(interp(mean_fpr_nn , fpr, tpr))
 roc_auc_nn = auc(fpr, tpr)
 MCC_nn=matthews_corrcoef(Y[test] , y_mlp)
 print("MCC nn:")
 print(MCC_nn)
 aucs_nn.append(roc_auc_nn)
 plt.plot(fpr, tpr, lw=2, alpha=0.3,
 label='ROC fold %d (AUC = %0.2f)' % (cont, roc_auc_nn))
 #nn_plot= plot_roc_curve(model, X_train[test],y_train[test] ,ax=ax2,
 #alpha=0.3,label='Fold %d (AUC = %0.2f)' % (cont, roc_auc_nn))
 cont=cont+1

print("Reporte de clasificacion keras:")
print()
print("scores model")
print('accuracy:')
print(scores_nn_acc)
print('precision 0:')
print(scores_nn_prec_0)
print('precision 1:')
print(scores_nn_prec_1)
print('recall 0:')
print(scores_nn_rec_0)
print('recall_1:')
print(scores_nn_rec_1)
print('f1_0:')
print(scores_nn_f1_0)
print('f1_1:')
print(scores_nn_f1_1)

print("means:")
print('accuracy:', media(scores_nn_acc))
print('precision 0:', media(scores_nn_prec_0))
print('precision 1:', media(scores_nn_prec_1))
print('recall 0:', media(scores_nn_rec_0))
print('recall 1:', media(scores_nn_rec_1))
print('f1 0:', media(scores_nn_f1_0))
print('f1 1:', media(scores_nn_f1_1))
#print('recall:', media(scores_nn_rec) )
#print('f1:', media(scores_nn)   )
print("std dev:")
print('accuracy:',  stdev(scores_nn_acc))
print('precision 0:', stdev(scores_nn_prec_0))
print('precision 1:', stdev(scores_nn_prec_1))
print('recall 0:', stdev(scores_nn_rec_0))
print('recall 1:', stdev(scores_nn_rec_1))
print('f1 0:', stdev(scores_nn_f1_0))
print('f1 1:', stdev(scores_nn_f1_1))
#means:
#accuracy: 0.8311887858591714
#precision 0: 0.8945322747773189
#precision 1: 0.7199214635132963
#recall 0: 0.8506877560049259
#recall 1: 0.7916630112689985
#means:
#accuracy: 0.8675794455713316
#precision 0: 0.9257507070806308
#precision 1: 0.7670252233287962
#recall 0: 0.8737069091135132
#recall 1: 0.8545186668652014
#f1 0: 0.8985199866708442
#f1 1: 0.807000680238174
#std dev:
#accuracy: 0.024864134988772053
#precision 0: 0.024337356104067034
#precision 1: 0.04939489110079366
#recall 0: 0.032725888948380515
#recall 1: 0.04741751552269759
#f1 0: 0.021173750906913244
#f1 1: 0.035531614483298966

plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
mean_tpr_nn = np.mean(tprs_nn, axis=0)
print("mean trp nn")
print(mean_tpr_nn)
print("mean fpr nn")
print(mean_fpr_nn)
mean_auc_nn = auc(mean_fpr_nn, mean_tpr_nn)
plt.plot(mean_fpr_nn, mean_tpr_nn, color='blue',
         label='Mean NN (AUC = %0.2f )' % (mean_auc_nn),
         lw=2, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Neuronal Network')
plt.legend(loc="lower right",fontsize = 5,ncol=5)
#plt.text(0.32,0.7,'More accurate area',fontsize = 12)
#plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()

##################################################################
tprs_rf = []
aucs_rf = []
mean_fpr_rf = np.linspace(0,1,100)
# plot arrows
#fig1 = plt.figure(figsize=[12,12])
fig_rf = plt.figure(figsize=[12,12])
ax3 = fig_rf.add_subplot(111,aspect = 'auto')

scores_rfc= []
scores_rfc_model= []
scores_rfc_acc=[]
scores_rfc_prec_0=[]
scores_rfc_prec_1=[]
scores_rfc_rec_0=[]
scores_rfc_rec_1=[]
scores_rfc_f1_0=[]
scores_rfc_f1_1=[]
cont=0
rfc=RandomForestClassifier(n_estimators=1000,criterion='entropy',oob_score=True)

for train, test in cv.split(X,Y):
 rfc.fit(X[train],Y[train])
 y_rfc = rfc.predict(X[test])
 y_rfc_proba = rfc.predict_proba(X[test])
 #y_test_pred_model = model.predict(x_train[test])
 #scores_rfc_model.append(scores_nn)
 acc=accuracy_score(Y[test] , y_rfc)
 scores_rfc_acc.append(acc)
 prec=precision_score(Y[test] , y_rfc, average=None)
 scores_rfc_prec_0.append(prec[0])
 scores_rfc_prec_1.append(prec[1])
 rec=recall_score(Y[test] , y_rfc, average=None)
 scores_rfc_rec_0.append(rec[0])
 scores_rfc_rec_1.append(rec[1])
 funo=f1_score(Y[test] , y_rfc, average=None)
 scores_rfc_f1_0.append(funo[0])
 scores_rfc_f1_1.append(funo[1])
 #print(classification_report(y_train[test] , y_log))
 fpr, tpr, threshold = roc_curve(Y[test] , y_rfc_proba[:, 1])
 print("fpr")
 print(fpr)
 print("tpr")
 print(tpr)
 #print(classification_report(y_train[test] , y_mlp))
 print("cont:")
 print(cont)
 tprs_rf.append(interp(mean_fpr_rf , fpr, tpr))
 roc_auc_rf = auc(fpr, tpr)
 MCC_rfc=matthews_corrcoef(Y[test] , y_rfc)
 print("MCC rfc:")
 print(MCC_rfc)
 aucs_rf.append(roc_auc_rf)
 rf_plot= plot_roc_curve(rfc, X[test], Y[test] ,ax=ax3,
 alpha=0.3,label='Fold %d (AUC = %0.2f)' % (cont, roc_auc_rf))
 cont=cont+1


print("Reporte de clasificacion rfc:")
print()
print("scores rfc")
print('accuracy:')
print(scores_rfc_acc)
print('precision 0:')
print(scores_rfc_prec_0)
print('precision 1:')
print(scores_rfc_prec_1)
print('recall 0:')
print(scores_rfc_rec_0)
print('recall_1:')
print(scores_rfc_rec_1)
print('f1_0:')
print(scores_rfc_f1_0)
print('f1_1:')
print(scores_rfc_f1_1)

print("means:")
print('accuracy:', media(scores_rfc_acc))
print('precision 0:', media(scores_rfc_prec_0))
print('precision 1:', media(scores_rfc_prec_1))
print('recall 0:', media(scores_rfc_rec_0))
print('recall 1:', media(scores_rfc_rec_1))
print('f1 0:', media(scores_rfc_f1_0))
print('f1 1:', media(scores_rfc_f1_1))
#print('recall:', media(scores_nn_rec) )
#print('f1:', media(scores_nn)   )
print("std dev:")
print('accuracy:', stdev(scores_rfc_acc))
print('precision 0:', stdev(scores_rfc_prec_0))
print('precision 1:', stdev(scores_rfc_prec_1))
print('recall 0:', stdev(scores_rfc_rec_0))
print('recall 1:', stdev(scores_rfc_rec_1))
print('f1 0:', stdev(scores_rfc_f1_0))
print('f1 1:', stdev(scores_rfc_f1_1))
#means:
#accuracy: 0.8753293731285616
#precision 0: 0.9085781255770141
#precision 1: 0.8077990146278242
#recall 0: 0.9065551102280172
#recall 1: 0.8110119393825252
#f1 0: 0.9071929322448127
#f1 1: 0.8079029731477924
#std dev:
#accuracy: 0.020763187855147462
#precision 0: 0.02491986243187761
#precision 1: 0.046922488678487945
#recall 0: 0.02455680726256119
#recall 1: 0.05025367697046399
#f1 0: 0.016509001487143227
#f1 1: 0.034127490498354895

plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
mean_tpr_rf = np.mean(tprs_rf, axis=0)
print("mean trp rf")
print(mean_tpr_rf)
print("mean fpr rf")
print(mean_fpr_rf)
mean_auc_rf = auc(mean_fpr_rf, mean_tpr_rf)
plt.plot(mean_fpr_rf, mean_tpr_rf, color='blue',
         label='Mean RF (AUC = %0.2f )' % (mean_auc_rf),
         lw=2, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC RF')
plt.legend(loc="lower right",fontsize = 5,ncol=5)
#plt.text(0.32,0.7,'More accurate area',fontsize = 12)
#plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()

##################################################################
tprs_svm = []
aucs_svm = []
mean_fpr_svm = np.linspace(0,1,100)
scores_svm= []
scores_svm_acc=[]
scores_svm_prec_0=[]
scores_svm_prec_1=[]
scores_svm_rec_0=[]
scores_svm_rec_1=[]
scores_svm_f1_0=[]
scores_svm_f1_1=[]
cont=0
# plot arrows
#fig1 = plt.figure(figsize=[12,12])
fig_svm = plt.figure(figsize=[12,12])
ax4 = fig_svm.add_subplot(111,aspect = 'auto')

svm=SVC(kernel='rbf',gamma=0.01,C=10,coef0=0,degree=1, probability= True)

for train, test in cv.split(X,Y):
 svm.fit(X[train], Y[train])
 y_svm = svm.predict(X[test])
 y_svm_proba = svm.predict_proba(X[test])
 #y_test_pred_model = model.predict(x_train[test])
 acc=accuracy_score(Y[test] , y_svm)
 scores_svm_acc.append(acc)
 prec=precision_score(Y[test] , y_svm, average=None)
 scores_svm_prec_0.append(prec[0])
 scores_svm_prec_1.append(prec[1])
 rec=recall_score(Y[test] , y_svm, average=None)
 scores_svm_rec_0.append(rec[0])
 scores_svm_rec_1.append(rec[1])
 funo=f1_score(Y[test] , y_svm, average=None)
 scores_svm_f1_0.append(funo[0])
 scores_svm_f1_1.append(funo[1])
 #print(classification_report(y_train[test] , y_log))
 fpr, tpr, threshold = roc_curve(Y[test] , y_svm_proba[:, 1])
 print("fpr")
 print(fpr)
 print("tpr")
 print(tpr)
 #print(classification_report(y_train[test] , y_mlp))
 tprs_svm.append(interp(mean_fpr_svm , fpr, tpr))
 roc_auc_svm = auc(fpr, tpr)
 MCC_svm =matthews_corrcoef(Y[test] , y_svm)
 print("MCC svm:")
 print(MCC_svm)
 aucs_svm.append(roc_auc_svm)
 svm_plot= plot_roc_curve(svm, X[test], Y[test] ,ax=ax4,
 alpha=0.3,label='Fold %d (AUC = %0.2f)' % (cont, roc_auc_svm))
 print("cont:")
 print(cont)
 cont=cont+1


print("Reporte de clasificacion svm:")
print()
print("scores svm")
print('accuracy:')
print(scores_svm_acc)
print('precision 0:')
print(scores_svm_prec_0)
print('precision 1:')
print(scores_svm_prec_1)
print('recall 0:')
print(scores_svm_rec_0)
print('recall_1:')
print(scores_svm_rec_1)
print('f1_0:')
print(scores_svm_f1_0)
print('f1_1:')
print(scores_svm_f1_1)

print("means svm:")
print('accuracy:', media(scores_svm_acc))
print('precision 0:', media(scores_svm_prec_0))
print('precision 1:', media(scores_svm_prec_1))
print('recall 0:', media(scores_svm_rec_0))
print('recall 1:', media(scores_svm_rec_1))
print('f1 0:', media(scores_svm_f1_0))
print('f1 1:', media(scores_svm_f1_1))
#print('recall:', media(scores_nn_rec) )
#print('f1:', media(scores_nn)   )
print("std dev:")
print('accuracy:',  stdev(scores_svm_acc))
print('precision 0:', stdev(scores_svm_prec_0))
print('precision 1:', stdev(scores_svm_prec_1))
print('recall 0:', stdev(scores_svm_rec_0))
print('recall 1:', stdev(scores_svm_rec_1))
print('f1 0:', stdev(scores_svm_f1_0))
print('f1 1:', stdev(scores_svm_f1_1))

#means svm:
#accuracy: 0.8474707814160147
#precision 0: 0.9015379801075996
#precision 1: 0.7478702617107958
#recall 0: 0.8688166771200649
#recall 1: 0.8043244107085467
#f1 0: 0.884434212580583
#f1 1: 0.7734790702445318
#std dev:
#accuracy: 0.022541681538953362
#precision 0: 0.0269069040716292
#precision 1: 0.04797722593656234
#recall 0: 0.02780311349617788
#recall 1: 0.04974731944259354
#f1 0: 0.018845986153566523
#f1 1: 0.03407186516528925

plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
mean_tpr_svm = np.mean(tprs_svm, axis=0)
print("mean trp svm")
print(mean_tpr_svm)
print("mean fpr svm")
print(mean_fpr_svm)
mean_auc_svm = auc(mean_fpr_svm, mean_tpr_svm)

plt.plot(mean_fpr_svm, mean_tpr_svm, color='blue',
         label='Mean SVM (AUC = %0.2f )' % (mean_auc_svm),
         lw=2, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC SVM')
plt.legend(loc="lower right",fontsize = 5,ncol=5)
#plt.text(0.32,0.7,'More accurate area',fontsize = 12)
#plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()

##################################################################################
#TEST part
#pyplot.figure()
#pyplot.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
#pyplot.plot(fpr_keras, tpr_keras, label='Neuronal Networks Classifier (area = {:.3f})'.format(auc_keras))
#pyplot.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
#ax = pyplot.gca()
