import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from mlxtend.plotting import plot_decision_regions
from itertools import cycle
from sklearn.svm import SVC,SVR
#from tensorflow.keras.losses import sparse_categorical_crossentropy
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from scipy import interp
#from statistics import stdev

def media(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t
    avg = sum_num / len(num)
    return avg

def stdev(num, media):
    sum = 0
    for t in num:
        sum = sum + (t-media)*(t-media)
    std = np.sqrt(sum / len(num))
    return std

random_state = np.random.RandomState(0)
seed = 7
np.random.seed(seed)

#Ejecutar programa 10 veces
#data=np.loadtxt("interface_data_sola.csv",delimiter=",", skiprows=1)
data=np.genfromtxt("interface_data_sola_filtrado.csv", delimiter=',', skip_header=1)
#data=pd.read_csv('interface_data_sola.csv', sep=',',header=None)
#data = np.loadtxt("interface_data_sola.csv", skiprows=1)
acc_forest=[]
micros_f1_forest=[]
OOB_forest=[]
precision_forest_0=[]
precision_forest_1=[]
recall_forest_0=[]
recall_forest_1=[]
f1_forest_0=[]
f1_forest_1=[]
MCC_forest=[]
true_pos=[]
true_neg=[]
false_pos=[]
false_neg=[]
rate_true_pos=[]
rate_true_neg=[]
rate_false_pos=[]
rate_false_neg=[]
tprs_rfc=[]
aucs_rfc=[]
#filt = an_array[an_array > 4]
print("data")
print(data)
X=data[:,0:6]
Y=data[:,7]
print(X)
print(Y)
print(Y.astype)
y=Y.reshape(-1)
y=y.astype(np.int)
# Binarize the output
#y=np.reshape(Y,-1)
#y=Y.flatten([1])
#Y=Y.reshape(-1)
#y = label_binarize(Y, classes=[0, 1])
#nc = len(y)
x_train, x_test, y_train, y_test = train_test_split(X ,y,test_size=0.30,random_state=42)
#X_train = preprocessing.scale(x_train)
#X_test = preprocessing.scale(x_test)
XA=x_train[:,[1, 5]]
print("XA")
print(XA)
XT=x_test[:,[1, 5]]
print("XT")
print(XT)
print(XT.size)
print(XT.astype)
print(y_test)
print(y_test.astype)
print(y_test.size)

rfc=RandomForestClassifier(n_estimators= 1000,
criterion='entropy', oob_score=True, max_depth=100)
cont=0
#rfc=RandomForestClassifier(random_state=random_state,
#n_estimators=500,criterion='entropy',oob_score=True)
#rfc_result=rfc.fit(XA,y_train)
for i in range(1,11):
 rfc.fit(XA,y_train)
#rfc_result=rfc.fit(x_train,y_train)
#scores = ['f1','precision','recall']
 print("Reporte de clasifocacion forest:")
 print()
#y_rfc =rfc_result.predict_classes(x_test)
 y_rfc =rfc.predict(XT)
 y_rfc_proba = rfc.predict_proba(XT)
 print(confusion_matrix(y_rfc, y_test ))
 print(classification_report(y_rfc , y_test ))
 print()
 acc_rfc=accuracy_score(y_test , y_rfc)
 print("accuaracy rfc")
 print(acc_rfc)
 acc_forest.append(acc_rfc)
 prec_rfc=precision_score(y_test , y_rfc, average=None)
 print("precision rfc")
 print(prec_rfc)
 precision_forest_0.append(prec_rfc[0])
 precision_forest_1.append(prec_rfc[1])
 rec_rfc=recall_score(y_test , y_rfc, average=None)
 print("recall rfc")
 print(rec_rfc)
 recall_forest_0.append(rec_rfc[0])
 recall_forest_1.append(rec_rfc[1])
 funo_rfc=f1_score(y_test , y_rfc, average=None)
 print("f1 score rfc")
 print(funo_rfc)
 f1_forest_0.append(funo_rfc[0])
 f1_forest_1.append(funo_rfc[1])
 fpr_rfc, tpr_rfc, threshold = roc_curve(y_test , y_rfc_proba[:, 1])
 print("fpr rfc")
 print(fpr_rfc)
 print("tpr rfc")
 print(tpr_rfc)
 print("Reporte de clasifocacion RF")
 print()
 # y_rfc =rfc_result.predict(x_test)
# print(confusion_matrix(yy_test , y_rfc))
 TN, FP, FN, TP = confusion_matrix(y_test , y_rfc).ravel()
 print("size test")
 print(y_test.size)
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
 rate_TP=(TP/y_test.size)*100
 rate_TN=(TN/y_test.size)*100
 rate_FP=(FP/y_test.size)*100
 rate_FN=(FN/y_test.size)*100
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
 micro_f1_rfc = f1_score(y_test , y_rfc, average='micro')
 print("f1 score micro rfc")
 print(micro_f1_rfc)
 micros_f1_forest.append(micro_f1_rfc)
 auc_forest = auc(fpr_rfc, tpr_rfc)
 mcc = matthews_corrcoef(y_test, y_rfc)
 print("MCC forest:")
 print(mcc)
 MCC_forest.append(mcc)
 #tprs_rfc.append(interp(mean_fpr_rfc , fpr_rfc, tpr_rfc))
 roc_auc_rfc = auc(fpr_rfc, tpr_rfc)
 #aucs_rfc.append(roc_auc_rfc)
 #plt.plot(fpr_rfc, tpr_rfc, lw=2, alpha=0.3,
 #label='ROC fold %d (AUC = %0.2f)' % (cont, roc_auc_rfc))
 print("cont:")
 print(cont)
 cont=cont+1
 OOB=rfc.oob_score_
 print("OOB error:")
 print(OOB)
 OOB_forest.append(OOB)
 IMP=rfc.feature_importances_.argsort()
 print("Mejores caracteristicas")
 print(IMP)

print("Reporte de clasificacion RF:")
print()
print("scores micro RF ")
print('Accuracy RF:')
print(acc_forest)
print('micro-F1 RF:')
print(micros_f1_forest)
print('precision RF 0:')
print(precision_forest_0)
print('precision RF 1:')
print(precision_forest_1)
print('recall RF 0:')
print(recall_forest_0)
print('recall RF 1:')
print(recall_forest_1)
print('F1 RF 0')
print(f1_forest_0)
print('F1 RF 1')
print(f1_forest_1)
print('OOB RF:')
print(OOB_forest)
print('MCC RF')
print(MCC_forest)

print("Means RF:")
print('Accuracy:', media(acc_forest))
print('micro-F1 RF:', media(micros_f1_forest))
print('precision RF 0:', media(precision_forest_0))
print('precision RF 1:', media(precision_forest_1))
print('recall RF 0:', media(recall_forest_0))
print('recall RF 1:', media(recall_forest_1))
print('F1 0:', media(f1_forest_0))
print('F1 1:', media(f1_forest_1))
print('OOB:', media(OOB_forest))
print('MCC:', media(MCC_forest))

print("Std Dev:")
print('Accuracy:', stdev(acc_forest, media(acc_forest)   ))
print('micro-F1 RF:', stdev(micros_f1_forest,  media(micros_f1_forest)   ))
print('precision RF 0:', stdev(precision_forest_0,   media(precision_forest_0) ))
print('precision RF 1:', stdev(precision_forest_1,  media(precision_forest_1) ))
print('recall RF 0:', stdev(recall_forest_0,  media(recall_forest_0)  ))
print('recall RF 1:', stdev(recall_forest_1,  media(recall_forest_1)  ))
print('F1 0:', stdev(f1_forest_0, media(f1_forest_0)  ))
print('F1 1:', stdev(f1_forest_1,  media(f1_forest_1) ))
print('OOB:', stdev(OOB_forest, media(OOB_forest) ))
print('MCC:', stdev(MCC_forest, media(MCC_forest)  ))

#x_axis = [i for i in range(10,110,10)]
x_axis = [i for i in range(1,11,1)]
print("x axis")
print(x_axis)

plt.plot(x_axis, acc_forest, marker='o', color='red',label='RF' )
plt.xlabel('Trian/Test time',fontsize = 10)
plt.ylabel('Accuracy rate',fontsize = 10)
#plt.title('Random Forest using Depth=100',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=1)
plt.show()

plt.plot(x_axis, micros_f1_forest, marker='o', color='red',label='RF' )
plt.xlabel('Trian/Test time',fontsize = 10)
plt.ylabel('Micro F1 rate',fontsize = 10)
#plt.title('Random Forest using Depth=100',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=1)
plt.show()

plt.plot(x_axis, precision_forest_0, marker='o', color='red', label='RF' )
plt.xlabel('Trian/Test time',fontsize = 10)
plt.ylabel('Precision rate No Hot',fontsize = 10)
#plt.title('Random Forest using Depth=100',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=1)
plt.show()

plt.plot(x_axis, precision_forest_1, marker='o', color='red', label='RF' )
plt.xlabel('Trian/Test time',fontsize = 10)
plt.ylabel('Precision rate Hot',fontsize = 10)
#plt.title('Random Forest using Depth=100',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=1)
plt.show()

plt.plot(x_axis, recall_forest_0, marker='o', color='red',label='RF' )
plt.xlabel('Trian/Test time',fontsize = 10)
plt.ylabel('Recall rate No Hot',fontsize = 10)
#plt.title('Random Forest using Depth=100',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=1)
plt.show()

plt.plot(x_axis, recall_forest_1, marker='o', color='red',label='RF' )
plt.xlabel('Trian/Test time',fontsize = 10)
plt.ylabel('Recall rate Hot',fontsize = 10)
#plt.title('Random Forest using Depth=100',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=1)
plt.show()

plt.plot(x_axis, f1_forest_0, marker='o', color='red',label='RF' )
plt.xlabel('Trian/Test time',fontsize = 10)
plt.ylabel('F1 score rate No Hot',fontsize = 10)
#plt.title('Random Forest using Depth=100',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=1)
plt.show()

plt.plot(x_axis, f1_forest_1, marker='o', color='red',label='RF' )
plt.xlabel('Trian/Test time',fontsize = 10)
plt.ylabel('F1 score rate Hot',fontsize = 10)
#plt.title('Random Forest using Depth=100',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=1)
plt.show()

plt.plot(x_axis, OOB_forest, marker='o', color='red',label='OOB')
plt.xlabel('Trian/Test time',fontsize = 10)
plt.ylabel('OOB rate',fontsize = 10)
#plt.title('Random Forest using Depth=100',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=1)
plt.show()

plt.plot(x_axis, MCC_forest, marker='o', color='red',label='RF' )
plt.xlabel('Trian/Test time',fontsize = 10)
plt.ylabel('MCC rate',fontsize = 10)
#plt.title('Random Forest using Depth=100',fontsize = 12)
plt.legend(loc="lower right",fontsize = 10 ,ncol=1)
plt.show()


scatter_kwargs = {'s': 25, 'edgecolor': None, 'alpha': 0.8}
contourf_kwargs = {'alpha': 0.3}
plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=16)
plt.grid(True)

print("Training projection")
ax=plot_decision_regions(XT, y_test, clf=rfc, legend=3,
scatter_kwargs=scatter_kwargs,  contourf_kwargs=contourf_kwargs)
            #          X_highlight=XT,
                #      Y_highlight=y1_test,
                    #  scatter_highlight_kwargs=scatter_highlight_kwargs)
print("Tr")
lw = 2
plt.xlabel('Area ($\AA^{2})$')
plt.ylabel('Energy $\displaystyle\ (E_h)$')
plt.title('')
#plt.title('Hyperplane projection between energy/Area for the random forest classifier')
#Hiperplane Energ/Area del clasificador SVMs de todos los residuos')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Non-hot spots/non-hot regions', 'Hot spot/hot regions'],framealpha=0.4, prop={'size': 18}, scatterpoints=1)
plt.show()
print("check final")
