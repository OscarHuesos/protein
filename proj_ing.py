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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize

random_state = np.random.RandomState(0)
seed = 7
np.random.seed(seed)

#Ejecutar programa 10 veces
#data=np.loadtxt("interface_data_sola.csv",delimiter=",", skiprows=1)
data=np.genfromtxt("interface_data_sola_filtrado.csv", delimiter=',', skip_header=1)
#data=pd.read_csv('interface_data_sola.csv', sep=',',header=None)
#data = np.loadtxt("interface_data_sola.csv", skiprows=1)

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
X_train = preprocessing.scale(x_train)
X_test = preprocessing.scale(x_test)

XA=x_train[:,[1, 5]]
print("XA")
print(XA)
XT=x_test[:,[1, 5]]
print("XT")
print(XT)
rfc=RandomForestClassifier(random_state=random_state,n_estimators=500,criterion='entropy',oob_score=True)
#rfc_result=rfc.fit(XA,y_train)
rfc.fit(XA,y_train)
#rfc_result=rfc.fit(x_train,y_train)
#scores = ['f1','precision','recall']
print("Reporte de clasifocacion forest:")
print()
#y_rfc =rfc_result.predict_classes(x_test)
y_rfc =rfc.predict(XT)
print(confusion_matrix(y_test , y_rfc))
print(classification_report(y_test , y_rfc ))
print()
#scores = ['f1','precision','recall']
#print("Reporte de clasifocacion forest:")
svm = SVC(gamma=0.01,kernel='rbf',C=100,probability=True,random_state=random_state)
svm.fit(XA, y_train)
print()
print(XT)
print(XT.size)
print(XT.astype)

print(y_test)
print(y_test.astype)
print(y_test.size)

scatter_kwargs = {'s': 25, 'edgecolor': None, 'alpha': 0.8}
contourf_kwargs = {'alpha': 0.3}
plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=16)
plt.grid(True)

print("Training projection")
ax=plot_decision_regions(XT, y_test, clf=rfc, legend=3,
scatter_kwargs=scatter_kwargs,
contourf_kwargs=contourf_kwargs)
            #          X_highlight=XT,
                #      Y_highlight=y1_test,
                    #  scatter_highlight_kwargs=scatter_highlight_kwargs)
#print("Tr")
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
