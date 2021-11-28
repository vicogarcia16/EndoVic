#Importación de las librerias necesarias
import numpy as np1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,train_test_split, StratifiedKFold, KFold, LeaveOneOut 
from sklearn.metrics import recall_score, f1_score, confusion_matrix, auc, roc_auc_score, accuracy_score, precision_score
from imblearn.metrics import specificity_score, geometric_mean_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.svm import LinearSVC
import joblib 

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
      lb = LabelBinarizer()
      lb.fit(y_test)
      y_test = lb.transform(y_test)
      y_pred = lb.transform(y_pred)
      return roc_auc_score(y_test, y_pred, average=average)

def clasif(v):
      knn = ''
      if v == 'KNN':
            grid_params = { 'n_neighbors': [1, 3, 5, 7, 9, 11], 'weights':['uniform', 'distance']  }
            knn= GridSearchCV(KNeighborsClassifier(), grid_params, cv= 10)
      elif v == 'NAB':
            grid_params = { 'priors': [None], 'var_smoothing':[1e-09,1e-02,1e-5] }
            knn= GridSearchCV(GaussianNB(), grid_params, cv= 10)
      elif v == 'RF':
            #grid_params = { 'n_estimators': [50, 75, 100, 125, 150, 175, 200, 225], 'max_depth': [2, 5, 8, 10]    }
            grid_params = { 'n_estimators': [50, 75, 100, 250, 500], 'max_depth': [2, 5]    }
            knn = GridSearchCV(RandomForestClassifier(), grid_params, cv = 10)
      elif v == 'MLP':
            grid_params = { 'solver': ['lbfgs', 'adam', 'sgd'], 'activation': ['logistic','tanh','relu'],  'max_iter': [50, 100, 250, 500] }    
            knn = GridSearchCV(MLPClassifier(), grid_params, cv = 10) 
      elif v =='ADA':
            grid_params = {'n_estimators': [50,100,250,500],'random_state': [0], 'algorithm': ['SAMME.R','SAMME']}
            knn = GridSearchCV(AdaBoostClassifier(), grid_params, cv = 10)    
      elif v == 'DTC':
            grid_params = {'random_state': [0], 'criterion': ['gini','entropy'],'splitter':['best','random'],'max_depth': [2,5]}   
            knn = GridSearchCV(DecisionTreeClassifier(), grid_params, cv = 10) 
      elif v =='SVC':
            grid_params = {'penalty':['l2'],'random_state': [0],'C':[1.0],'loss':['hinge', 'squared_hinge'],'dual':[True,False],'tol':[1e-4],\
                            'multi_class':['ovr', 'crammer_singer'],'max_iter':[50, 100, 250, 500]}
            knn = GridSearchCV(LinearSVC(), grid_params, cv = 10) 
      return knn
            
            
      


#Leer los datos del archivo iris.data y extraer lo necesario
data = np1.genfromtxt('/home/pi/Desktop/EndoVic/data/ct.csv', delimiter=',')

y= data[:,-1]
X = data[:,0:-1]


print('\n\nMatriz 1\n\n',X,'\n')
print('\n\nMatriz 2\n\n',y, '\n')

stra = StratifiedKFold(n_splits = 5)
stra.get_n_splits(X)
knn= clasif('ADA')
knn.fit(X,y)
print('Mejores parametros: ', knn.best_params_)
#print('\nValor Knn: ',knn)
param = knn.best_params_
pred=[]
scores =[]
ytestt =[]
ytestt = np1.array(ytestt)
predt=[]
predt = np1.array(predt)
list3=[]
prom=0
i = 0
for train_index, test_index in stra.split(X,y):
      #normalizar = MinMaxScaler()
      #normalizar = StandardScaler()
      i = i+1
      print("\nFOLD [",i,"]")
      print("\nTRAIN:", train_index, "\nTEST:", test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      #print('\nEntrenamiento X\n\n', X_train,'\n')
      #print('\nEntrenamiento y\n\n', y_train, '\n')
      #print('\nPruebas X\n\n', X_test, '\n')
      #print('\nPruebas y\n\n', y_test, '\n')
      #normalizar.fit(X_train)
      #X_train = normalizar.transform(X_train)
      #X_test = normalizar.transform(X_test)
      #joblib.dump(normalizar, 'irisnorm')
      #knn.fit(X_train,y_train)
      #clasificador = KNeighborsClassifier(n_neighbors = knn.best_params_['n_neighbors'], weights= knn.best_params_['weights'] )
      #clasificador = GaussianNB(priors=knn.best_params_['priors'],var_smoothing = knn.best_params_['var_smoothing'] )
      #clasificador = RandomForestClassifier(n_estimators = knn.best_params_['n_estimators'], max_depth = knn.best_params_['max_depth'])
      #clasificador = MLPClassifier(solver = knn.best_params_['solver'], activation= knn.best_params_['activation'], max_iter= knn.best_params_['max_iter'])
      clasificador = AdaBoostClassifier(n_estimators = param['n_estimators'], random_state= knn.best_params_['random_state'], algorithm=knn.best_params_['algorithm'])
      #clasificador = DecisionTreeClassifier(random_state=knn.best_params_['random_state'],criterion= knn.best_params_['criterion'],splitter = knn.best_params_['splitter'], max_depth = knn.best_params_['max_depth'] )
      #clasificador = LinearSVC(penalty = knn.best_params_['penalty'],random_state = knn.best_params_['random_state'],C = knn.best_params_['C'],loss = knn.best_params_['loss'],dual = knn.best_params_['dual'],tol = knn.best_params_['tol'], multi_class = knn.best_params_['multi_class'],max_iter = knn.best_params_['max_iter'] )
      print(clasificador.get_params)
      pred = knn.predict(X_test)
      print('\nPredicción: ',pred,'\n')
      ytestt = np1.hstack([ytestt,y_test])
      predt = np1.hstack([predt,pred])
      print('Clases verdaderas', y_test)
      scores = knn.score(X_test, y_test)
      print('\nScore: ',scores, '\n')
      prom = prom + scores
porcentaje = prom/stra.n_splits
print('\n--------------------------------------------------------------------------------------') 
print('\nEl promedio de clasificación: ',porcentaje)
print('\n----------------------Métricas de evaluación - Clasificación--------------------------')    
#metricas
mc = confusion_matrix(ytestt, predt)
print('\nMatriz de Confusión: \n',mc, '\n')
acc= accuracy_score(ytestt, predt, normalize= True)
print('Exactitud: ',acc, '\n')
sen= recall_score(ytestt, predt, average = 'weighted')
print('Sensibilidad: ',sen, '\n')
esp = specificity_score(ytestt, predt, average = 'weighted')
print('Especificidad: ',esp, '\n')
pre = precision_score(ytestt, predt, average = 'weighted')
print('Precisión: ',pre, '\n')
f1 = f1_score(ytestt, predt, average = 'weighted')
print('Puntaje F1: ',f1, '\n')
auc = roc_auc_score(ytestt, predt, sample_weight=None )
print('Área bajo la curva: ',auc, '\n')
gm = geometric_mean_score(ytestt, predt, average = 'weighted')
print('Media Geométrica: ',gm, '\n')

# Guardar el modelo.
joblib.dump(knn, '/home/pi/Desktop/modelo_entrenado.pkl') 

# Cargar el modelo.
knn1 = joblib.load('/home/pi/Desktop/modelo_entrenado.pkl')
# Cargar datos de prueba
X0 = np1.genfromtxt('/home/pi/Desktop/maps1.csv', delimiter=',')
X0 = np1.array(X0.reshape(-1, 14))
#Realizar predicción de la clase a la que pertenece
# pred1 = knn1.predict(X0)
# print('\nPredicción: ',pred1,'\n')

# fo = open('archivo_metricas' + ".csv", "a")
# fo.write(str(param) + "," + str(porcentaje) + "," + str(mc)+ "," + str(acc)+ "," + str(sen)+ "," + str(esp)+ "," + str(pre) + "," + str(f1)+ "," + str(auc)+ "," + str(gm)+ "\n")
# fo.close  