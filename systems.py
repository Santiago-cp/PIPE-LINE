
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np

class ML_System_Regression():

    def __init__(self):
        pass

    def load_data(self):
        path = "C:/Users/totoy/Documents/UNIVERSIDAD/PYTHON/PRACTICA 23082024/"
        dataset = pd.read_csv(path + "iris_dataset.csv", sep=";", decimal=",")
        prueba = pd.read_csv(path + "iris_prueba.csv", sep=";", decimal=",")
        
        covariables = [x for x in dataset.columns if x not in ["y"]]
        X = dataset[covariables]
        y = dataset["y"]
        
        X_nuevo = prueba[covariables]
        y_nuevo = prueba["y"]
        return X, y, X_nuevo, y_nuevo

    def preprocessing_Z(self, X):
        Z = preprocessing.StandardScaler()
        Z.fit(X)
        X_Z = Z.transform(X)
        return Z, X_Z

    def training_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        
        Z_1, X_train_Z = self.preprocessing_Z(X_train)
        X_test_Z = Z_1.transform(X_test)
        
        modelo1 = LogisticRegression(random_state=123)
        parametros = {'C': np.arange(0.1, 5.1, 0.1)}
        grilla1 = GridSearchCV(estimator=modelo1, param_grid=parametros, 
                               scoring=make_scorer(accuracy_score), cv=5, n_jobs=-1)
        grilla1.fit(X_train_Z, y_train)
        y_hat_test = grilla1.predict(X_test_Z)
        
        Z_2, X_test_Z = self.preprocessing_Z(X_test)
        X_train_Z = Z_2.transform(X_train)
        
        modelo2 = LogisticRegression(random_state=123)
        grilla2 = GridSearchCV(estimator=modelo2, param_grid=parametros, 
                               scoring=make_scorer(accuracy_score), cv=5, n_jobs=-1)
        grilla2.fit(X_test_Z, y_test)
        y_hat_train = grilla2.predict(X_train_Z)
        
        e1 = accuracy_score(y_test, y_hat_test)
        e2 = accuracy_score(y_train, y_hat_train)
        
        if (np.abs(e1 - e2) < 0.05):
            modelo_completo = LogisticRegression(C=grilla1.best_params_['C'], random_state=123)
            modelo_completo.fit(X, y)
            return modelo_completo
        else:
            grilla_completa = GridSearchCV(estimator=LogisticRegression(random_state=123), param_grid=parametros, 
                                           scoring=make_scorer(accuracy_score), cv=5, n_jobs=-1)
            grilla_completa.fit(X, y)
            return grilla_completa

    def forecast(self, grilla_completa, X_nuevo):
        yhat_nuevo = grilla_completa.predict(X_nuevo)
        return yhat_nuevo

    def accuracy(self, ytrue, yhat):
        var = np.abs((ytrue - yhat) / ytrue)
        return 100 * np.mean(var <= 0.02)

    def evaluate_model(self, y_nuevo, yhat_nuevo):
        return self.accuracy(y_nuevo, yhat_nuevo)

    def ML_system_regression(self):
        try:
            X, y, X_nuevo, y_nuevo = self.load_data()
            grilla_completa = self.training_model(X, y)
            yhat_nuevo = self.forecast(grilla_completa, X_nuevo)
            metric = self.evaluate_model(y_nuevo, yhat_nuevo)
            return {'success': True, 'accuracy': metric}
        except Exception as e:
            return {'success': False, 'message': str(e)}

