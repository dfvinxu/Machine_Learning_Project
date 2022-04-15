import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report

test = pd.read_csv(r'C:\Users\dfvin\OneDrive\Documentos\Bootcamp_DS\Alumno\3-Machine_Learning\Proyecto_Machine_Learning\src\data\processed\test.csv')
target = pd.read_csv(r'C:\Users\dfvin\OneDrive\Documentos\Bootcamp_DS\Alumno\3-Machine_Learning\Proyecto_Machine_Learning\src\data\processed\target.csv')
train = pd.read_csv(r'C:\Users\dfvin\OneDrive\Documentos\Bootcamp_DS\Alumno\3-Machine_Learning\Proyecto_Machine_Learning\src\data\processed\train.csv')

# División del dataset de train
X = train.drop(columns='stroke')
y = train['stroke']

# Oversample para compensar el desbalanceo en la target
oversample = RandomOverSampler(sampling_strategy=0.60)
X_over, y_over = oversample.fit_resample(X, y)

# División en train y test
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.20, random_state=42)

# GridSearchCV para el RandomForestClassifier
rf = RandomForestClassifier()

n_estimators = np.arange(1, 10, 1)
max_depth = np.arange(1, 6, 1)
min_samples_split = np.arange(1, 8, 1)
min_samples_leaf = np.arange(1, 8, 1)
bootstrap = [True, False]

grid_rf = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
        }

gs_rf = GridSearchCV(rf, param_grid=grid_rf, cv=10, scoring='recall',  n_jobs = -1)
gs_rf.fit(X_train, y_train)
rf_best = gs_rf.best_params_

# GridSearchCV para el BalancedBaggingClassifier
bbc_clf = BalancedBaggingClassifier()

n_estimators = np.arange(2, 12, 2)
bootstrap = [True, False]
replacement = [True, False]

grid_bbc = {
        'n_estimators': n_estimators,
        'bootstrap': bootstrap,
        'replacement': replacement,
        }

gs_bbc = GridSearchCV(bbc_clf, param_grid=grid_bbc, cv=10, scoring='recall',  n_jobs = -1)
gs_bbc.fit(X_train, y_train)
bbc_best = gs_bbc.best_params_

# Creación del modelo con BalancedBagginClassifier en base a un RandomForestClassifier en función de los parámetros obtenidos con los GridSearchCV
bbc = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(max_depth=4, bootstrap=True, min_samples_leaf=1, min_samples_split=3, n_estimators=2, random_state=42),
                                bootstrap=True,
                                replacement=True,
                                random_state=42,
                                n_estimators=10)

# Entrenamiento del modelo
bbc.fit(X_train, y_train)

# Predicciones sobre el dataset de train
y_pred = bbc.predict(X_test)

# Predicciones sobre el dataset de test
predicciones = bbc.predict(test)
# mostrar_resultados(target, predicciones)

# Función para mostrar los resultados obtenidos
def mostrar_resultados(target, y_pred):
    conf_matrix = confusion_matrix(target, y_pred)
    plt.figure(figsize=(7,4))
    sns.heatmap(conf_matrix, annot=True)
    plt.title('Confusion matrix')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    print (classification_report(target, y_pred))

def save_file(model, file_name):
    with open(file_name, "wb") as exit_file:
        pickle.dump(model, exit_file)

# Guardamos el modelo en la ruta especificada
# save_file(bbc, r'C:\Users\dfvin\OneDrive\Documentos\Bootcamp_DS\Alumno\3-Machine_Learning\Proyecto_Machine_Learning\src\model\my_model')