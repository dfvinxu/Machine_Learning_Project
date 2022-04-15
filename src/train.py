import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedBaggingClassifier

from main import train, test, target, save_file, mostrar_resultados

# División del dataset de train
X = train.drop(columns='stroke')
y = train['stroke']

# Oversample para compensar el desbalanceo en la target
oversample = RandomOverSampler(sampling_strategy=0.60)
X_over, y_over = oversample.fit_resample(X, y)

# División en train y test
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.20, random_state=42)

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

# Muestra los resultados obtenidos al aplicar el modelo al dataset de test
mostrar_resultados(target, predicciones)

# Guarda el modelo en la ruta especificada
save_file(bbc, r'C:\Users\dfvin\OneDrive\Documentos\Bootcamp_DS\Alumno\3-Machine_Learning\Proyecto_Machine_Learning\src\model\new_model')