import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Función para crear el modelo cada vez
def log_reg_model(X_train, X_test, y_train, y_test):
    clf_base = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver='newton-cg')
    clf_base.fit(X_train, y_train)
    return clf_base

# Función que mostrará los resultados
def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(9,6))
    sns.heatmap(conf_matrix, annot=True)
    plt.title('Confusion matrix')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    print (classification_report(y_test, pred_y))

# Guardar el modelo
def save_file(model, file_name):
    with open(file_name, "wb") as exit_file:
        pickle.dump(model, exit_file)

# Cargar el modelo
def load_file(file_name):
    with open(file_name, "rb") as input_file:
        model = pickle.load(input_file)
    return model