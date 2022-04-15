 <!-- IMAGEN ML PROJECT -->
 <br />
 <p align="center">
   <img src="https://www.profesionalreview.com/wp-content/uploads/2019/08/Machine-Learning-1-1024x732.png" alt="drawing" width="500"/>
   </a>

   <h3 align="center">ML PROJECT: Predicción de accidentes cerebrovasculares</h3>
<br>
<br>

### Sobre el proyecto

El poryecto se centra en el entrenamiento de un modelo de machine learning para predecir con antelación ataques cerebrovasculares. Para ello, he utilizado el dataset *Stroke Prediction Dataset* publicado en kaggle por [FEDESORIANO](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

 ### Librerias utilizadas

 Para el tratamiento de los datos y obtención de los gráficos, he utilizado las siguientes librerías: - 
 * [Pandas](https://pandas.pydata.org/)
 * [Numpy](https://numpy.org/)
 * [Seaborn](https://seaborn.pydata.org/)
 * [Matplotlib](https://matplotlib.org/)
 * [Scickit-Learn](https://scikit-learn.org/stable/index.html)
 * [Imbalanced_Learn](https://imbalanced-learn.org/stable/)
 * [Pickle](https://docs.python.org/3/library/pickle.html)
 * [Collections](https://docs.python.org/3/library/collections.html)

### Resumen del proceso de trabajo con datos

Lo primero de todo es el tratamiento de los datos que contiene el dataset con el que entrenamos el modelo. Por este motivo, es importante analizar qué tipo de datos tenemos, ver como se relacionan entre ellos y, sobertodo, con la *target*. En este caso, existen varias columnas de tipo *object* y algunos NaN en la columna *bmi*. Además, se trata de un dataset desbalanceado, ya que la *target* contiene 4861 negativos y tan solo 249 positivos.
<br>
Una vez realizado un análisis de los datos que contiene el dataset a partir de un pequño EDA, toca realizar una serie de transformaciones en los datos para que el modelo pueda utilizarlos para ser entrenado. De este modo llevo a cabo una serie de cambios en las variables categóricas binarias para obtener 0s y 1s, trato los NaN de la variable *bmi* a partir de la aplicación de un *KNNImputer*, aplico un *get_dummies* a las columnas *work_type* y *smoking_status*...
<br>
Llegados a este punto, ya podemos hacernos una idea del tipo de modelo que voy a necesitar para el proyecto. En este caso, se trata de un modelo de clasificación ya que el objetivo es clasificar los resultados en positivos o negativos en función de si los sujetos pueden sufrir un accidente cerebrovascular en base a los datos recogidos.
<br>
Una vez ralizadas las transformaciones necesarias y teniendo claro qué tipo de modelo voy a necesitar, llevo a cabo la división del dataset original en tres datasets diferentes: uno será el dataset de train con el que entrenaré los modelos; otro será el dataset de test con el que pondré a prueba el modelo entrenado; y por último, un dataset con la target para realizar las comprobaciones a partir del dataset de test.
<br>
El siguiente paso que llevo a cabo es un *oversample* de los datos del dataset de train para poder obetener más datos de casos positivos y así poder entrenar un mejor modelo. Una vez hecho esto, divido en *train* y *test* y empiezo a entenar diferentes modelos.
<br>
Empiezo entrenando modelos sencillos para ver qué tal se comportan con los datos tal como estan en este punto. Lo primero que hago es entrenar un modelo de regresión logística y un random forest. A partir de aquí, pruebo otros modelos, como XGBoost o Balanced Baggin Classifier y busco los mejores parámetros para los diferentes modelos con el uso de GridSearchCV.
<br>
Tras entrenar los diferentes modelos y buscar los mejores parámetros para cada uno de ellos, veo que el mejor que logro obtener es un Balanced Bagging Classifier con un Random Forest Classifier como *base_estimator*, obeniendo una recall del 70% para los falsos positivos y un 97& para los casos positivos.
