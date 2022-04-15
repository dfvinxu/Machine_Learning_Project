 <!-- IMAGEN ML PROJECT -->
 <br />
 <p align="center">
   <img src="https://www.elon.co/wp-content/uploads/2020/11/machine-learning-banner-web-icon-set-mineria-datos-algoritmo-red-neuronal_35632-107-1-600x183.jpg" alt="drawing" width="300"/>
   </a>

   <h3 align="center">ML Project: Predicción de accidentes cerebrovasculares</h3>


 <!-- TABLA DE CONTENIDOS -->
 <details open="open">
   <summary>Tabla de contenidos</summary>
   <ol>
     <li>
       <a href="#sobre-el-proyecto">Sobre el proyecto</a>
       <ul>
         <li><a href="#hipotesis">Hipótesis</a></li>
       </ul>
     <li>
       <a href="#librerias-utilizadas">Librerias utilizadas</a>
     <li>
      <a href="#resumen-del-proceso-de-trabajo-con-datos">Resumen del proceso de trabajo con datos</a>
      <ul>
        <li><a href="#obtención-y-limpieza-de-datos">Obtención y limpieza de datos</a></li>
        <li><a href="#elaboración-de-los-gráficos">Elaboración de los gráficos</a></li>
      <ul>
   </ol>
 </details>



 <!-- SOBRE EL PROYECTO -->
### Sobre el proyecto

El poryecto se centra en el entrenamiento de un modelo de machine learning para predecir con antelación ataques cerebrovasculares. Para ello, he utilizado el dataset *Stroke Prediction Dataset* publicado en kaggle por [FEDESORIANO] (https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
      
Este proyecto se centra en el análisis de los datos sobre las diferentes causas de muerte en España durante un periodo de 40 años. La finalidad del proyecto es realizar una comparativa de las principales causas. Todos los datos utilizados para la realización del mismo han sido obtenidos directamente de la página oficial del <a href="https://www.ine.es/index.htm">Instituto Nacional de Estadística</a> (INE).

<p align="center">
   <img src="https://i.pinimg.com/originals/c8/4d/a7/c84da7d0e5972410491435b06da4a175.jpg" alt="drawing" width="900"/>
 
 #### Hipótesis
 
 La hipótesis planteada es si el COVID-19 fue la principal causa de muerte en España en 2020.
 
 Una segunda hipótesis, al hilo de la primera, es que hay otras "pandemias" que causan un alto número de muertos en España.

 ### Librerias utilizadas

 Para el tratamiento de los datos y obtención de los gráficos, he utilizado las siguientes librerías: - 
 * [Pandas](https://pandas.pydata.org/)
 * [Numpy](https://numpy.org/)
 * [Seaborn](https://seaborn.pydata.org/)
 * [Matplotlib](https://matplotlib.org/)

### Resumen del proceso de trabajo con datos

#### Obtención y limpieza de datos

Para la realización de este proyecto he utilizado cuatro datasets diferentes que me he descargado de la base de datos del INE. En los cuatro datasets los datos estan clasificados por causas globales, sexo, edad y año.Una breve descripción de los archivos .csv obtenidos sería la siguiente:

- Primer dataset: muestra de todas las muertes en España para el periodo 1980-2020. 
- Segundo dataset: muestra los datos de las muertes causadas por enfermedades del sistema circulatorio.
- Tercer dataset: muestra los datos de las muertes causadas por tumores malignos.
- Cuarto dataset: muestra los datos de las muertes causadas por enfermedades infecciosas y parasitarias.

Una vez leído el fichero .csv en Python, el proceso ha sido similar en los cuatro casos:

- Eliminar aquellas columnas que no iban a resultar útiles (como la de sexo y la de edad, ya que en este proyecto me he basado en datos totales de la población, sin distinción por sexo ni edad).
- Trabajar los datos para obtener distintos dataframes con los que ir elaborando los diferentes gráficos.

#### Elaboración de los gráficos

En los cuatro casos, lo primero que he hecho ha sido realizar un gráfico con cada uno de los datasets con la evolución de las muertes a lo largo del periodo estudiado (1980-2020).

Una vez obtenido este primer gráfico que nos muestra los totales de muertes a lo largo del periodo, he realizado diferentes gráficos para cada dataset comparando las muertes en los años 1980 y 2020.
