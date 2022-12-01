# **Implementación principal**

El archivo **['main.ipynb'](https://github.com/myrosandrade89/TC3007C_AI/blob/develop/implementation/main.ipynb)** contiene el flujo principal del sistema; a continuación se expondrá esta.

**Es muy importante resaltar que se trata de una implementación generalizada, por lo que la precisión de las predicciones puede variar. Además, también se buscó realizar una implementación que reciba cualquier conjunto de datos, pero se recomienda hacer un análisis de los datos antes de introducirlo al modelo ya que formatos no esperados pueden provocar errores en la ejecución.**

Se cuenta con una lista de requerimientos para correr el sistema en el docuemnto 'requirements.txt'

---

### **Importado de funciones**

Los archivos de: configuración de almacenamiento, el etl, los modelos y la transformación para la predicción fueron importados como funciones de python.

---

### **Definición de variables**

Ya que se trata de una implementación generalizada, todas las celdas con el título '_THESE VARIABLES MUST BE GIVEN BY THE USER, PLEASE EDIT TO USE_' deberán ser editadas por el usuario. A continuación se enlistan las variables a definir por el usuario:

- '_dataset_path_': ubicación del archivo (_path_) a modelar. Esta ubicación no se debe encontrar en un directorio en específico. Seguir el formato del ejemplo: duplicado del caracter '\\' y con terminación del nombre del archivo. El archivo **debe** ser **CSV**.

- '_original_name_dataset_': nombre del archivo a modelar (sin la extensión '.csv').

- '_target_column_name_': nombre de la columna a predecir (case sensitive).

- '_smote_': booleano que define si los modelos utilizarán el algoritmo SMOTE para balancear la clase a predecir. En caso de que **no se pueda aplicar smote** por un alto desbalance en el conjunto de datos, los modelos imprimirán el mensaje 'There are not enough instances from a class to use the SMOTE algorithm, training without smote'.

---

### **Configuración del almacenamiento**

Esta función recibe la ubicación del archivo y el nombre del archivo. Una vez dada la ubicación del archivo que se desea modelar, se asegura de que no exista ninguna archivo o carpeta con este mismo nombre por lo que se elimina cualquier carpeta o archivo ya existente de este de la carpeta 'data' y de la carpeta 'joblibs'. Después se crean las carpetas correspondientes que contendrán los modelos, archivos de entrenamientos, codificadores, etc.

---

### **Preparación de los datos**

Esta función recibe el nombre del archivo y el nombre de la columna a predecir. Realizará transformaciones **generalizadas** y guardará estas en archivos con extensión '.joblib', además de los archivos para el entrenamientos y el test.

En el archivo se muestra un DataFrame que expone el total de los datos y el porcentaje de churn y no churn. Esta matriz fue generada de uno de los joblibs que generó el etl.

---

### **Entrenamiento del modelo**

Para los modelos mlp, decission_tree, logistic_regression y random_forest, se tiene la siguiente implementación:

La función recibe como parámetros el nombre de la columna a predecir, el nombre del conjunto de datos a modelar y la variable smote que define si se utilizará o no el algoritmo SMOTE; esta función regresa una matriz de confusión y los resultados del modelo (string con el porcentaje). También se observa que se imprime la precisión del k-cross fold validation en una gráfica y la precisión del modelo en el entrenamiento y el test.

Al llamar a la matriz de confusión, podemos notar el DataFrame como salida. También se importa una matriz que contiene los _true_predicted_percentage_ del modelo que fue generada en la función del modelo y guardada como joblib.
