# **Preparación de los datos | ETL**

_Autores:_

- _Myroslava Sánchez Andrade A01730712_
- _Karen Rugerio Armenta A01733228_
- _José Antonio Bobadilla García A01734433_
- _Alejandro Castro Reus A01731065_

_Fecha de creación: 01/10/2022_

_Última modificación: 27/10/2022_

---

## **Extracción**

Los datos fueron extraídos de la plataforma Kaggle, del set de datos **[telecom_churn_me.csv](https://www.kaggle.com/datasets/mark18vi/telecom-churn-data?resource=download)**. Este conjunto de datos fue seleccionado por nuestro socio formador `NAATIK | AI Solutions`.
<br>Este conjunto de datos de una empresa de telecomunicaciones contiene la información de la cuenta de los clientes y si estos abandonaron o no la compañía en el último mes.

### **Análisis de los datos**

Antes de realizar cualquier tipo de procesamiento de los datos, estos primero fueron analizados para determinar el tipo de herramientas y técniccas a utilizar.

#### **_Herramientas y tecnologías utilizadas_**

Como tecnología principal, se hizo uso de Python, utilizando las librerías de Pandas, Numpy y Scikit-Learn, se optó por la utilización de estas tecnologías debido a las especificaciones del socio formador, quién aseguró que no se contaría con sets de datos de tipo Big Data y que optó por una ejecución rápida. La razón por la cual no se decidió usar Pyspark es porque ésta herramienta es mayormente utilizada para procesar grandes volúmenes de datos y ayudan a analizar diferentes tipos de datos y no es lo que se trabajará en este reto.

#### **_Modelo de almacenamiento de datos_**

El socio formador no desea hacer uso de tecnologías tales como cómputo en la nube o servidores propios. Las especificaciones incluyen el uso del programa en una computadora perteneciente a la empresa y que pueda ser ejecutado de manera local. Es por ello que como almacenamiento se realizó un archivo con el set de datos limpio y listo para ser procesado. Este archivo es de tipo .csv y está dividido en 3 sets de datos para implementar un k-fold cross validation.

#### **_Big Data_**

Este proyecto no puede ser considerado Big Data, ya que no cuenta con las 5Vs para ser considerado un proyecto de esta magnitud. En primer lugar, nos encontramos con un set de archivos que no tiene un gran Volumen, ya que cuenta con poco más de un millón de datos. Así mismo no se cuenta con Velocidad, es decir, no se adquieren nuevos datos en poco tiempo. De igual manera no se tiene Variedad, ya que sólamente contamos con datos de tipo float, integer y object. El set de datos fue adquirido de Kaggle, pero no hay una forma de saber qué tan veráz es. Finalmente se considera el Valor de las variables será definido una vez realizado el modelo, es por ello que la solución de este reto no contemplará el manejo de datos como Big Data.

---

## **Transformación**

Sabemos que para realizar un modelo de predicción, es indispensable que los datos con los que se trabaje sean los correctos. Es por eso que se realizaron los siguientes procesos para asegurarnos de que los datos que serán alimentados para nuestro modelo sean de gran utilidad y eviten la creación de sesgos.

Uno de los requerimientos del socio formador, fue que el código del procesamiento de los datos, fuera lo más generalizado posible para que este pudiera ser utilizado con cualquier otro conjunto de datos. A pesar de que se intentó generalizar lo más posible el código, se recomienda que se haga un análisis y posibles cambios dado un nuevo conjunto de datos.

#### **_Análisis de columnas_**

- **Evaluación de columnas**:
  Las columnas que tuvieran un 65% o más de valores nulos fueron eliminadas para evitar afectar el procedimiento de entrenamiento.
  <br>De igual manera, se eliminaron las columnas que estuvieran completamente compuestas de valores únicos, esto porque se pretende generalizar el código y una columna sin valores repetidos **puede** ser una sin significado (ejemplo: ID, Nombre, etc.).
  <br>También se eliminaron las columnas que estuverian completamente compuestas de un sólo valor, pues no aportatían a un modelo.

- **One Hot Encoding**:
  Las columnas categóricas fueron identificadas a patir de sus valores únicos. Si se cuenta con un pequeño número de valores únicos, **puede** que se trate de una columna categórica.
  <br>Una vez
