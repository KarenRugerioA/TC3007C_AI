# **Implementación principal**

El archivo **['main.ipynb'](https://github.com/myrosandrade89/TC3007C_AI/blob/develop/implementation/main.ipynb)** contiene el flujo principal de la aplicación. A continuación se mostrará la justificación de la toma de decisiones y la explicación del flujo.

---

## **Análisis del problema**

- #### **_Uso de Big Data_**

  Realizando un análisis de los datos nos dimos cuenta que la cantidad de datos es mínima para ser considerada big data, ya que para que se considere big data, los datos tienen que ser tan grandes, rápidos o complejos que es difícil o imposible procesarlos con los métodos tradicionales. Los datos actuales no cuentan con las 5Vs para ser considerado un proyecto de esta magnitud.

  En primer lugar, nos encontramos con un set de archivos que no tiene un gran Volumen, ya que cuenta con poco más de un millón de instancias. Así mismo no se cuenta con Velocidad, es decir, no se adquieren nuevos datos en poco tiempo. De igual manera no se tiene Variedad, ya que solamente contamos con datos de tipo float, integer y object. El set de datos fue adquirido de Kaggle, pero no hay una forma de saber qué tan Veraz es. Finalmente se considera que el Valor de las variables será definido una vez realizado el modelo.

  Así mismo, los socios formadores comentan que en ningún momento consideran trabajar con archivos que cumplan las condiciones de Big Data y como se trata de un set de datos cuyo contexto es el análisis de churn en una empresa, no se trabajará la solución como Big Data. Así mismo, los socios formadores no cuentan con servidores o herramientas que permitan el procesamiento de Big Data, ya que todo se trabajará en un equipo de la empresa de manera local. Debido a esta limitante, tampoco se puede emplear una solución que requiera de un alto poder computacional de procesamiento.

- #### **Herramientas y tecnologías utilizadas**

  Como tecnología principal se hizo uso de Python, utilizando las librerías de Pandas que nos ayudará a trabajar con los datos en formato de Dataset, lo que permite que sea más fácil de manipular, Numpy que nos ayudará a realizar operaciones matriciales con los data frames y Scikit-Learn cuya API contiene diversas herramientas que ayudan a realizar el aprendizaje automático, incluyendo clasificación, regresión, y reducción de dimensionalidad.
  <br>Se optó por la utilización de estas tecnologías debido a las especificaciones del socio formador, quién aseguró que no se contaría con sets de datos de tipo Big Data y que optó por una ejecución rápida. La razón por la cual no se decidió usar Pyspark es porque ésta herramienta es mayormente utilizada para procesar grandes volúmenes de datos y diferentes tipos de datos y no es lo que se trabajará en este reto.
  <br>Por otro lado, para la implementación de modelos más robustos (random forest y convolutional neural networks) se utilizó la herramienta TensorFlow, pues esta herramienta nos permite hacer uso de la GPU para un procesamiento más rápido.
