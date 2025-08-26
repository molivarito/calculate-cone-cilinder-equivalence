# Análisis de Resonadores Cónicos Acústicos

Esta es una herramienta de escritorio con una Interfaz Gráfica de Usuario (GUI) desarrollada en Python para el análisis y la visualización de las propiedades acústicas de resonadores cónicos. La aplicación permite comparar las frecuencias de resonancia y las longitudes equivalentes de cilindros calculadas a través de diferentes modelos teóricos.

## Características Principales

*   **Interfaz Gráfica Intuitiva:** Permite modificar fácilmente los parámetros del resonador y visualizar los resultados al instante.
*   **Cálculo de Resonancias:** Implementa dos modelos para calcular las frecuencias de resonancia:
    *   Un modelo ideal.
    *   Un modelo más robusto que incluye la impedancia de radiación en los extremos del cono.
*   **Análisis de Longitud Equivalente (`L_eq`):** Calcula la longitud de un cilindro equivalente utilizando dos aproximaciones teóricas:
    *   Teoría de Perturbaciones.
    *   Aproximación de Onda Esférica.
*   **Visualización de Datos Completa:** Los resultados se presentan en cuatro pestañas distintas:
    1.  **Tabla de Resultados:** Datos numéricos detallados por cada modo de resonancia.
    2.  **Análisis de Frecuencia:** Gráficos de `L_eq` y el error relativo de cada modelo en función de la frecuencia.
    3.  **Visualización Comparativa:** Compara la geometría del cono físico con las longitudes equivalentes de todos los modos calculados.
    4.  **Geometría por Modo:** Una vista detallada que superpone el cono físico y los cilindros equivalentes para un modo seleccionado.
*   **Parámetros Personalizables:** El usuario puede definir la longitud del cono, los radios inicial y final, la velocidad del sonido y el número de modos a calcular.
*   **Exportación de Gráficos:** Cada gráfico se puede guardar en formatos de alta calidad como **PDF, PNG y SVG**, ideal para publicaciones académicas.
*   **Estilo de Publicación:** Los gráficos se generan con un estilo preconfigurado para cumplir con los estándares de publicaciones científicas.

## Vistas de la Aplicación

A continuación se muestran ejemplos de las diferentes vistas que ofrece la herramienta.

**Vista Principal y Controles**

La ventana principal con los controles de parámetros y los botones de acción.

*(Imagen de la ventana principal de la aplicación aquí)*

**Pestaña: Tabla de Resultados**

Muestra los valores numéricos de las frecuencias y las longitudes equivalentes calculadas.

*(Imagen de la pestaña de la tabla aquí)*

**Pestaña: Análisis L vs. Frecuencia**

Gráficos que muestran la longitud equivalente y el error de los modelos en función de la frecuencia.

*(Imagen de la pestaña de análisis de frecuencia aquí)*

## Modelos Teóricos Utilizados

La aplicación se basa en los siguientes modelos acústicos:

1.  **Frecuencias de Referencia (`f_ref`)**
    *   **Modelo Ideal:** Calcula las frecuencias sin considerar las pérdidas por radiación en los extremos.
    *   **Modelo con Radiación:** Resuelve numéricamente una ecuación más compleja que incorpora la impedancia de radiación en los extremos abiertos del cono. Este método utiliza `scipy.optimize.minimize_scalar` para encontrar los números de onda (`k`) que minimizan el determinante de la matriz del sistema, lo que corresponde a las resonancias.

2.  **Longitud Equivalente (`L_eq`)**
    Una vez obtenidas las frecuencias de referencia, se calculan las longitudes de un cilindro "equivalente" (con un radio promedio `r_avg`) que resonaría a esas mismas frecuencias, utilizando dos aproximaciones:
    *   **Teoría de Perturbaciones:** Un modelo que aproxima el efecto de la conicidad como una pequeña perturbación sobre un cilindro.
    *   **Aproximación de Onda Esférica:** Un modelo que considera el campo acústico dentro del cono como una sección de una onda esférica.

El objetivo es validar qué tan bien estas aproximaciones de `L_eq` coinciden con la longitud equivalente de referencia, que se calcula directamente a partir de las frecuencias de resonancia (`L_eq_ref = n*c / (2*f_ref)`).

## Requisitos

*   Python 3.x
*   Tkinter (generalmente incluido en la instalación estándar de Python)
*   NumPy
*   SciPy
*   Matplotlib

Puedes instalar las dependencias necesarias con pip:
```bash
pip install numpy scipy matplotlib
```

## Cómo Usar

1.  Asegúrate de tener Python y las librerías requeridas instaladas.
2.  Ejecuta el script desde tu terminal:
    ```bash
    python valida_cono.py
    ```
3.  En la GUI, ajusta los parámetros del resonador (longitud, radios, etc.) según sea necesario.
4.  Activa o desactiva la inclusión de la impedancia de radiación.
5.  Haz clic en el botón **"Calculate & Update"**.
6.  Explora los resultados en las diferentes pestañas.
7.  Para guardar un gráfico, navega a la pestaña correspondiente y haz clic en el botón **"Save Plot"**.

## Estructura del Código

El código está contenido en un único archivo, `valida_cono.py`, y se organiza de la siguiente manera:

*   **Configuración Global y Constantes:** Define el estilo de los gráficos de Matplotlib y constantes físicas como la densidad del aire.
*   **Lógica de Cálculo:** Un conjunto de funciones (`calculate_ideal_resonances`, `calculate_resonances_with_radiation`, etc.) que contienen la física y las matemáticas del problema. Estas funciones están desacopladas de la GUI.
*   **Clase Principal de la GUI (`AcousticConeGUI`):** Una clase que hereda de `tkinter.Tk` y que construye y gestiona todos los elementos de la interfaz de usuario, las interacciones y la visualización de datos.
