# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 22:48:48 2024

@author: GonCue
"""
################################### IMPORTAR LIBRERIAS

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import jarque_bera
from sklearn.metrics import mean_squared_error
import seaborn as sns
from scipy import stats
import numpy as np


#################################### LEER ARCHIVO

# Ruta del archivo Excel
ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'
# La notación r antes de la cadena de la ruta (r'C:\...') indica que la cadena 
#es una cadena de texto sin procesar, lo que significa que los caracteres de 
#barra invertida \ no se interpretarán como caracteres de escape. 
#Esto es especialmente útil en rutas de archivo en sistemas Windows.

# Leer el archivo Excel en un DataFrame
df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0)
#los años están en la primera columna

# Mostrar el DataFrame
print(df_original)

########################################### REPRESENTACION

plt.rcParams['figure.dpi'] = 900  # Ajusta el valor según tus necesidades
#RESOLUCIÓN PARA TODAS LAS FIGURAS

# Configurar el estilo de Seaborn
sns.set_style("whitegrid")

# Crear el gráfico de línea utilizando Seaborn
sns.lineplot(data=df_original, x=df_original.index, y='Electricity production')

# Establecer los ticks del eje x para mostrar solo algunos valores
ticks_to_show = df_original.index[::12]  # Mostrar cada 12 meses, por ejemplo
plt.xticks(ticks_to_show, rotation=45)  # Rotar los ticks para mayor legibilidad

# Añadir etiquetas y título
plt.xlabel('Fecha EN18-FEB24')
plt.ylabel('Valor')
plt.title('Producción de energía eléctrica mensual')

# Mostrar el gráfico
plt.tight_layout()  # Ajustar el diseño para evitar solapamientos
plt.show()


######################################## ACF Y PACF

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Visualizamos los gráficos ACF y PACF
plot_acf(df_original['Electricity production'])
#Autocorrelacion estacional cada 2,3 meses
plot_pacf(df_original['Electricity production'], lags=15)  # Ajusta el número de lags según tu preferencia

plt.show()

################################ PRUEBA DICKY FULLER ESTACIONARIEDAD

from statsmodels.tsa.stattools import adfuller

result = adfuller(df_original['Electricity production'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
#En la prueba de Dickey-Fuller, la hipótesis nula es que la serie temporal 
#tiene una raíz unitaria, lo que implica que no es estacionaria


################################### TRATAMIENTO DE LA ESTACIONARIEDAD
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Ruta del archivo Excel
ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'

# Leer el archivo Excel en un DataFrame
df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0, parse_dates=True)

# Configurar Seaborn para los gráficos
sns.set(style="whitegrid")

# Seleccionar solo algunos índices de fecha para mostrar en los ejes x
n = 10  # Mostrar solo cada n valores en el eje x
xticks_indices = df_original.index[::len(df_original.index) // n]

# Descomponer la serie temporal para extraer la componente estacional
decomposition = seasonal_decompose(df_original, period=12)  # Asumiendo una estacionalidad mensual

# DIFERENCIACION
df_diff = df_original.diff().dropna()

# TRANSFORMACION LOGARITMICA
df_log = np.log(df_original)

# SUAVIZADO
df_ewma = df_original.ewm(span=12).mean()

# Graficar todo
plt.figure(figsize=(15, 25))

# Diferenciación
plt.subplot(715)
sns.lineplot(data=df_diff, legend=False)
plt.title('Datos Diferenciados')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Fecha')
plt.ylabel('Valor')

# Transformación Logarítmica
plt.subplot(716)
sns.lineplot(data=df_log, legend=False)
plt.title('Datos Transformados (Logaritmo)')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Fecha')
plt.ylabel('Valor')

# Suavizado
plt.subplot(717)
sns.lineplot(data=df_ewma, legend=False)
plt.title('Datos Suavizados (EWMA)')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Fecha')
plt.ylabel('Valor')

plt.tight_layout()
plt.show()


############################################ PRUEBA DE ESTACIONARIEDAD


from statsmodels.tsa.stattools import adfuller

# Función para realizar la prueba ADF
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

# Aplicar la prueba ADF a cada serie transformada
adf_test(df_diff)
adf_test(df_log)
adf_test(df_ewma)


#La mejor serie es la diferenciada


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

######################### DACF Y PACF DE LA SERIE DIFERENCIADA
# Visualizamos los gráficos ACF y PACF
plot_acf(df_diff,lags=35)
#determinamos MA
plot_pacf(df_diff, lags=35)  # Ajusta el número de lags según tu preferencia
#determinamos AR
plt.show()



##################################### DESCOMPOSICIÓN DE LA SERIE ORIGINAL


import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Ruta del archivo Excel
ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'

# Leer el archivo Excel en un DataFrame
df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0, parse_dates=True)

# Descomponer la serie temporal para extraer la componente estacional
decomposition = seasonal_decompose(df_original, period=12)  # Asumiendo una estacionalidad mensual

# Configurar Seaborn para los gráficos
sns.set(style="whitegrid")

# Seleccionar solo algunos índices de fecha para mostrar en los ejes x
n = 10  # Mostrar solo cada n valores en el eje x
xticks_indices = df_original.index[::len(df_original.index) // n]

# Graficar la descomposición estacional con Seaborn
plt.figure(figsize=(10, 8))
plt.subplot(411)
sns.lineplot(data=df_original, legend=False)
plt.title('Original')
plt.xticks(xticks_indices)
plt.subplot(412)
sns.lineplot(data=decomposition.trend, legend=False)
plt.title('Tendencia')
plt.xticks(xticks_indices)
plt.subplot(413)
sns.lineplot(data=decomposition.seasonal, legend=False)
plt.title('Estacionalidad')
plt.xticks(xticks_indices)
plt.subplot(414)
sns.lineplot(data=decomposition.resid, legend=False)
plt.title('Residuo')
plt.xticks(xticks_indices)
plt.tight_layout()
plt.show()



################################################# AJUSTE DE MODELOS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Leer la serie original
ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'
df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0, parse_dates=True)

# Ajustar un modelo ARIMA
p = 2  # Orden del componente autoregresivo
d = 1  # Orden de diferenciación
q = 1  # Orden del componente de media móvil

model_arima = ARIMA(df_original, order=(p, d, q))
result_arima = model_arima.fit()

# Ajustar un modelo SARIMA
P = 1  # Orden del componente autoregresivo estacional
D = 1  # Orden de diferenciación estacional
Q = 1  # Orden del componente de media móvil estacional
m = 12  # Período estacional (asumimos estacionalidad mensual)

model_sarima = SARIMAX(df_original, order=(p, d, q), seasonal_order=(P, D, Q, m))
result_sarima = model_sarima.fit()

# Graficar la serie original y los modelos ARIMA y SARIMA ajustados
plt.figure(figsize=(12, 6))

# Plot ARIMA
plt.subplot(1, 2, 1)
sns.lineplot(data=df_original.iloc[:], color='blue')
sns.lineplot(data=result_arima.fittedvalues.iloc[2:], label='Ajuste ARIMA', color='orange')
plt.title('Serie Original y Ajuste ARIMA')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Representar solo 10 valores en el eje x
plt.legend()

# Plot SARIMA
plt.subplot(1, 2, 2)
sns.lineplot(data=df_original.iloc[:], color='blue')
sns.lineplot(data=result_sarima.fittedvalues.iloc[2:], label='Ajuste SARIMA', color='green')
plt.title('Serie Original y Ajuste SARIMA')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Representar solo 10 valores en el eje x
plt.legend()

plt.tight_layout()
plt.show()



################################################ PREDICCIONES ARIMA
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

# Leer la serie original
ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'
df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0, parse_dates=True)

# Ajustar un modelo ARIMA
p = 1  # Orden del componente autoregresivo
d = 1  # Orden de diferenciación
q = 1  # Orden del componente de media móvil

# Dividir los datos en entrenamiento y prueba
train_size = int(len(df_original) * 0.8)  # Usaremos el 80% de los datos para entrenamiento
train, test = df_original.iloc[:train_size], df_original.iloc[train_size:]

# Ajustar el modelo ARIMA solo a los datos de entrenamiento
model_arima = ARIMA(train, order=(p, d, q))
result_arima = model_arima.fit()
    
 
# Realizar predicciones en los datos de prueba
forecast_values = result_arima.forecast(steps=len(test))
 
    
 
# Configurar estilo Seaborn
sns.set(style="whitegrid")

# Plotear los datos de entrenamiento
plt.figure(figsize=(14, 6))

sns.lineplot(data=train, color='orange')

# Plotear el modelo ajustado
sns.lineplot(data=result_arima.fittedvalues.iloc[2:], color='red')

# Plotear los datos de prueba
sns.lineplot(data=test, color='blue')

# Plotear las predicciones
sns.lineplot(data=forecast_values, color='green')

plt.title('Datos de entrenamiento, modelo ajustado, datos de prueba y predicciones')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.xticks(rotation=65)
plt.tight_layout()
plt.legend().remove()
plt.show()




############################################ PREDICCIONES ARIMA AJUSTE AIC

import itertools
import numpy as np

# Definir los rangos para los valores de p, d y q
p_range = range(0, 4)
d_range = range(0, 2)
q_range = range(0, 4)

# Calcular todas las combinaciones posibles de p, d y q
pdq_combinations = list(itertools.product(p_range, d_range, q_range))

# Inicializar variables para el mejor AIC y los mejores parámetros
best_aic = np.inf
best_params = None

# Dividir los datos en entrenamiento y prueba
train_size = int(len(df_original) * 0.8)  # Usaremos el 80% de los datos para entrenamiento
train, test = df_original.iloc[:train_size], df_original.iloc[train_size:]

# Iterar sobre todas las combinaciones de p, d y q
for pdq in pdq_combinations:
    try:
        # Ajustar el modelo ARIMA con la combinación actual de p, d y q
        model = ARIMA(train, order=pdq)
        result = model.fit()
        
        # Calcular el AIC para el modelo actual
        aic = result.aic
        
        # Actualizar los mejores parámetros si se encuentra un AIC más bajo
        if aic < best_aic:
            best_aic = aic
            best_params = pdq
    except:
        continue

# Ajustar el modelo ARIMA con los mejores parámetros encontrados
best_model_arima = ARIMA(train, order=best_params)
best_result_arima = best_model_arima.fit()

# Imprimir los mejores parámetros y el AIC correspondiente
print("Mejores parámetros:", best_params)
print("AIC correspondiente:", best_aic)

 
# Realizar predicciones en los datos de prueba
forecast_values2 = best_result_arima.forecast(steps=len(test))
 
# Configurar estilo Seaborn
sns.set(style="whitegrid")

# Plotear los datos de entrenamiento
plt.figure(figsize=(14, 6))

sns.lineplot(data=train, color='orange')

# Plotear el modelo ajustado
sns.lineplot(data=best_result_arima.fittedvalues.iloc[2:], color='red')

# Plotear los datos de prueba
sns.lineplot(data=test, color='blue')

# Plotear las predicciones
sns.lineplot(data=forecast_values2, color='green')

plt.title('Datos de entrenamiento, modelo ajustado, datos de prueba y predicciones')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.xticks(rotation=65)
plt.tight_layout()
plt.legend().remove()
plt.show()


#################################### VALIDACIÓN DEL MODELO MEDIANTE CORRELOGRAMA Y RESIDUOS

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Obtener los residuos del modelo SARIMA
residuals = best_result_arima.resid

# Crear el correlograma de los residuos
plt.figure(figsize=(10, 3))

# Autocorrelación
plt.subplot(1, 2, 1)
plot_acf(residuals, ax=plt.gca(), lags=40)
plt.title('Correlograma de autocorrelación de los residuos')

# Autocorrelación parcial
plt.subplot(1, 2, 2)
plot_pacf(residuals, ax=plt.gca(), lags=29)
plt.title('Correlograma parcial de autocorrelación de los residuos')

plt.tight_layout()
plt.show()


residuos = best_result_arima.resid


import matplotlib.pyplot as plt

# Gráfico de dispersión de los residuos
plt.scatter(range(len(residuos)), residuos)
plt.title('Gráfico de dispersión de los residuos')
plt.xlabel('Índice de observación')
plt.ylabel('Residuo')
plt.show()

# Histograma de los residuos
plt.hist(residuos, bins=20)
plt.title('Histograma de los residuos')
plt.xlabel('Residuo')
plt.ylabel('Frecuencia')
plt.show()

# Gráfico de autocorrelación de los residuos
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(residuos, lags=10)
plt.title('Gráfico de autocorrelación de los residuos')
plt.xlabel('Lag')
plt.ylabel('Autocorrelación')
plt.show()




################################################## ARIMA ESTACIONAL
import itertools

# Definir los rangos de órdenes a probar
p = range(0, 4)  # Rango de órdenes AR
d = range(0, 2)  # Rango de órdenes de diferenciación
q = range(0, 4)  # Rango de órdenes MA
P = range(0, 3)  # Rango de órdenes estacionales AR
D = range(0, 2)  # Rango de órdenes de diferenciación estacional
Q = range(0, 3)  # Rango de órdenes estacionales MA
s = [3]  # Estacionalidad trimestral

# Crear una lista de todas las combinaciones de órdenes
orders = list(itertools.product(p, d, q))
seasonal_orders = list(itertools.product(P, D, Q, s))

# Definir variables para almacenar el mejor modelo y su AIC
best_aic = float("inf")
best_model = None
best_order = None
best_seasonal_order = None

# Iterar sobre todas las combinaciones de órdenes
for order in orders:
    for seasonal_order in seasonal_orders:
        try:
            # Ajustar el modelo SARIMA con los órdenes actuales
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            result = model.fit()
            
            # Calcular el AIC del modelo actual
            aic = result.aic
            
            # Actualizar el mejor modelo y su AIC si se encuentra uno con un AIC menor
            if aic < best_aic:
                best_aic = aic
                best_model = result
                best_order = order
                best_seasonal_order = seasonal_order
        except:
            continue

# Imprimir los mejores parámetros y el AIC correspondiente
print("Mejores parámetros:", best_order, best_seasonal_order)
print("AIC correspondiente:", best_aic)




import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

# Leer la serie original
ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'
df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0, parse_dates=True)


# Dividir los datos en entrenamiento y prueba
train_size = int(len(df_original) * 0.8)  # Usaremos el 80% de los datos para entrenamiento
train, test = df_original.iloc[:train_size], df_original.iloc[train_size:]

# Ajustar el modelo SARIMA con los mejores parámetros encontrados
best_model_sarima = SARIMAX(train, order=(3,1,3), seasonal_order=(1,1,1,12))  # Estacionalidad trimestral
best_result_sarima = best_model_sarima.fit()

# Imprimir los mejores parámetros y el AIC correspondiente
print(" parámetros:", best_result_sarima.summary().tables[0])
print("AIC correspondiente:", best_result_sarima.aic)

# Realizar predicciones en los datos de prueba
forecast_values_sarima = best_result_sarima.forecast(steps=len(test))

# Configurar estilo Seaborn
sns.set(style="whitegrid")

# Plotear los datos de entrenamiento
plt.figure(figsize=(14, 6))
sns.lineplot(data=train, color='orange')

# Plotear el modelo ajustado
sns.lineplot(data=best_result_sarima.fittedvalues.iloc[2:], color='red')

# Plotear los datos de prueba
sns.lineplot(data=test, color='blue')

# Plotear las predicciones
sns.lineplot(data=forecast_values_sarima, color='green')

plt.title('Datos de entrenamiento, modelo ajustado, datos de prueba y predicciones SARIMA order=(3,1,3), seasonal_order=(1,1,1,12))')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.xticks(rotation=65)
plt.tight_layout()
plt.legend().remove()
plt.show()


############################################### RESIDUOS ANÄLISIS

import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf  # Importar plot_acf

# Leer la serie original
ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'
df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0, parse_dates=True)

# Dividir los datos en entrenamiento y prueba
train_size = int(len(df_original) * 0.8)  # Usaremos el 80% de los datos para entrenamiento
train, test = df_original.iloc[:train_size], df_original.iloc[train_size:]

# Ajustar el modelo SARIMA con los mejores parámetros encontrados
best_model_sarima = SARIMAX(train, order=(3,1,3), seasonal_order=(1,1,1,12))  # Estacionalidad trimestral
best_result_sarima = best_model_sarima.fit()

# Obtener los residuos del modelo
residuals = best_result_sarima.resid

# Configurar estilo Seaborn
sns.set(style="whitegrid")

# Crear subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# Serie temporal de los residuos
sns.lineplot(data=residuals[1:], color='blue', ax=axes[0])
axes[0].set_title('Serie Temporal de Residuos del Modelo SARIMA')
axes[0].set_xlabel('Fecha')
axes[0].set_ylabel('Residuos')
axes[0].tick_params(rotation=65)

# Gráfico de autocorrelación de los residuos
plot_acf(residuals, lags=30, alpha=0.05, ax=axes[1])
axes[1].set_title('Gráfico de Autocorrelación de Residuos')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Autocorrelación')

# Ajustar el diseño y mostrar el plot
plt.tight_layout()
plt.show()



################################################## PREDICCIONES ETS ADITIVO

#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# Ruta del archivo Excel
#ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'

# Leer el archivo Excel en un DataFrame
#df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0)

# Seleccionar la serie temporal deseada del DataFrame (cambiar 'nombre_columna' por el nombre real de la columna)
#serie_temporal = df_original['Electricity production']

# Dividir los datos en 80% para entrenamiento y 20% para prueba
#train_size = int(len(serie_temporal) * 0.8)
#train_data = serie_temporal[:train_size]
#test_data = serie_temporal[train_size:]

# Ajustar el modelo ETS al 80% de los datos
#model = ETSModel(train_data, error='add', trend='add', seasonal='add', seasonal_periods=12)  # Ajusta los parámetros según tu caso
#fit = model.fit()

# Predecir valores para el 20% restante
#forecast = fit.forecast(steps=len(test_data))  


# Graficar resultados utilizando Seaborn y ajustando los valores del eje x
#sns.set(style="whitegrid")
#plt.figure(figsize=(10, 6))
#sns.lineplot(data=train_data, label='Entrenamiento')
#sns.lineplot(data=test_data, label='Prueba')
#sns.lineplot(data=fit.fittedvalues, label='Ajuste')
#sns.lineplot(data=forecast, label='Predicción')
#plt.title('Alisado exponencial ETS aditivo')
#plt.xlabel('Fecha')
#plt.ylabel('Valor')
#plt.xticks(rotation=55)  # Rotar los valores del eje x
#plt.xticks(ticks=plt.xticks()[0][::2], labels=df_original.index[train_size::2])  # Separar los valores del eje x cada dos valores para los datos de prueba
#plt.legend()
#plt.show()



############################################ PREDICCIONES ETS MULTIPLICATIVO

#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# Ruta del archivo Excel
#ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'

# Leer el archivo Excel en un DataFrame
#df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0)

# Seleccionar la serie temporal deseada del DataFrame (cambiar 'nombre_columna' por el nombre real de la columna)
#serie_temporal = df_original['Electricity production']

# Dividir los datos en 80% para entrenamiento y 20% para prueba
#train_size = int(len(serie_temporal) * 0.8)
#train_data = serie_temporal[:train_size]
#test_data = serie_temporal[train_size:]

# Ajustar el modelo ETS multiplicativo al 80% de los datos
#model = ETSModel(train_data, error='mul', trend='mul', seasonal='mul', seasonal_periods=12)  # Ajusta los parámetros según tu caso
#fit = model.fit()

# Predecir valores para el 20% restante
#forecast = fit.forecast(steps=len(test_data))  



# Graficar resultados utilizando Seaborn y ajustando los valores del eje x
#sns.set(style="whitegrid")
#plt.figure(figsize=(10, 6))
#sns.lineplot(data=train_data, label='Entrenamiento')
#sns.lineplot(data=test_data, label='Prueba')
#sns.lineplot(data=fit.fittedvalues, label='Ajuste')
#sns.lineplot(data=forecast, label='Predicción')
#plt.title('Alisado exponencial utilizando modelos ETS (Multiplicativo)')
#plt.xlabel('Fecha')
#plt.ylabel('Valor')
#plt.xticks(rotation=55)  # Rotar los valores del eje x
#plt.xticks(ticks=plt.xticks()[0][::3], labels=df_original.index[train_size::3])  # Separar los valores del eje x cada tres valores para los datos de prueba
#plt.legend()
#plt.show()




########################################### PREDICCIONES ETS ADITIVO SIN TENDENCIA


#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# Ruta del archivo Excel
#ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'

# Leer el archivo Excel en un DataFrame
#df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0)

# Seleccionar la serie temporal deseada del DataFrame (cambiar 'Electricity production' por el nombre real de la columna)
#serie_temporal = df_original['Electricity production']

# Dividir los datos en 80% para entrenamiento y 20% para prueba
#train_size = int(len(serie_temporal) * 0.8)
#train_data = serie_temporal[:train_size]
#test_data = serie_temporal[train_size:]

# Ajustar el modelo ETS al 80% de los datos sin componente de tendencia
#model = ETSModel(train_data, error='add', trend=None , seasonal='add', seasonal_periods=12)  
#fit = model.fit()

# Predecir valores para el 20% restante
#forecast = fit.forecast(steps=len(test_data))  

# Graficar resultados utilizando Seaborn y ajustando los valores del eje x
#sns.set(style="whitegrid")
#plt.figure(figsize=(10, 6))
#sns.lineplot(data=train_data, label='Entrenamiento')
#sns.lineplot(data=test_data, label='Prueba')
#sns.lineplot(data=fit.fittedvalues, label='Ajuste')
#sns.lineplot(data=forecast, label='Predicción')
#plt.title('Alisado exponencial ETS sin tendencia')
#plt.xlabel('Fecha')
#plt.ylabel('Valor')
#plt.xticks(rotation=55)  # Rotar los valores del eje x
#plt.xticks(ticks=plt.xticks()[0][::2], labels=df_original.index[train_size::2])  # Separar los valores del eje x cada dos valores para los datos de prueba
#plt.legend()
#plt.show()





########################################## PREDICCIONES ETS ADITIVO-MULTIPLICATIVO

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# Ruta del archivo Excel
ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'

# Leer el archivo Excel en un DataFrame
df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0)

# Seleccionar la serie temporal deseada del DataFrame (cambiar 'nombre_columna' por el nombre real de la columna)
serie_temporal = df_original['Electricity production']

# Dividir los datos en 80% para entrenamiento y 20% para prueba
train_size = int(len(serie_temporal) * 0.8)
train_data = serie_temporal[:train_size]
test_data = serie_temporal[train_size:]

# Ajustar el modelo ETS multiplicativo al 80% de los datos
model = ETSModel(train_data, error='add', trend='add', seasonal='mul', seasonal_periods=12)  # Ajusta los parámetros según tu caso
fit = model.fit()

# Predecir valores para el 20% restante
forecast = fit.forecast(steps=len(test_data))  



# Graficar resultados utilizando Seaborn y ajustando los valores del eje x
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=train_data, label='Entrenamiento')
sns.lineplot(data=test_data, label='Prueba')
sns.lineplot(data=fit.fittedvalues, label='Ajuste')
sns.lineplot(data=forecast, label='Predicción')
plt.title('Alisado exponencial utilizando modelos ETS (Multiplicativo)')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.xticks(rotation=55)  # Rotar los valores del eje x
plt.xticks(ticks=plt.xticks()[0][::3], labels=df_original.index[train_size::3])  # Separar los valores del eje x cada tres valores para los datos de prueba
plt.legend()
plt.show()



################################################# PREDICCIONES ETS MULTIPLICATIVO-ADITIVO

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# Ruta del archivo Excel
ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'

# Leer el archivo Excel en un DataFrame
df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0)

# Seleccionar la serie temporal deseada del DataFrame (cambiar 'nombre_columna' por el nombre real de la columna)
serie_temporal = df_original['Electricity production']

# Dividir los datos en 80% para entrenamiento y 20% para prueba
train_size = int(len(serie_temporal) * 0.8)
train_data = serie_temporal[:train_size]
test_data = serie_temporal[train_size:]

# Ajustar el modelo ETS multiplicativo al 80% de los datos
model = ETSModel(train_data, error='mul', trend='add', seasonal='mul', seasonal_periods=12)  # Ajusta los parámetros según tu caso
fit = model.fit()

# Predecir valores para el 20% restante
forecast = fit.forecast(steps=len(test_data))  



# Graficar resultados utilizando Seaborn y ajustando los valores del eje x
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=train_data, label='Entrenamiento')
sns.lineplot(data=test_data, label='Prueba')
sns.lineplot(data=fit.fittedvalues, label='Ajuste')
sns.lineplot(data=forecast, label='Predicción')
plt.title('Alisado exponencial utilizando modelos ETS (Multiplicativo)')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.xticks(rotation=55)  # Rotar los valores del eje x
plt.xticks(ticks=plt.xticks()[0][::3], labels=df_original.index[train_size::3])  # Separar los valores del eje x cada tres valores para los datos de prueba
plt.legend()
plt.show()



############################################################ ALISADO EXPONENCIAL SIMPLE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Leer la serie original
ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'
df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0, parse_dates=True)

# Dividir los datos en entrenamiento y prueba
train_size = int(len(df_original) * 0.8)  # Usaremos el 80% de los datos para entrenamiento
train, test = df_original.iloc[:train_size], df_original.iloc[train_size:]

# Ajustar el modelo de Suavizado Exponencial Simple (SES)
model_ses = SimpleExpSmoothing(train)
model_ses_result = model_ses.fit()

# Realizar predicciones para el 20% final de la muestra
forecast_values_ses = model_ses_result.forecast(len(test))

# Configurar estilo Seaborn
sns.set(style="whitegrid")

# Plotear los datos de entrenamiento
plt.figure(figsize=(14, 6))
sns.lineplot(data=train, color='orange')

# Plotear los datos de prueba
sns.lineplot(data=test, color='blue')

# Plotear las predicciones del modelo SES
sns.lineplot(data=forecast_values_ses, color='green')

plt.title('Datos de entrenamiento, datos de prueba y predicciones del Modelo SES')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.xticks(rotation=65)
plt.tight_layout()
plt.legend(['Entrenamiento', 'Prueba', 'Predicciones'])
plt.show()





#################################################### ALISADO EXPONENCIAL DOBLE HOLT

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Leer la serie original
ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'
df_original = pd.read_excel(ruta_archivo, sheet_name="datos", index_col=0, parse_dates=True)

# Dividir los datos en entrenamiento y prueba
train_size = int(len(df_original) * 0.8)  # Usaremos el 80% de los datos para entrenamiento
train, test = df_original.iloc[:train_size], df_original.iloc[train_size:]

# Ajustar el modelo de Alisado Exponencial Doble de Holt
model_holt = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
model_holt_result = model_holt.fit()

# Realizar predicciones para el 20% final de la muestra
forecast_values_holt = model_holt_result.forecast(len(test))

# Configurar estilo Seaborn
sns.set(style="whitegrid")

# Plotear los datos de entrenamiento
plt.figure(figsize=(14, 6))
sns.lineplot(data=train, color='orange')

# Plotear los datos de prueba
sns.lineplot(data=test, color='blue')

# Plotear las predicciones del modelo de Alisado Exponencial Doble de Holt
sns.lineplot(data=forecast_values_holt, color='green')

plt.title('Datos de entrenamiento, datos de prueba y predicciones del Modelo de Holt-Winters')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.xticks(rotation=65)
plt.tight_layout()
plt.legend(['Entrenamiento', 'Prueba', 'Predicciones'])
plt.show()


################################################################ ANÁLISIS ESPECTRAL

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Función personalizada para convertir el formato de fecha "ENE 2018" a una fecha de Python
def parse_date(date_string):
    month_str, year_str = date_string.split()
    month_dict = {
        'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04', 'MAY': '05', 'JUN': '06',
        'JUL': '07', 'AGO': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12'
    }
    return pd.to_datetime(f"{year_str}-{month_dict[month_str]}")

# Paso 1: Leer los datos desde el archivo Excel con el parser personalizado
ruta_archivo = r'C:\Users\34653\Desktop\UNED\segundo cuatrimestre\analisis cuantitativo de ciclo ENERGIA\datos_mensuales.xlsx'
datos = pd.read_excel(ruta_archivo, parse_dates=['fecha'], date_parser=parse_date)

# Convertir los datos a una serie temporal
datos.set_index('fecha', inplace=True)

# Paso 2: Aplicar el filtro HP para obtener la tendencia
cycle, trend = sm.tsa.filters.hpfilter(datos['Electricity production'], lamb=1600)

# Paso 3: Calcular la serie libre de tendencia
serie_libre_tendencia = datos['Electricity production'] - trend

# Configurar el estilo de seaborn
sns.set(style="whitegrid")

# Paso 4: Representar las series originales y la tendencia en un gráfico usando Seaborn
plt.figure(figsize=(12, 6))
sns.lineplot(data=datos['Electricity production'], label='Serie Original')
sns.lineplot(data=trend, label='Tendencia', linestyle='--', color='red')
plt.title('Serie Original y Tendencia')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.legend()
plt.show()

# Paso 5: Representar la serie libre de tendencia en otro gráfico usando Seaborn
plt.figure(figsize=(12, 6))
sns.lineplot(data=serie_libre_tendencia, label='Serie Libre de Tendencia', color='green')
plt.title('Serie Libre de Tendencia')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.legend()
plt.show()


####################################################### PERIODIGRAMA DE LA SERIE SIN TENDENCIA
from scipy.signal import periodogram

# Calcular el periodograma de la serie libre de tendencia
frecuencias, periodograma = periodogram(serie_libre_tendencia)

# Graficar el periodograma
plt.figure(figsize=(10, 6))
plt.plot(frecuencias, periodograma)
plt.title('Periodograma de la Serie Libre de Tendencia (Después del Filtro HP)')
plt.xlabel('Frecuencia')
plt.ylabel('Densidad Espectral de Potencia (PSD)')
plt.grid(True)
plt.show()


############################################################ análisis de picos de frecuencia
from scipy.signal import find_peaks

# Frecuencia esperada de la estacionalidad (por ejemplo, frecuencia anual)
frecuencia_esperada = 1 / 4  # Para un componente estacional trimestral

# Encuentra los índices de los picos más prominentes en el periodograma
picos_indices, _ = find_peaks(periodograma)

# Calcula las frecuencias correspondientes a estos picos
frecuencias_picos = frecuencias[picos_indices]

# Define un margen de tolerancia alrededor de la frecuencia esperada
margen_tolerancia = 0.02  # Por ejemplo, un margen del 2%

# Encuentra los picos que coinciden con la frecuencia esperada dentro del margen de tolerancia
picos_coincidentes = frecuencias_picos[
    (frecuencias_picos > frecuencia_esperada - margen_tolerancia) &
    (frecuencias_picos < frecuencia_esperada + margen_tolerancia)
]

print("Picos que coinciden con la frecuencia esperada:", picos_coincidentes)




################################################## transformada rápida de fourier

import numpy as np
import matplotlib.pyplot as plt

# Calcula la serie libre de tendencia
serie_libre_tendencia = datos['Electricity production'] - trend

# Calcula el periodograma de la serie libre de tendencia
psd = np.abs(np.fft.fft(serie_libre_tendencia))**2

# Frecuencias correspondientes
n = len(serie_libre_tendencia)
frecuencias = np.fft.fftfreq(n, 1)  # Frecuencias en Hz

# Filtra las frecuencias negativas y sus PSD correspondientes
mascara = frecuencias > 0
frecuencias = frecuencias[mascara]
psd = psd[mascara]

# Visualiza el periodograma
plt.figure(figsize=(10, 5))
plt.plot(frecuencias, psd, color="green")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad Espectral de Potencia (PSD)')
plt.title('Periodograma de la Serie Temporal sin Tendencia')
plt.grid(True)
plt.show()

######################################################### transformada rápida de fourier normalizada


# Normalizar el periodograma calculado por la FFT
psd_normalizado = psd / (n**2)

# Visualizar el periodograma normalizado
plt.figure(figsize=(10, 5))
plt.plot(frecuencias, psd_normalizado, color="red")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad Espectral de Potencia (PSD)')
plt.title('Periodograma Normalizado de la Serie Temporal sin Tendencia')
plt.grid(True)
plt.show()


################################################################### analisis sesudo

import numpy as np
import matplotlib.pyplot as plt

# Calcular el periodograma
frecuencias, periodograma = periodogram(serie_libre_tendencia)

# Normalizar el periodograma para obtener la contribución proporcional a la varianza
contribucion_proporcional = periodograma / np.sum(periodograma)

# Encontrar el pico principal
indice_pico_principal = np.argmax(contribucion_proporcional)
frecuencia_pico_principal = frecuencias[indice_pico_principal]
periodo_pico_principal = 1 / frecuencia_pico_principal

# Visualizar el periodograma
plt.figure(figsize=(10, 6))
plt.plot(frecuencias, contribucion_proporcional, label='Contribución proporcional')
plt.axvline(frecuencia_pico_principal, color='r', linestyle='--', label='Pico principal')
plt.xlabel('Frecuencia')
plt.ylabel('Contribución proporcional a la varianza')
plt.title('Contribución proporcional a la varianza de cada frecuencia')
plt.grid(True)
plt.legend()

# Crear el eje superior con los periodos correspondientes
eje_superior = plt.twiny()
eje_superior.set_xlim(0, len(frecuencias))
eje_superior.set_xticks(np.arange(0, len(frecuencias), step=10))
eje_superior.set_xticklabels([f'{1/f:.1f}' for f in 1/frecuencias[np.arange(0, len(frecuencias), step=10)]])
eje_superior.set_xlabel('Periodo (meses)')

plt.show()

# Imprimir la tabla con los valores del periodograma
print("Tabla con los valores del periodograma:")
print("Frecuencia\tContribución proporcional")
for f, contribucion in zip(frecuencias, contribucion_proporcional):
    print(f"{f:.4f}\t\t{contribucion:.4f}")

# Imprimir información sobre el pico principal
print("\nPico principal:")
print(f"Frecuencia: {frecuencia_pico_principal:.4f}")
print(f"Periodo correspondiente: {periodo_pico_principal:.1f} meses")

#################################################### BANDAS DE SIGNIFICACION AL 95%

import numpy as np
import matplotlib.pyplot as plt

# Calcular el periodograma
frecuencias, periodograma = periodogram(serie_libre_tendencia)

# Normalizar el periodograma para obtener la contribución proporcional a la varianza
contribucion_proporcional = periodograma / np.sum(periodograma)

# Encontrar el pico principal
indice_pico_principal = np.argmax(contribucion_proporcional)
frecuencia_pico_principal = frecuencias[indice_pico_principal]
periodo_pico_principal = 1 / frecuencia_pico_principal

# Calcular las bandas de significación al 95%
n = len(serie_libre_tendencia)
banda_superior = 1.36 / np.sqrt(n)
banda_inferior = -1.36 / np.sqrt(n)

# Visualizar el periodograma con las bandas de significación
plt.figure(figsize=(10, 6))
plt.plot(frecuencias, contribucion_proporcional, label='Contribución proporcional')
plt.axvline(frecuencia_pico_principal, color='r', linestyle='--', label='Pico principal')
plt.axhline(banda_superior, color='g', linestyle=':', label='Banda de significación (95%)')
plt.axhline(banda_inferior, color='g', linestyle=':')
plt.xlabel('Frecuencia')
plt.ylabel('Contribución proporcional a la varianza')
plt.title('Contribución proporcional a la varianza de cada frecuencia')
plt.grid(True)
plt.legend()

# Crear el eje superior con los periodos correspondientes
eje_superior = plt.twiny()
eje_superior.set_xlim(0, len(frecuencias))
eje_superior.set_xticks(np.arange(0, len(frecuencias), step=10))
eje_superior.set_xticklabels([f'{1/f:.1f}' for f in 1/frecuencias[np.arange(0, len(frecuencias), step=10)]])
eje_superior.set_xlabel('Periodo (meses)')

plt.show()

# Imprimir la tabla con los valores del periodograma
print("Tabla con los valores del periodograma:")
print("Frecuencia\tContribución proporcional")
for f, contribucion in zip(frecuencias, contribucion_proporcional):
    print(f"{f:.4f}\t\t{contribucion:.4f}")

# Imprimir información sobre el pico principal
print("\nPico principal:")
print(f"Frecuencia: {frecuencia_pico_principal:.4f}")
print(f"Periodo correspondiente: {periodo_pico_principal:.1f} meses")


import numpy as np
import matplotlib.pyplot as plt

# Calcular el periodograma
frecuencias, periodograma = periodogram(serie_libre_tendencia)

# Normalizar el periodograma para obtener la contribución proporcional a la varianza
contribucion_proporcional = periodograma / np.sum(periodograma)

# Encontrar el pico principal
indice_pico_principal = np.argmax(contribucion_proporcional)
frecuencia_pico_principal = frecuencias[indice_pico_principal]
periodo_pico_principal = 1 / frecuencia_pico_principal

# Calcular las bandas de significación al 99%
n = len(serie_libre_tendencia)
banda_superior_99 = 0.99 / 2
banda_inferior_99 = 1 - (0.99 / 2)

# Visualizar el periodograma con las bandas de significación
plt.figure(figsize=(10, 6))
plt.plot(frecuencias, contribucion_proporcional, label='Contribución proporcional')
plt.axvline(frecuencia_pico_principal, color='r', linestyle='--', label='Pico principal')
plt.axhline(banda_superior_99, color='g', linestyle=':', label='Banda de significación (99%)')
plt.axhline(banda_inferior_99, color='g', linestyle=':')
plt.xlabel('Frecuencia')
plt.ylabel('Contribución proporcional a la varianza')
plt.title('Contribución proporcional a la varianza de cada frecuencia')
plt.grid(True)
plt.legend()

# Crear el eje superior con los periodos correspondientes
eje_superior = plt.twiny()
eje_superior.set_xlim(0, len(frecuencias))
eje_superior.set_xticks(np.arange(0, len(frecuencias), step=10))
eje_superior.set_xticklabels([f'{1/f:.1f}' for f in 1/frecuencias[np.arange(0, len(frecuencias), step=10)]])
eje_superior.set_xlabel('Periodo (meses)')

plt.show()

# Imprimir la tabla con los valores del periodograma
print("Tabla con los valores del periodograma:")
print("Frecuencia\tContribución proporcional")
for f, contribucion in zip(frecuencias, contribucion_proporcional):
    print(f"{f:.4f}\t\t{contribucion:.4f}")

# Imprimir información sobre el pico principal
print("\nPico principal:")
print(f"Frecuencia: {frecuencia_pico_principal:.4f}")
print(f"Periodo correspondiente: {periodo_pico_principal:.1f} meses")


########################################################################### prediccion

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

# Supongamos que ya tienes serie_libre_tendencia calculado
# Definir los datos
datos['const'] = 1  # Añadir una columna constante para el término constante
datos['cos_23t'] = np.cos(23 * np.arange(1, len(datos) + 1))
datos['sin_23t'] = np.sin(23 * np.arange(1, len(datos) + 1))

# Definir la variable dependiente utilizando la serie libre de tendencia calculada previamente
y = datos['Electricity production'] - trend

# Definir las variables independientes
X = datos[['const', 'cos_23t', 'sin_23t']]

# Ajustar el modelo de regresión lineal
modelo = sm.OLS(y, X).fit()

# Imprimir resumen del modelo
print(modelo.summary())

# Calcular la contribución del armónico número 23
contribucion_armónico_23 = modelo.params['cos_23t'] + modelo.params['sin_23t']
print("Contribución del armónico número 23:", contribucion_armónico_23)

# Hacer predicciones hacia adelante (por ejemplo, para los próximos 5 períodos)
nuevos_periodos = np.arange(len(datos) + 1, len(datos) + 6)
X_nuevos = pd.DataFrame({'const': 1, 'cos_23t': np.cos(23 * nuevos_periodos), 'sin_23t': np.sin(23 * nuevos_periodos)})
predicciones = modelo.predict(X_nuevos)

# Visualizar la predicción junto con los valores anteriores de la serie libre de tendencia
sns.lineplot(data=pd.concat([y, pd.Series(predicciones, index=nuevos_periodos)]), label='Valores anteriores y predicciones')
plt.xlabel('Periodo')
plt.ylabel('Valor')
plt.title('Predicción de la serie libre de tendencia con el armónico número 23')
plt.show()


import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

# Supongamos que ya tienes serie_libre_tendencia calculado
# Definir los datos
datos['const'] = 1  # Añadir una columna constante para el término constante
datos['cos_23t'] = np.cos(23 * np.arange(1, len(datos) + 1))
datos['sin_23t'] = np.sin(23 * np.arange(1, len(datos) + 1))

# Definir la variable dependiente utilizando la serie libre de tendencia calculada previamente
y = datos['Electricity production'] - trend

# Definir las variables independientes
X = datos[['const', 'cos_23t', 'sin_23t']]

# Ajustar el modelo de regresión lineal
modelo = sm.OLS(y, X).fit()

# Imprimir resumen del modelo
print(modelo.summary())

# Calcular la contribución del armónico número 23
contribucion_armónico_23 = modelo.params['cos_23t'] + modelo.params['sin_23t']
print("Contribución del armónico número 23:", contribucion_armónico_23)

# Hacer predicciones hacia adelante (por ejemplo, para los próximos 5 períodos)
nuevos_periodos = np.arange(len(datos) + 1, len(datos) + 6)
X_nuevos = pd.DataFrame({'const': 1, 'cos_23t': np.cos(23 * nuevos_periodos), 'sin_23t': np.sin(23 * nuevos_periodos)})
predicciones = modelo.predict(X_nuevos)

# Visualizar la predicción junto con los valores anteriores de la serie libre de tendencia
sns.lineplot(data=pd.concat([y, pd.Series(predicciones, index=nuevos_periodos)]), label='Valores anteriores y predicciones', color='green')
plt.xlabel('Periodo')
plt.ylabel('Valor')
plt.title('Predicción de la serie libre de tendencia con el armónico número 23')
plt.show()



import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Supongamos que ya tienes serie_libre_tendencia calculado
# Definir los datos
datos['const'] = 1  # Añadir una columna constante para el término constante
datos['cos_23t'] = np.cos(23 * np.arange(1, len(datos) + 1))
datos['sin_23t'] = np.sin(23 * np.arange(1, len(datos) + 1))

# Definir la variable dependiente utilizando la serie libre de tendencia calculada previamente
y = datos['Electricity production'] - trend

# Determinar el número de puntos correspondientes al 20% final de los datos
n_prediccion = int(len(datos) * 0.2)

# Definir los datos de entrenamiento y de prueba
datos_entrenamiento = datos.iloc[:-n_prediccion]
datos_prueba = datos.iloc[-n_prediccion:]

# Definir las variables independientes para el entrenamiento y la prueba
X_entrenamiento = datos_entrenamiento[['const', 'cos_23t', 'sin_23t']]
X_prueba = datos_prueba[['const', 'cos_23t', 'sin_23t']]

# Definir la variable dependiente para el entrenamiento y la prueba
y_entrenamiento = y.iloc[:-n_prediccion]
y_prueba = y.iloc[-n_prediccion:]

# Ajustar el modelo de regresión lineal con los datos de entrenamiento
modelo = sm.OLS(y_entrenamiento, X_entrenamiento).fit()

# Hacer predicciones para los datos de prueba
predicciones = modelo.predict(X_prueba)

# Visualizar las predicciones junto con los valores reales
plt.figure(figsize=(10, 6))
sns.lineplot(data=y_prueba, label='Valores reales')
sns.lineplot(x=datos_prueba.index, y=predicciones, label='Predicciones')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación entre valores reales y predicciones')
plt.legend()
plt.show()


import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Supongamos que ya tienes serie_libre_tendencia calculado
# Definir los datos
datos['const'] = 1  # Añadir una columna constante para el término constante
datos['cos_23t'] = np.cos(23 * np.arange(1, len(datos) + 1))
datos['sin_23t'] = np.sin(23 * np.arange(1, len(datos) + 1))

# Definir la variable dependiente utilizando la serie libre de tendencia calculada previamente
y = datos['Electricity production'] - trend

# Determinar el número de puntos correspondientes al 20% final de los datos
n_prediccion = int(len(datos) * 0.2)

# Definir los datos de entrenamiento y de prueba
datos_entrenamiento = datos.iloc[:-n_prediccion]
datos_prueba = datos.iloc[-n_prediccion:]

# Definir las variables independientes para el entrenamiento y la prueba
X_entrenamiento = datos_entrenamiento[['const', 'cos_23t', 'sin_23t']]
X_prueba = datos_prueba[['const', 'cos_23t', 'sin_23t']]

# Definir la variable dependiente para el entrenamiento y la prueba
y_entrenamiento = y.iloc[:-n_prediccion]
y_prueba = y.iloc[-n_prediccion:]

# Ajustar el modelo de regresión lineal con los datos de entrenamiento
modelo = sm.OLS(y_entrenamiento, X_entrenamiento).fit()

# Hacer predicciones para los datos de prueba
predicciones = modelo.predict(X_prueba)

# Visualizar las predicciones junto con los valores reales y los datos de entrenamiento
plt.figure(figsize=(10, 6))
sns.lineplot(data=y_prueba, label='Valores reales (prueba)')
sns.lineplot(x=datos_prueba.index, y=predicciones, label='Predicciones')
sns.lineplot(x=datos_entrenamiento.index, y=y_entrenamiento, label='Valores reales (entrenamiento)', color='orange')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación entre valores reales y predicciones')
plt.legend()
plt.show()


import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Supongamos que ya tienes serie_libre_tendencia calculado
# Definir los datos
datos['const'] = 1  # Añadir una columna constante para el término constante
datos['cos_23t'] = np.cos(23 * np.arange(1, len(datos) + 1))
datos['sin_23t'] = np.sin(23 * np.arange(1, len(datos) + 1))

# Definir la variable dependiente utilizando la serie libre de tendencia calculada previamente
y = datos['Electricity production'] - trend

# Determinar el número de puntos correspondientes al 20% final de los datos
n_prediccion = int(len(datos) * 0.2)

# Definir los datos de entrenamiento y de prueba
datos_entrenamiento = datos.iloc[:-n_prediccion]
datos_prueba = datos.iloc[-n_prediccion:]

# Definir las variables independientes para el entrenamiento y la prueba
X_entrenamiento = datos_entrenamiento[['const', 'cos_23t', 'sin_23t']]
X_prueba = datos_prueba[['const', 'cos_23t', 'sin_23t']]

# Definir la variable dependiente para el entrenamiento y la prueba
y_entrenamiento = y.iloc[:-n_prediccion]
y_prueba = y.iloc[-n_prediccion:]

# Ajustar el modelo de regresión lineal con los datos de entrenamiento
modelo = sm.OLS(y_entrenamiento, X_entrenamiento).fit()

# Hacer predicciones para los datos de prueba
predicciones = modelo.predict(X_prueba)

# Sumar las predicciones a la tendencia para obtener la serie original
predicciones_serie_original = predicciones + trend.iloc[-n_prediccion:]

# Visualizar las predicciones junto con los valores reales y los datos de entrenamiento
plt.figure(figsize=(10, 6))
sns.lineplot(data=datos_prueba['Electricity production'], label='Valores reales (prueba)')
sns.lineplot(x=datos_prueba.index, y=predicciones_serie_original, label='Predicciones')
sns.lineplot(x=datos_entrenamiento.index, y=datos_entrenamiento['Electricity production'], label='Valores reales (entrenamiento)', color='orange')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación entre valores reales y predicciones (serie original)')
plt.legend()
plt.show()








import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Supongamos que ya tienes serie_libre_tendencia calculado
# Definir los datos
datos['const'] = 1  # Añadir una columna constante para el término constante
datos['cos_23t'] = np.cos(23 * np.arange(1, len(datos) + 1))
datos['sin_23t'] = np.sin(23 * np.arange(1, len(datos) + 1))

# Definir la variable dependiente utilizando la serie libre de tendencia calculada previamente
y = datos['Electricity production'] - trend

# Determinar el número de puntos correspondientes al 20% final de los datos
n_prediccion = int(len(datos) * 0.2)

# Definir los datos de entrenamiento y de prueba
datos_entrenamiento = datos.iloc[:-n_prediccion]
datos_prueba = datos.iloc[-n_prediccion:]

# Definir las variables independientes para el entrenamiento y la prueba
X_entrenamiento = datos_entrenamiento[['const', 'cos_23t', 'sin_23t']]
X_prueba = datos_prueba[['const', 'cos_23t', 'sin_23t']]

# Definir la variable dependiente para el entrenamiento y la prueba
y_entrenamiento = y.iloc[:-n_prediccion]
y_prueba = y.iloc[-n_prediccion:]

# Ajustar el modelo de regresión lineal con los datos de entrenamiento
modelo = sm.OLS(y_entrenamiento, X_entrenamiento).fit()

# Hacer predicciones para los datos de prueba
predicciones = modelo.predict(X_prueba)

# Sumar las predicciones a la tendencia para obtener la serie original
predicciones_serie_original = predicciones + trend[-n_prediccion:]

# Visualizar las predicciones junto con los valores reales y los datos de entrenamiento
plt.figure(figsize=(10, 6))
sns.lineplot(data=datos_prueba['Electricity production'], label='Valores reales (prueba)')
sns.lineplot(x=datos_prueba.index, y=predicciones_serie_original, label='Predicciones')
sns.lineplot(x=datos_entrenamiento.index, y=datos_entrenamiento['Electricity production'], label='Valores reales (entrenamiento)', color='orange')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación entre valores reales y predicciones (serie original)')
plt.legend()
plt.show()



import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Supongamos que ya tienes serie_libre_tendencia calculado
# Definir los datos
datos['const'] = 1  # Añadir una columna constante para el término constante
frecuencia_principal = 0.1622  # Frecuencia principal proporcionada
periodo_principal = 1 / frecuencia_principal  # Calcular el período correspondiente
datos['cos_principal'] = np.cos(2 * np.pi * np.arange(1, len(datos) + 1) / periodo_principal)
datos['sin_principal'] = np.sin(2 * np.pi * np.arange(1, len(datos) + 1) / periodo_principal)

# Definir la variable dependiente utilizando la serie libre de tendencia calculada previamente
y = datos['Electricity production'] - trend

# Determinar el número de puntos correspondientes al 20% final de los datos
n_prediccion = int(len(datos) * 0.2)

# Definir los datos de entrenamiento y de prueba
datos_entrenamiento = datos.iloc[:-n_prediccion]
datos_prueba = datos.iloc[-n_prediccion:]

# Definir las variables independientes para el entrenamiento y la prueba
X_entrenamiento = datos_entrenamiento[['const', 'cos_principal', 'sin_principal']]
X_prueba = datos_prueba[['const', 'cos_principal', 'sin_principal']]

# Definir la variable dependiente para el entrenamiento y la prueba
y_entrenamiento = y.iloc[:-n_prediccion]
y_prueba = y.iloc[-n_prediccion:]

# Ajustar el modelo de regresión lineal con los datos de entrenamiento
modelo = sm.OLS(y_entrenamiento, X_entrenamiento).fit()

# Hacer predicciones para los datos de prueba
predicciones = modelo.predict(X_prueba)

# Sumar las predicciones a la tendencia para obtener la serie original
predicciones_serie_original = predicciones + trend.iloc[-n_prediccion:]

# Visualizar las predicciones junto con los valores reales y los datos de entrenamiento
plt.figure(figsize=(10, 6))
sns.lineplot(data=datos_prueba['Electricity production'], label='Valores reales (prueba)')
sns.lineplot(x=datos_prueba.index, y=predicciones_serie_original, label='Predicciones')
sns.lineplot(x=datos_entrenamiento.index, y=datos_entrenamiento['Electricity production'], label='Valores reales (entrenamiento)', color='orange')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación entre valores reales y predicciones (serie original)')
plt.legend()
plt.show()




import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Supongamos que ya tienes serie_libre_tendencia calculado
# Definir los datos
datos['const'] = 1  # Añadir una columna constante para el término constante
frecuencia_principal = 0.1622  # Frecuencia principal proporcionada
periodo_principal = 1 / frecuencia_principal  # Calcular el período correspondiente
j = np.arange(1, len(datos) + 1)
datos['cos_principal'] = np.cos(2 * np.pi * j / periodo_principal)
datos['sin_principal'] = np.sin(2 * np.pi * j / periodo_principal)

# Definir la variable dependiente utilizando la serie libre de tendencia calculada previamente
y = datos['Electricity production'] - trend

# Determinar el número de puntos correspondientes al 20% final de los datos
n_prediccion = int(len(datos) * 0.2)

# Definir los datos de entrenamiento y de prueba
datos_entrenamiento = datos.iloc[:-n_prediccion]
datos_prueba = datos.iloc[-n_prediccion:]

# Definir las variables independientes para el entrenamiento y la prueba
X_entrenamiento = datos_entrenamiento[['const', 'cos_principal', 'sin_principal']]
X_prueba = datos_prueba[['const', 'cos_principal', 'sin_principal']]

# Definir la variable dependiente para el entrenamiento y la prueba
y_entrenamiento = y.iloc[:-n_prediccion]
y_prueba = y.iloc[-n_prediccion:]

# Ajustar el modelo de regresión lineal con los datos de entrenamiento
modelo = sm.OLS(y_entrenamiento, X_entrenamiento).fit()

# Hacer predicciones para los datos de prueba
predicciones = modelo.predict(X_prueba)

# Sumar las predicciones a la tendencia para obtener la serie original
predicciones_serie_original = predicciones + trend.iloc[-n_prediccion:]

# Visualizar las predicciones junto con los valores reales y los datos de entrenamiento
plt.figure(figsize=(10, 6))
sns.lineplot(data=datos_prueba['Electricity production'], label='Valores reales (prueba)')
sns.lineplot(x=datos_prueba.index, y=predicciones_serie_original, label='Predicciones')
sns.lineplot(x=datos_entrenamiento.index, y=datos_entrenamiento['Electricity production'], label='Valores reales (entrenamiento)', color='green')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación entre valores reales y predicciones (serie original)')
plt.legend()
plt.show()


import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Supongamos que ya tienes serie_libre_tendencia calculado
# Definir los datos
datos['const'] = 1  # Añadir una columna constante para el término constante
frecuencia_principal = 0.1622  # Frecuencia principal proporcionada
periodo_principal = 1 / frecuencia_principal  # Calcular el período correspondiente
j = np.arange(1, len(datos) + 1)
datos['cos_principal'] = np.cos(2 * np.pi * j / periodo_principal)
datos['sin_principal'] = np.sin(2 * np.pi * j / periodo_principal)

# Definir la variable dependiente utilizando la serie libre de tendencia calculada previamente
y = datos['Electricity production'] - trend

# Aplicar la transformación logarítmica a la variable dependiente
y_log = np.log(y)

# Determinar el número de puntos correspondientes al 20% final de los datos
n_prediccion = int(len(datos) * 0.2)

# Definir los datos de entrenamiento y de prueba
datos_entrenamiento = datos.iloc[:-n_prediccion]
datos_prueba = datos.iloc[-n_prediccion:]

# Definir las variables independientes para el entrenamiento y la prueba
X_entrenamiento = datos_entrenamiento[['const', 'cos_principal', 'sin_principal']]
X_prueba = datos_prueba[['const', 'cos_principal', 'sin_principal']]

# Definir la variable dependiente para el entrenamiento y la prueba (transformada logarítmicamente)
y_entrenamiento_log = y_log.iloc[:-n_prediccion]
y_prueba_log = y_log.iloc[-n_prediccion:]

# Ajustar el modelo de regresión lineal con los datos de entrenamiento (transformados logarítmicamente)
modelo_log = sm.OLS(y_entrenamiento_log, X_entrenamiento).fit()

# Hacer predicciones para los datos de prueba
predicciones_log = modelo_log.predict(X_prueba)

# Sumar las predicciones transformadas a la tendencia para obtener la serie original
predicciones_serie_original_log = np.exp(predicciones_log) + trend.iloc[-n_prediccion:]

# Visualizar las predicciones junto con los valores reales y los datos de entrenamiento
plt.figure(figsize=(10, 6))
sns.lineplot(data=datos_prueba['Electricity production'], label='Valores reales (prueba)')
sns.lineplot(x=datos_prueba.index, y=predicciones_serie_original_log, label='Predicciones (transformación logarítmica)')
sns.lineplot(x=datos_entrenamiento.index, y=datos_entrenamiento['Electricity production'], label='Valores reales (entrenamiento)', color='green')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Comparación entre valores reales y predicciones (serie original)')
plt.legend()
plt.show()





