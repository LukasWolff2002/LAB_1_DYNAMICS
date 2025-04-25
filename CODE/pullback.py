import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

g = 9.81

# Cargar el archivo txt en un DataFrame de pandas
data = pd.read_csv('DATA/Pullback.txt', sep='\s+', header=None, names=["t", "a1", "a2"])

#Data es un txt que contine las siguientes columnas
#tiempo, aceleracion superior, acelracion base

def plot_data (data):
    plt.figure(figsize=(10, 5))

    # Gráfico de t vs a1
    plt.plot(data['t'], data['a1'], label='a1', color='blue')
    plt.title('Gráfico de t vs a1')
    plt.xlabel('Tiempo (t)')
    plt.ylabel('a1')
    plt.tight_layout()
    #plt.show()
    plt.close()

plot_data(data)

#De esta forma, es posible generar una regresion logaritmica de la aceleracion
#Para eso se separara el dataset en dos datasets, ya que se hicieron 2 iteraciones del pullback
# Filtrar el intervalo de 0-10 segundos para el primer test
data_0_10 = data[(data['t'] >= 0) & (data['t'] <= 10)]

# Encontrar el primer máximo en el intervalo 0-10 segundos
max_idx = data_0_10['a1'].idxmax()
max_time = data_0_10.loc[max_idx, 't']
print(f"Primer máximo en el intervalo 0-10 segundos: {max_time} segundos")

# Filtrar el intervalo de 30-35 segundos para el primer test
data_30_35 = data[(data['t'] >= 30) & (data['t'] <= 35)]

# Encontrar el mínimo absoluto en el intervalo 30-35 segundos (el valor más cercano a 0)
min_idx = (data_30_35['a1'].abs()).idxmin()
min_time = data_30_35.loc[min_idx, 't']
min_value = data_30_35.loc[min_idx, 'a1']
print(f"Mínimo absoluto en el intervalo 30-35 segundos: {min_time} segundos, valor: {min_value}")

# Ahora filtramos el dataframe entre el primer máximo y el mínimo absoluto
first_dataframe = data[(data['t'] >= max_time) & (data['t'] <= min_time)]

# Ajustamos el tiempo en el primer dataframe para que empiece en 0
first_dataframe['t'] = first_dataframe['t'] - first_dataframe['t'].iloc[0]

# El primer dataframe está listo
print(f"Primer dataframe tiene {len(first_dataframe)} filas")
first_dataframe.head()

# Filtrar el intervalo de 30-40 segundos para el segundo test
data_30_40 = data[(data['t'] >= 30) & (data['t'] <= 40)]

# Encontrar el máximo en el intervalo 30-40 segundos
max_30_40_idx = data_30_40['a1'].idxmax()
max_30_40_time = data_30_40.loc[max_30_40_idx, 't']
max_30_40_value = data_30_40.loc[max_30_40_idx, 'a1']
print(f"Máximo en el intervalo 30-40 segundos: {max_30_40_time} segundos, valor: {max_30_40_value}")

# Recortar el segundo dataframe para que tenga la misma longitud que el primer dataframe
second_dataframe = data[(data['t'] >= max_30_40_time) & (data['t'] <= max_30_40_time + (first_dataframe['t'].iloc[-1] - first_dataframe['t'].iloc[0]))]

# Ajustamos el tiempo en el segundo dataframe para que empiece en 0
second_dataframe['t'] = second_dataframe['t'] - second_dataframe['t'].iloc[0]

# El segundo dataframe está listo
print(f"Segundo dataframe tiene {len(second_dataframe)} filas")
second_dataframe.head()

def plot_pullback(data):
    # Graficar el primer intervalo (0-35 segundos)
    plt.figure(figsize=(6, 5))
    plt.plot(data['t'], data['a1'], label='a1', color='blue')
    plt.title('Primer intervalo: t vs a1')
    plt.xlabel('Tiempo (t)')
    plt.ylabel('a1')
    plt.tight_layout()
    #plt.show()
    plt.close()

plot_pullback(first_dataframe)
plot_pullback(second_dataframe)

#Ahora agrego una nueva columna a cada dataframe para poder hacer la regrecion lineal
first_dataframe['ln_a1'] = np.log(np.abs(first_dataframe['a1'] * g))
second_dataframe['ln_a1'] = np.log(np.abs(second_dataframe['a1'] * g))

def regrecion_lineal (data):

    # Realizar la regresión lineal
    slope, intercept, r_value, p_value, std_err = linregress(data['t'], data['ln_a1'])

    # Crear la línea de regresión
    regression_line = slope * data['t'] + intercept

    # Crear el gráfico
    plt.figure(figsize=(10, 5))

    # Gráfico de ln_a1
    plt.plot(data['t'], data['ln_a1'], label='ln_a1', color='blue')

    # Gráfico de la línea de regresión
    plt.plot(data['t'], regression_line, label=f'Regresión Lineal: y = {slope:.2f}x + {intercept:.2f}', color='red', linestyle='--')

    # Título y etiquetas
    plt.title('t vs ln_a1 con Regresión Lineal')
    plt.xlabel('Tiempo (t)')
    plt.ylabel('ln_a1')

    # Mostrar la leyenda
    plt.legend()

    # Ajustar y mostrar el gráfico
    plt.tight_layout()
    plt.show()
    plt.close()

    return np.abs(slope), intercept

firs_slope, first_intercept = regrecion_lineal(first_dataframe)
second_slope, second_intercept = regrecion_lineal(second_dataframe)

#Luego, como es aceleracion, se que la aceleracion maxima se puede calcular como:
#a_max = -rho omega_n**2 e**(-betha omegan t) cos(omega_d t - phi - 2phi1)
#Pero es maximo cuando cos() = +-1

#Por lo tanto, y aplicando log natural
#ln(a_max) = ln(rho omegan**2) - betha omegan t 

#Por lo tanto
#rho omegan**2 = intercept
#betha omegan = slope, ambos son positivos

#Ademas hay condiciones iniciales, velocidad y desplazamiento.

#Recordar que puedo calcular Td. a partir de las mediciones de tiempo








