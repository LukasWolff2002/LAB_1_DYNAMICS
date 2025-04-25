import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sympy import symbols, Eq, solve, sqrt, pi

g = 9.81

# Cargar el archivo txt en un DataFrame de pandas
data = pd.read_csv('DATA/Pullback.txt', sep='\\s+', header=None, names=["t", "a1", "a2"])

#Data es un txt que contine las siguientes columnas
#tiempo, aceleracion superior, acelracion base

#En primer lugar hay que limpiar el dataset para quedarse solo con los maximos
def limpiar_datos(data):
    # Inicializamos listas para los resultados filtrados
    cleaned_data = []
    
    # Variables de control
    current_sign = None  # Para controlar si estamos en un intervalo positivo o negativo
    interval_values = []  # Para almacenar los valores de cada intervalo

    # Iterar sobre los datos
    for i in range(len(data) - 1):
        current_value = data.iloc[i]['a1']
        
        # Determinar el signo de la aceleración actual
        if current_value > 0:
            sign = 'positive'
        elif current_value < 0:
            sign = 'negative'
        else:
            sign = current_sign  # Si el valor es 0, mantener el signo del intervalo anterior

        # Si el signo cambia (de positivo a negativo o de negativo a positivo), procesamos el intervalo
        if sign != current_sign:
            if current_sign == 'positive' and len(interval_values) > 0:
                # Al final de un intervalo positivo, tomamos el máximo
                max_value = max(interval_values)
                max_idx = interval_values.index(max_value)
                cleaned_data.append((data.iloc[i - len(interval_values) + max_idx]['t'], max_value))
            elif current_sign == 'negative' and len(interval_values) > 0:
                # Al final de un intervalo negativo, tomamos el mínimo
                min_value = min(interval_values)
                min_idx = interval_values.index(min_value)
                cleaned_data.append((data.iloc[i - len(interval_values) + min_idx]['t'], min_value))
            
            # Reiniciar la lista de intervalos y cambiar el signo
            interval_values = [current_value]
            current_sign = sign
        else:
            interval_values.append(current_value)

    # Procesar el último intervalo (puede quedar sin procesar en el bucle)
    if current_sign == 'positive' and len(interval_values) > 0:
        max_value = max(interval_values)
        max_idx = interval_values.index(max_value)
        cleaned_data.append((data.iloc[-len(interval_values) + max_idx]['t'], max_value))
    elif current_sign == 'negative' and len(interval_values) > 0:
        min_value = min(interval_values)
        min_idx = interval_values.index(min_value)
        cleaned_data.append((data.iloc[-len(interval_values) + min_idx]['t'], min_value))

    # Convertir a un DataFrame
    cleaned_df = pd.DataFrame(cleaned_data, columns=["t", "a1"])
    return cleaned_df

# Limpiar los datos
data = limpiar_datos(data)

def plot_data (data):
    plt.figure(figsize=(10, 5))

    # Gráfico de t vs a1
    plt.plot(data['t'], data['a1'], label='a1', color='blue')
    plt.title('Gráfico de t vs a1')
    plt.xlabel('Tiempo (t)')
    plt.ylabel('a1')
    plt.tight_layout()
    plt.show()
    plt.close()

plot_data(data)

#De esta forma, es posible generar una regresion logaritmica de la aceleracion
#Para eso se separara el dataset en dos datasets, ya que se hicieron 2 iteraciones del pullback
# Filtrar el intervalo de 0-10 segundos para el primer test
# Filtrar el intervalo de 0-10 segundos para el primer test
data_0_10 = data[(data['t'] >= 0) & (data['t'] <= 10)]

# Encontrar el primer máximo en el intervalo 0-10 segundos
max_idx = data_0_10['a1'].idxmax()
max_time = data_0_10.loc[max_idx, 't']

# Filtrar el intervalo de 30-35 segundos para el primer test
data_30_35 = data[(data['t'] >= 30) & (data['t'] <= 35)]

# Encontrar el mínimo absoluto en el intervalo 30-35 segundos (el valor más cercano a 0)
min_idx = (data_30_35['a1'].abs()).idxmin()
min_time = data_30_35.loc[min_idx, 't']
min_value = data_30_35.loc[min_idx, 'a1']

# Ahora filtramos el dataframe entre el primer máximo y el mínimo absoluto
first_dataframe = data[(data['t'] >= max_time) & (data['t'] <= min_time)].copy()

# Ajustamos el tiempo en el primer dataframe para que empiece en 0
first_dataframe['t'] = first_dataframe['t'] - first_dataframe['t'].iloc[0]

# El primer dataframe está listo
first_dataframe.head()

# Filtrar el intervalo de 30-40 segundos para el segundo test
data_30_40 = data[(data['t'] >= 30) & (data['t'] <= 40)]

# Encontrar el máximo en el intervalo 30-40 segundos
max_30_40_idx = data_30_40['a1'].idxmax()
max_30_40_time = data_30_40.loc[max_30_40_idx, 't']
max_30_40_value = data_30_40.loc[max_30_40_idx, 'a1']

# Recortar el segundo dataframe para que tenga la misma longitud que el primer dataframe
second_dataframe = data[(data['t'] >= max_30_40_time) & (data['t'] <= max_30_40_time + (first_dataframe['t'].iloc[-1] - first_dataframe['t'].iloc[0]))].copy()

# Ajustamos el tiempo en el segundo dataframe para que empiece en 0
second_dataframe['t'] = second_dataframe['t'] - second_dataframe['t'].iloc[0]

# El segundo dataframe está listo
second_dataframe.head()

def plot_pullback(data):
    # Graficar el primer intervalo (0-35 segundos)
    plt.figure(figsize=(6, 5))
    plt.plot(data['t'], data['a1'], label='a1', color='blue')
    plt.title('Primer intervalo: t vs a1')
    plt.xlabel('Tiempo (t)')
    plt.ylabel('a1')
    plt.tight_layout()
    plt.show()
    plt.close()

plot_pullback(first_dataframe)
plot_pullback(second_dataframe)

# Ahora agrego una nueva columna a cada dataframe para poder hacer la regresión lineal
first_dataframe.loc[:, 'ln_a1'] = np.log(np.abs(first_dataframe['a1'] * g))
second_dataframe.loc[:, 'ln_a1'] = np.log(np.abs(second_dataframe['a1'] * g))

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

def calcular_Td(data):
    # Calcular las diferencias de tiempo entre puntos consecutivos
    delta_t = data['t'].diff().dropna()  # diff() calcula t(i+1) - t(i), y dropna elimina el primer NaN
    
    # Calcular el promedio de las diferencias de tiempo
    Td = 2*delta_t.mean()  # Promedio de las diferencias de tiempo
    
    return Td



first_slope, first_intercept = regrecion_lineal(first_dataframe)
second_slope, second_intercept = regrecion_lineal(second_dataframe)

first_Td = calcular_Td(first_dataframe)
second_Td = calcular_Td(second_dataframe)

#Luego, como es aceleracion, se que la aceleracion maxima se puede calcular como:
#a_max = -rho omega_n**2 e**(-betha omegan t) cos(omega_d t - phi - 2phi1)
#Pero es maximo cuando cos() = +-1

#Por lo tanto, y aplicando log natural
#ln(a_max) = ln(rho omegan**2) - betha omegan t 

#Por lo tanto
#rho omegan**2 = intercept
#betha omegan = slope, ambos son positivos

#Ademas hay condiciones iniciales, velocidad y desplazamiento.
#Conozco Td, donde Td = 2pi/omega_d = 2pi/omega_n sqrt(1 - (betha)**2)

#Por lo tanto tengo dos ecuaciones

# Definir las incógnitas

def solve_sistem(first_slope, first_Td):
    # Definir las variables simbólicas
    betha, omega_n = symbols('betha omega_n')

    # Ecuaciones del primer test
    eq1 = Eq(betha * omega_n, first_slope)
    eq2 = Eq((2 * pi) / first_Td, omega_n * sqrt(1 - betha**2))

    # Resolver el sistema de ecuaciones
    solution1 = solve((eq1, eq2), (betha, omega_n))

    # Verificar si hay soluciones
    if solution1:
        # Desempaquetar las soluciones de la tupla
        beta_value, omega_n_value = solution1[0]  # Accede a la primera tupla de soluciones
        
        return beta_value, omega_n_value
    
first_beta, first_omega_n = solve_sistem(first_slope, first_Td)
print('\nResultados de la regrecion lineal:')
print(f"Primer test: beta = {first_beta}, omega_n = {first_omega_n}")

# Repetir el proceso para el segundo test
second_beta, second_omega_n = solve_sistem(second_slope, second_Td)
print(f"Segundo test: beta = {second_beta}, omega_n = {second_omega_n}")

# Función para graficar la transformada de Fourier
def plot_fourier_transform(data, title="Transformada de Fourier"):
    N = len(data['t'])
    dt = data['t'].iloc[1] - data['t'].iloc[0]  # Paso de tiempo
    f = np.fft.fftfreq(N, dt)  # Frecuencias
    
    # Realizar la transformada de Fourier (FFT)
    fft_values = np.fft.fft(data['a1'])  # Transformada de Fourier de la aceleración

    # Normalizar la magnitud
    fft_values = np.abs(fft_values) / N  # Normalización

    # Tomar solo la mitad positiva de las frecuencias
    positive_freqs = f[:N // 2]
    positive_fft_values = fft_values[:N // 2]  # Magnitud de la FFT

    # Verificar si la FFT tiene algún valor significativo
    if np.all(positive_fft_values == 0):
        print("Advertencia: La FFT no muestra valores significativos.")
    else:
        # Encontrar el índice del máximo valor de la FFT
        max_idx = np.argmax(positive_fft_values)
        max_freq = positive_freqs[max_idx]  # Frecuencia correspondiente al máximo
        
        # Graficar la transformada de Fourier
        plt.figure(figsize=(10, 5))
        plt.plot(positive_freqs, positive_fft_values, label="FFT de a1")
        
        # Agregar una línea vertical en el máximo
        plt.axvline(x=max_freq, color='red', linestyle='--', label=f'Máximo en {max_freq:.2f} Hz')
        
        # Título y etiquetas
        plt.title(title)
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Magnitud")
        
        # Mostrar leyenda
        plt.legend()
        
        # Ajustar y mostrar el gráfico
        plt.tight_layout()
        plt.show()

        return max_freq


# Graficar la transformada de Fourier para el primer y segundo test
max_frec_furier_first = plot_fourier_transform(first_dataframe, title="Transformada de Fourier - Primer Test")
max_frec_furier_second = plot_fourier_transform(second_dataframe, title="Transformada de Fourier - Segundo Test")

print('\nSegun la regrecion lineal:')
print(f'la frecuencia de la transformada de Fourier es: {first_omega_n/(2*np.pi)} Hz')
print(f'la frecuencia de la transformada de Fourier es: {second_omega_n/(2*np.pi)} Hz')

print('\nSegun la transformada de Fourier:')
print(f'La frecuencia de la transformada de Fourier es: {max_frec_furier_first} Hz')
print(f'La frecuencia de la transformada de Fourier es: {max_frec_furier_second} Hz')