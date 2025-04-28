#El ensayo pullback calcula valores como omega_n, beta
#Por lo tanto, extraigo los valores del archivo de texto


import os

# Nombre del archivo
file_name = "CODE/valores.txt"  # Cambia esto por tu nombre real si es diferente
prefix = os.path.splitext(os.path.basename(file_name))[0]

# Crear un diccionario para guardar las variables
variables = {}

valores_omega_n = 0
valores_beta = 0

# Leer el archivo
with open(file_name, "r") as file:
    lines = file.readlines()
    for line in lines:
        if ":" in line:
            key, value = line.split(":")
            key = key.strip()
            value = float(value.strip())
            # Crear el nombre de la variable con el prefijo del archivo
            variable_name = f"{prefix}_{key}"
            variables[variable_name] = value

# Asignar las variables dinámicamente
for var_name, var_value in variables.items():
    globals()[var_name] = var_value

#Ahora tengo las variables para usarlas
#print(valores_omega_n)
#print(valores_beta)

omega_n = valores_omega_n
beta = valores_beta






#Para plotear la curva de transmisibilidad, primero debo conocer la frecuencia de exitacion del suelo
#En primer lugar leo el archivo txt y me quedo con los peaks de aceleracion, tanto para a1 como a2
import pandas as pd

# Cargar el archivo
data = pd.read_csv('DATA/Armonica_Base.txt', sep='\s+', names=['t', 'a1', 'a2'])

def extract_peaks_with_time(time_series, value_series, threshold=0.0005):
    peaks_time = []
    peaks_value = []
    
    current_times = [time_series.iloc[0]]
    current_values = [value_series.iloc[0]]
    current_sign = value_series.iloc[0] >= 0
    
    for t, v in zip(time_series.iloc[1:], value_series.iloc[1:]):
        value_sign = v >= 0
        if value_sign == current_sign:
            current_times.append(t)
            current_values.append(v)
        else:
            # Cambio de signo detectado: verificar si el cambio es suficiente
            amplitude = max(current_values) - min(current_values)
            if amplitude >= threshold:
                if current_sign:
                    idx = current_values.index(max(current_values))
                else:
                    idx = current_values.index(min(current_values))
                
                peaks_time.append(current_times[idx])
                peaks_value.append(current_values[idx])
            
            # Reiniciar
            current_times = [t]
            current_values = [v]
            current_sign = value_sign

    # Agregar el último segmento
    if current_values:
        amplitude = max(current_values) - min(current_values)
        if amplitude >= threshold:
            if current_sign:
                idx = current_values.index(max(current_values))
            else:
                idx = current_values.index(min(current_values))
            
            peaks_time.append(current_times[idx])
            peaks_value.append(current_values[idx])
    
    # Retornar como DataFrame
    return pd.DataFrame({'t': peaks_time, 'valor': peaks_value})

# Aplicar a a1 y a2 usando la nueva función
peaks_a1 = extract_peaks_with_time(data['t'], data['a1'])
peaks_a2 = extract_peaks_with_time(data['t'], data['a2'])

#De esta forma, ahora tengo los picos de aceleracion tanto para a1 como para a2

import matplotlib.pyplot as plt

# Gráfico de peaks_a1
plt.figure(figsize=(10, 5))
plt.plot(peaks_a1['t'], peaks_a1['valor'], linestyle='-', color='red', label='Picos a1')
plt.xlabel('Tiempo [s]')
plt.ylabel('Aceleración a1')
plt.title('Picos conectados en a1')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico de peaks_a2
plt.figure(figsize=(10, 5))
plt.plot(peaks_a2['t'], peaks_a2['valor'], linestyle='-', color='blue', label='Picos a2')
plt.xlabel('Tiempo [s]')
plt.ylabel('Aceleración a2')
plt.title('Picos conectados en a2')
plt.legend()
plt.grid(True)
plt.show()

#Por lo tanto, ahora debo obtener la frecuencia de excitacion del suelo


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Calcular diferencias entre picos ---
delta_t = np.diff(peaks_a2['t'].values)  # fuerza a ser np.ndarray

# --- Calcular frecuencia ---
frecuencia = 1 / delta_t  # f = 1/T

# --- Tiempo medio entre dos picos ---
t_frecuencia = (peaks_a2['t'].values[:-1] + peaks_a2['t'].values[1:]) / 2

# --- Ahora sí los arrays tienen la misma longitud (2034) ---
frecuencia_a2 = pd.DataFrame({
    't': t_frecuencia,
    'frecuencia': frecuencia
})

# --- Graficar ---
plt.figure(figsize=(10, 5))
plt.plot(frecuencia_a2['t'], frecuencia_a2['frecuencia'], linestyle='-')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia Instantánea [Hz]')
plt.title('Frecuencia Instantánea basada en picos de a2')
plt.grid(True)
plt.show()

#Puedo alicar una convolucion para suavizar la frecuencia y tener datos mas precisos para trabajar

# Primero definimos una ventana de suavizado
window_size = 5  # Número de puntos para promediar (puedes cambiarlo)

# Creamos una ventana (simplemente un vector de unos)
window = np.ones(window_size) / window_size

# Aplicamos convolución a la frecuencia
frecuencia_suavizada = np.convolve(frecuencia_a2['frecuencia'], window, mode='same')

# Agregamos la frecuencia suavizada al DataFrame
frecuencia_a2['frecuencia_suavizada'] = frecuencia_suavizada

# --- Graficar ---
plt.figure(figsize=(10, 5))
plt.plot(frecuencia_a2['t'], frecuencia_a2['frecuencia'], linestyle='-', label='Frecuencia Original')
plt.plot(frecuencia_a2['t'], frecuencia_a2['frecuencia_suavizada'], color='red', linewidth=2, label='Frecuencia Suavizada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.title('Frecuencia Instantánea y Suavizada')
plt.legend()
plt.grid(True)
plt.show()


#Los datos aun estan muy sucios


import numpy as np
import matplotlib.pyplot as plt

# --- Definir señal y tiempos ---
t = data['t'].values
a2 = data['a2'].values

# --- Calcular el paso de muestreo ---
dt = np.mean(np.diff(t))  # Puede no ser exactamente uniforme, por eso promedio

# --- Número de puntos ---
N = len(a2)

# Hacer la FFT primero sobre toda la señal
fft_a2_full = np.fft.fft(a2)

# Luego, para graficar el espectro:
frecuencias = np.fft.fftfreq(len(a2), d=dt)
mask = frecuencias >= 0
frecuencias_plot = frecuencias[mask]
fft_a2_plot = fft_a2_full[mask]

# Para graficar el espectro
amplitud = np.abs(fft_a2_plot)

# --- Graficar solo para espectro ---
plt.figure(figsize=(10, 5))
plt.plot(frecuencias_plot, amplitud)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.title('Espectro de Frecuencia de a2')
plt.grid(True)
plt.xlim(0, 50)  # Limitar la frecuencia visible
plt.show()
