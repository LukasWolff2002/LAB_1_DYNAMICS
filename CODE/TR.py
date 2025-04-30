
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

# === Leer omega_n y beta ===
valores = {}
with open('CODE/valores.txt', 'r') as file:
    for line in file:
        if ':' in line:
            key, value = line.strip().split(':')
            valores[key.strip()] = float(value.strip())

omega_n = valores['omega_n']
beta = valores['beta']

# === Leer datos ===
data = pd.read_csv('DATA/Armonica_Base.txt', sep=r'\s+', names=['t', 'a1', 'a2'])

# === Función para detectar picos ===
def extract_peaks(time_series, value_series, threshold=0.0005):
    peaks_time = []
    current_times = [time_series.iloc[0]]
    current_values = [value_series.iloc[0]]
    current_sign = value_series.iloc[0] >= 0

    for t, v in zip(time_series.iloc[1:], value_series.iloc[1:]):
        value_sign = v >= 0
        if value_sign == current_sign:
            current_times.append(t)
            current_values.append(v)
        else:
            amplitude = max(current_values) - min(current_values)
            if amplitude >= threshold:
                idx = np.argmax(np.abs(current_values))
                peaks_time.append(current_times[idx])
            current_times = [t]
            current_values = [v]
            current_sign = value_sign

    return np.array(peaks_time)

# === Obtener frecuencia de excitación ===
peaks_t = extract_peaks(data['t'], data['a2'])
periodos = np.diff(peaks_t)
frecuencias = 1 / periodos
frecuencia_media = np.mean(frecuencias)

# === Calcular TR experimental (continua mediante suavizado/interpolación) ===
TR_exp_discreta = np.abs(data['a1']) / np.abs(data['a2'])

# Reasociar a un eje de frecuencia usando ventana deslizante
fs = 1 / np.mean(np.diff(data['t']))  # frecuencia de muestreo
n = len(data)
frecuencia_fft = np.fft.rfftfreq(n, d=1/fs)
fft_a1 = np.abs(np.fft.rfft(data['a1']))
fft_a2 = np.abs(np.fft.rfft(data['a2']))
TR_fft = fft_a1 / fft_a2

# Interpolación para hacerla curva continua
TR_interp = interp1d(frecuencia_fft, TR_fft, kind='cubic', bounds_error=False, fill_value="extrapolate")

# === Calcular TR teórico ===
def TR_teorico(f, fn, beta):
    r = f / fn
    num = 1 + (2 * beta * r)**2
    den = (1 - r**2)**2 + (2 * beta * r)**2
    return np.sqrt(num / den)

f_teo = np.linspace(0.1, frecuencia_fft.max(), 500)
TR_teo = TR_teorico(f_teo, omega_n / (2 * np.pi), beta)
TR_exp = TR_interp(f_teo)

# === Graficar ===
plt.figure(figsize=(10, 6))
plt.plot(f_teo, TR_exp, '-', label='Experimental (interpolada)', alpha=0.7)
plt.plot(f_teo, TR_teo, '-', label=r'Teórica: $f_n={:.2f}\,\mathrm{{Hz}},\, \beta={:.4f}$'.format(omega_n/(2*np.pi), beta))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Función de Transmisibilidad TR')
plt.grid(True)
plt.legend()
plt.title('TR Experimental (Curva) vs Teórico')
plt.tight_layout()

os.makedirs('INFORME/GRAFICOS', exist_ok=True)
plt.savefig('INFORME/GRAFICOS/TR_vs_teorico.png', dpi=300)
plt.show()
