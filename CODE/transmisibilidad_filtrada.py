
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# === Leer omega_n y beta ===
valores = {}
with open('CODE/valores.txt', 'r') as file:
    for line in file:
        if ':' in line:
            key, value = line.strip().split(':')
            valores[key.strip()] = float(value.strip())

omega_n_init = valores['omega_n']
beta_init = valores['beta']

# === Leer datos ===
data = pd.read_csv('DATA/Armonica_Base.txt', sep=r'\s+', names=['t', 'a1', 'a2'])

# === FFT para espectro continuo ===
fs = 1 / np.mean(np.diff(data['t'])) #Frecuencia de muestreo
n = len(data)
frecuencia_fft = np.fft.rfftfreq(n, d=1/fs)
fft_a1 = np.abs(np.fft.rfft(data['a1']))
fft_a2 = np.abs(np.fft.rfft(data['a2']))
TR_fft = fft_a1 / fft_a2

# === Filtrar frecuencias entre 1 y 10 Hz ===
mask = (frecuencia_fft >= 1) & (frecuencia_fft <= 10)
f_filtered = frecuencia_fft[mask]
TR_filtered = TR_fft[mask]

# Interpolación
TR_interp = interp1d(f_filtered, TR_filtered, kind='cubic', bounds_error=False, fill_value="extrapolate")

# === Modelo teórico ===
def TR_teorico(f, fn, beta):
    r = f / fn
    num = 1 + (2 * beta * r)**2
    den = (1 - r**2)**2 + (2 * beta * r)**2
    return np.sqrt(num / den)

# === Ajuste automático de fn y beta ===
f_teo = np.linspace(1, 10, 500)
TR_exp = TR_interp(f_teo)
popt, _ = curve_fit(TR_teorico, f_teo, TR_exp, p0=[omega_n_init / (2 * np.pi), beta_init], bounds=([0, 0], [np.inf, 1]))
fn_fit, beta_fit = popt

TR_fit = TR_teorico(f_teo, fn_fit, beta_fit)

# === Graficar ===
plt.figure(figsize=(10, 6))
plt.plot(f_teo, TR_exp, '-', label='Experimental (filtrada)', alpha=0.7)
plt.plot(f_teo, TR_fit, '--', label=r'Ajuste: $f_n={:.2f}\,\mathrm{{Hz}},\, \beta={:.4f}$'.format(fn_fit, beta_fit))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Función de Transmisibilidad TR')
plt.grid(True)
plt.legend()
plt.title('TR Experimental (Filtrada 1-10 Hz) vs Teórico Ajustado')
plt.tight_layout()

os.makedirs('INFORME/GRAFICOS', exist_ok=True)
plt.savefig('INFORME/GRAFICOS/TR_vs_teorico_filtrada.png', dpi=300)
plt.show()

# === Imprimir resultados ajustados ===
print(f"Frecuencia natural inicial: f_n = {omega_n_init / (2 * np.pi):.4f} Hz")
print(f"Amortiguamiento inicial: beta = {beta_init:.4f}")
print("-------------------------------------------------")
print(f"Frecuencia natural ajustada: f_n = {fn_fit:.4f} Hz")
print(f"Amortiguamiento ajustado: beta = {beta_fit:.4f}")
