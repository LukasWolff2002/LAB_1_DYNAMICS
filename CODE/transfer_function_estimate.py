import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt, detrend, welch, csd

# === Leer señales ===
data = pd.read_csv('DATA/Concepcion_27F_60.txt', sep=r'\s+', names=['t', 'a1', 'a2'])

# === Convertir aceleraciones de cm/s² a m/s² ===
a1 = data['a1'].values * 0.01
a2 = data['a2'].values * 0.01
t = data['t'].values
dt = np.mean(np.diff(t))
fs = 1 / dt

# === Filtro pasa-altos (Butterworth) ===
b, a = butter(N=4, Wn=0.5, btype='highpass', fs=fs)
a1_filt = filtfilt(b, a, a1)
a2_filt = filtfilt(b, a, a2)

# === Eliminar tendencia ===
a1_filt = detrend(a1_filt)
a2_filt = detrend(a2_filt)

# === Función de transferencia usando Welch y CSD ===
f, Pxy = csd(a1_filt, a2_filt, fs=fs, nperseg=1024)
_, Pxx = welch(a2_filt, fs=fs, nperseg=1024)
Hf = np.abs(Pxy) / Pxx

# === Graficar ===
plt.figure(figsize=(10, 6))
plt.semilogy(f, Hf, label='Transfer Function |H(f)|')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Ganancia (amplitud)')
plt.title('Función de Transferencia entre Aceleración Total y Base')
plt.grid(True)
plt.legend()
plt.tight_layout()

os.makedirs('INFORME/GRAFICOS', exist_ok=True)
plt.savefig('INFORME/GRAFICOS/TransferFunction_TFEstimate.png', dpi=300)
plt.show()