
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Leer parámetros dinámicos ===
valores = {}
with open('CODE/valores.txt', 'r') as file:
    for line in file:
        if ':' in line:
            key, value = line.strip().split(':')
            valores[key.strip()] = float(value.strip())

omega_n = valores['omega_n']
beta = valores['beta']
fn = omega_n / (2 * np.pi)

# === Parámetros del sistema ===
m = 0.6073
k = m * omega_n**2
c = 2 * m * beta * omega_n

# === Leer registro sísmico (Caso 3: tiempo, S1, S3) ===
data = pd.read_csv('DATA/Concepcion_27F_60.txt', sep=r'\s+', names=['t', 'a1', 'a2'])

# Convertir aceleración del suelo a m/s²
ug = data['a2'].values * 0.01  # de cm/s² a m/s²

# === Método de Newmark ===
dt = np.mean(np.diff(data['t']))
n = len(data)
p = -m * ug  # fuerza equivalente de base

# Condiciones iniciales
u = np.zeros(n)
v = np.zeros(n)
a = np.zeros(n)

# Newmark parameters (average acceleration)
gamma = 0.5
beta_n = 0.25

# Inicialización
a[0] = (p[0] - c * v[0] - k * u[0]) / m

# Pre-cálculo de constantes
a0 = 1 / (beta_n * dt**2)
a1 = gamma / (beta_n * dt)
a2 = 1 / (beta_n * dt)
a3 = 1 / (2 * beta_n) - 1
a4 = gamma / beta_n - 1
a5 = dt * 0.5 * (gamma / beta_n - 2)

keff = k + a0 * m + a1 * c

# Iteración Newmark (corregida)
for i in range(1, n):
    dp = p[i] - p[i-1] +          m * (a0 * u[i-1] + a2 * v[i-1] + a3 * a[i-1]) +          c * (a1 * u[i-1] + a4 * v[i-1] + a5 * a[i-1])

    du = dp / keff
    u[i] = u[i-1] + du
    v[i] = v[i-1] + gamma / (beta_n * dt) * du - gamma / beta_n * v[i-1] + dt * (1 - gamma / (2 * beta_n)) * a[i-1]
    a[i] = a0 * du - a2 * v[i-1] - a3 * a[i-1]

# === Calcular aceleración absoluta ===
u_abs = a + ug

# === Graficar ===
plt.figure(figsize=(10, 6))
plt.plot(data['t'], u_abs, label='Aceleración Absoluta (Newmark)')
plt.plot(data['t'], data['a1'] * 0.01, label='Aceleración Medida S1 (m/s²)', alpha=0.7)
plt.xlabel('Tiempo [s]')
plt.ylabel('Aceleración [m/s²]')
plt.title('Respuesta Sísmica Absoluta vs Medida - Caso 3')
plt.grid(True)
plt.legend()
plt.tight_layout()

os.makedirs('INFORME/GRAFICOS', exist_ok=True)
plt.savefig('INFORME/GRAFICOS/Respuesta_Newmark_vs_Medida.png', dpi=300)
plt.show()
