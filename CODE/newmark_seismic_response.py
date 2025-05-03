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

omega_n = valores['omega_n']  # rad/s
beta = valores['beta']        # damping ratio
fn = omega_n / (2 * np.pi)    # Hz

# === Parámetros del sistema ===
m = 0.6073  # kg
k = m * omega_n**2
c = 2 * m * beta * omega_n

# === Leer registro sísmico (Concepción 27F - Caso 3) ===
data = pd.read_csv('DATA/Concepcion_27F_60.txt', sep=r'\s+', names=['t', 'a1', 'a2']) #Cambiar archivo a Kobe tambien

ug = data['a2'].values * 0.01  # m/s², aceleración base (input)
a1_measured = data['a1'].values * 0.01  # m/s², aceleración salida medida
t = data['t'].values
dt = np.mean(np.diff(t))
n = len(t)

# === Método de Newmark (gamma=0.5, beta=0.25) ===
u = np.zeros(n)
v = np.zeros(n)
a = np.zeros(n)
p = -m * ug

gamma = 0.5
beta_n = 0.25

a0 = 1 / (beta_n * dt**2)
a1_ = gamma / (beta_n * dt)
a2_ = 1 / (beta_n * dt)
a3 = 1 / (2 * beta_n) - 1
a4 = gamma / beta_n - 1
a5 = dt * 0.5 * (gamma / beta_n - 2)

keff = k + a0 * m + a1_ * c

a[0] = (p[0] - c * v[0] - k * u[0]) / m

for i in range(1, n):
    dp = p[i] - p[i-1] + m * (a0 * u[i-1] + a2_ * v[i-1] + a3 * a[i-1]) + c * (a1_ * u[i-1] + a4 * v[i-1] + a5 * a[i-1])
    du = dp / keff
    u[i] = u[i-1] + du
    v[i] = v[i-1] + gamma / (beta_n * dt) * du - gamma / beta_n * v[i-1] + dt * (1 - gamma / (2 * beta_n)) * a[i-1]
    a[i] = (u[i] - u[i-1] - dt * v[i-1] - dt**2 / 2 * a[i-1]) * (2 / dt**2)

# === Aceleración absoluta ===
u_abs = a + ug

# === Guardar resultados numéricos ===
df_result = pd.DataFrame({'t': t, 'u_abs': u_abs, 'a1_measured': a1_measured})
os.makedirs('INFORME/RESULTADOS', exist_ok=True)
df_result.to_csv('INFORME/RESULTADOS/resultados_aceleracion.csv', index=False)

# === Graficar comparación ===
plt.figure(figsize=(10, 6))
plt.plot(t, u_abs, label='Aceleración Absoluta (Newmark)', linewidth=1.2)
plt.plot(t, a1_measured, label='Aceleración Medida S1 (m/s²)', alpha=0.7, linewidth=1)
plt.xlabel('Tiempo [s]')
plt.ylabel('Aceleración [m/s²]')
plt.title(f'Respuesta Sísmica Absoluta vs Medida\nMétodo de Newmark | fn = {fn:.3f} Hz | ζ = {beta:.3f}')
plt.grid(True)
plt.legend()
plt.tight_layout()

os.makedirs('INFORME/GRAFICOS', exist_ok=True)
plt.savefig('INFORME/GRAFICOS/Respuesta_Newmark_vs_Medida.png', dpi=300)
plt.show()
