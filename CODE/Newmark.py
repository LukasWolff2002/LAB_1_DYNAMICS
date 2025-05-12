import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def respnewmark(m, T, b, P, Fs, xo=0.0, vo=0.0, beta=0.25, gama=0.5):
    """
    Método de Newmark para un sistema SDOF:
    - m: masa [kg]
    - T: periodo natural [s]
    - b: razón de amortiguamiento crítico ζ
    - P: vector de fuerza externa [N]
    - Fs: frecuencia de muestreo [Hz]
    - xo, vo: condiciones iniciales de desplazamiento [m] y velocidad [m/s]
    - beta, gama: parámetros de Newmark
    Retorna desplazamientos x, velocidades v, aceleraciones relativas a y aceleraciones absolutas at.
    """
    P = P.flatten()
    N = len(P)
    x = np.zeros(N)
    v = np.zeros(N)
    a = np.zeros(N)

    # Propiedades del sistema
    w = 2 * np.pi / T
    k = m * w**2
    c = 2 * m * w * b

    dt = 1 / Fs
    dt2 = dt**2

    # Condiciones iniciales
    x[0] = xo
    v[0] = vo
    a[0] = (P[0] - c * v[0] - k * x[0]) / m

    # Coeficientes Newmark
    k1 = k + gama * c / (beta * dt) + m / (beta * dt2)
    A = m / (beta * dt) + c * gama / beta
    B = m / (2*beta) + dt * (gama/(2*beta) - 1) * c

    # Iteración en el tiempo
    for i in range(N-1):
        deltaP = P[i+1] - P[i] + A*v[i] + B*a[i]
        deltax = deltaP / k1
        deltav = gama*deltax/(beta*dt) - gama*v[i]/beta + dt*(1 - gama/(2*beta))*a[i]
        deltaa = deltax/(beta*dt2) - v[i]/(beta*dt) - a[i]/(2*beta)

        x[i+1] = x[i] + deltax
        v[i+1] = v[i] + deltav
        a[i+1] = a[i] + deltaa

    # Aceleración absoluta
    at = a - P / m

    return x, v, a, at

# === Leer parámetros dinámicos ===
valores = {}
with open('CODE/valores.txt', 'r') as file:
    for line in file:
        if ':' in line:
            key, value = line.strip().split(':', 1)
            valores[key.strip()] = float(value.strip())

omega_n = valores['omega_n']    # rad/s
beta_ratio = valores['beta']     # damping ratio ζ
fn = omega_n / (2 * np.pi)       # Hz

# === Parámetros del sistema SDOF ===
m = 0.6073                       # masa [kg]
T = 2 * np.pi / omega_n         # periodo natural [s]
b = beta_ratio                   # ζ

# === Leer excitación sísmica ===
df = pd.read_csv(
    'DATA/Kobe.txt',
    sep=r'\s+',
    header=None,
    names=['t','a1_meas','a2_base']
)
t = df['t'].values
ug = df['a2_base'].values * 0.01  # m/s²
dt = np.mean(np.diff(t))
Fs = 1 / dt

# Vector de fuerza sísmica: F = -m·ag
P = -m * ug

# Condiciones iniciales
xo, vo = 0.0, 0.0
beta_nm, gama_nm = 0.25, 0.5

# === Calcular respuesta con Newmark ===
x, v, a_rel, a_abs = respnewmark(m, T, b, P, Fs, xo, vo, beta_nm, gama_nm)

# === Guardar resultados ===
os.makedirs('RESULTADOS', exist_ok=True)
pd.DataFrame({
    't [s]':         t,
    'x [m]':         x,
    'v [m/s]':       v,
    'a_rel [m/s²]':  a_rel,
    'a_abs [m/s²]':  a_abs
}).to_csv('RESULTADOS/respnewmark_Concepcion.csv', index=False)

# === Graficar resultados ===
plt.figure(figsize=(10,6))
plt.plot(t, x,       label='Desplazamiento x(t)')
plt.plot(t, v,       label='Velocidad v(t)')
plt.plot(t, a_abs,   label='Aceleración absoluta a(t)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Respuesta')
plt.title(f'Respuesta Newmark SDOF | fn={fn:.3f} Hz | ζ={beta_ratio:.3f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('INFORME/GRAFICOS/respnewmark_Kobe.png', dpi=300)
plt.show()
