import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# =========================
# métricas
# =========================
mse = lambda a,b: np.mean((a-b)**2)
snr = lambda s,a: np.inf if np.sum((s-a)**2)==0 else 10*np.log10(np.sum(s**2)/np.sum((s-a)**2))
E_t = lambda x: np.sum(np.abs(x)**2)
E_f = lambda x: np.sum(np.abs(np.fft.fft(x))**2)/len(x)
db  = lambda X: 20*np.log10(np.abs(X)+1e-12)

# =========================
# helpers
# =========================
def savefig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=200, bbox_inches="tight")
    plt.show()

def rfft_sig(x, fs):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), d=1/fs)
    return f, X

# =====================================================
# Señal de prueba: 2 Hz + 5 Hz (notch en 5 Hz)
# =====================================================
fs = 100
t = np.arange(0, 1, 1/fs)
x = np.sin(2*np.pi*2*t) + np.sin(2*np.pi*5*t)

f, X = rfft_sig(x, fs)
H = np.ones_like(X)
H[np.abs(f-5) <= 0.5] = 0  # notch 5 Hz, bw=0.5
x_f = np.fft.irfft(X*H, n=len(x))

# Figura 1: señal original y filtrada (global)
plt.figure(figsize=(9,4))
plt.plot(t, x, lw=2, color="tab:blue", label="Original")
plt.plot(t, x_f, lw=2, color="tab:orange", label="Filtrada")
plt.grid(True, alpha=0.3)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Prueba (2 Hz + 5 Hz) en el tiempo")
plt.xlim(t[0], t[-1])
plt.legend()
savefig("fig01_prueba_tiempo.png")

# Figura 2: solo la componente de 2 Hz (global)
plt.figure(figsize=(9,4))
plt.plot(t, x_f, lw=2, color="tab:orange")
plt.grid(True, alpha=0.3)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Señal de prueba filtrada (componente de 2 Hz)")
plt.xlim(t[0], t[-1])
savefig("fig02_prueba_solo_2hz.png")

print("=== Parseval prueba ===")
print("E tiempo:", E_t(x), " | E freq:", E_f(x))
print("E tiempo (f):", E_t(x_f), " | E freq (f):", E_f(x_f))

# =====================================================
# Audio real: pasa-bajas simple en FFT (global)
# =====================================================
fs_a, xa = wavfile.read("audio.wav")

# a mono si es estéreo
xa = xa.mean(axis=1) if xa.ndim > 1 else xa

# float + normalización
xa = xa.astype(float)
xa /= np.max(np.abs(xa)) + 1e-12

ta = np.arange(len(xa))/fs_a
fa, Xa = rfft_sig(xa, fs_a)

cutoff = 1000
Ha = (fa <= cutoff).astype(float)
xa_f = np.fft.irfft(Xa*Ha, n=len(xa))

# Figura 3: Audio original en tiempo (GLOBAL)
plt.figure(figsize=(10,4))
plt.plot(ta, xa, lw=1)
plt.grid(True, alpha=0.3)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Audio original en el tiempo (vista global)")
plt.xlim(ta[0], ta[-1])
savefig("fig03_audio_tiempo_original.png")

# Figura 4: Audio filtrado en tiempo (GLOBAL)
plt.figure(figsize=(10,4))
plt.plot(ta, xa_f, lw=1)
plt.grid(True, alpha=0.3)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Audio filtrado en el tiempo (vista global)")
plt.xlim(ta[0], ta[-1])
savefig("fig04_audio_tiempo_filtrado.png")

# Figura 5: Espectro del audio (dB)
plt.figure(figsize=(10,4))
plt.plot(fa, db(Xa), lw=1)
plt.xlim(0, 5000)
plt.grid(True, alpha=0.3)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.title("Espectro del audio (dB)")
savefig("fig05_audio_espectro_db.png")

# Figura 6: Comparación global Antes vs Después (dB)
plt.figure(figsize=(10,4))
plt.plot(fa, db(Xa), lw=1, alpha=0.7, label="Antes")
plt.plot(fa, db(Xa*Ha), lw=1.2, alpha=0.9, label="Después")
plt.xlim(0, 5000)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.title(f"Espectro antes vs después (LPF {cutoff} Hz)")
savefig("fig06_audio_antes_despues_db.png")

print("\n=== Parseval audio ===")
print("E tiempo:", E_t(xa), " | E freq:", E_f(xa))
print("E tiempo (f):", E_t(xa_f), " | E freq (f):", E_f(xa_f))

print("\n=== Métricas audio ===")
print("MSE:", mse(xa, xa_f))
print("SNR (dB):", snr(xa, xa_f))
