import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ==========================================
# 1. DEFINICIÓN DE MÉTRICAS Y FUNCIONES ÚTILES
# ==========================================

# Funciones lambda (anonimas) para cálculos matemáticos rápidos:
# mse: Error Cuadrático Medio (promedio de las diferencias al cuadrado).
mse = lambda a,b: np.mean((a-b)**2)

# snr: Relación Señal-Ruido en dB. Calcula la proporción entre la potencia de la señal y el error.
# Si el error es 0, retorna infinito (np.inf), sino aplica la fórmula logarítmica.
snr = lambda s,a: np.inf if np.sum((s-a)**2)==0 else 10*np.log10(np.sum(s**2)/np.sum((s-a)**2))

# E_t: Energía total calculada en el dominio del TIEMPO (suma de amplitudes al cuadrado).
E_t = lambda x: np.sum(np.abs(x)**2)

# E_f: Energía total calculada en el dominio de la FRECUENCIA.
# Se divide por len(x) para cumplir con el Teorema de Parseval en la implementación de numpy.
E_f = lambda x: np.sum(np.abs(np.fft.fft(x))**2)/len(x)

# db: Convierte magnitud lineal a escala logarítmica (Decibelios).
# Se suma 1e-12 para evitar log(0) y errores matemáticos.
db  = lambda X: 20*np.log10(np.abs(X)+1e-12)

def savefig(name):
    """Función auxiliar para ajustar márgenes, guardar la imagen y mostrarla."""
    plt.tight_layout()
    plt.savefig(name, dpi=200)
    plt.show()

def rfft_sig(x, fs):
    """
    Calcula la FFT (Fast Fourier Transform) para señales reales.
    Retorna:
    - f: Eje de frecuencias (eje X).
    - X: Coeficientes complejos de la transformada (eje Y).
    """
    X = np.fft.rfft(x)                 # Transformada rápida (solo parte positiva del espectro)
    f = np.fft.rfftfreq(len(x), d=1/fs) # Genera el vector de frecuencias correspondiente
    return f, X

# ==========================================
# 2. EXPERIMENTO 1: SEÑAL SINTÉTICA (PRUEBA) aprender como funciona el codigo 
# ==========================================
print("Iniciando prueba con señal sintética...")

# Parámetros de la señal
fs = 100                    # Frecuencia de muestreo (100 muestras por segundo)
t = np.arange(0, 1, 1/fs)   # Vector de tiempo: 1 segundo de duración

# Generación de la señal: Suma de una onda de 2 Hz (deseada) y una de 5 Hz (ruido)
x = np.sin(2*np.pi*2*t) + np.sin(2*np.pi*5*t)

# Transformación al dominio de la frecuencia
f, X = rfft_sig(x, fs)

# Diseño del Filtro Notch (Rechaza-banda)
H = np.ones_like(X)         # Crea un vector de '1's del mismo tamaño que el espectro
# Aplica la máscara: Pone '0' donde la frecuencia (f) está cerca de 5 Hz (+/- 0.5 Hz)
H[np.abs(f-5) <= 0.5] = 0   

# Aplicación del filtro y reconstrucción
# Se multiplica el espectro X por la máscara H (filtrado en frecuencia)
# Luego se aplica la Transformada Inversa (irfft) para volver al tiempo
x_f = np.fft.irfft(X*H, n=len(x))

# --- Gráficas del Experimento 1 ---
plt.figure(figsize=(9,4))
plt.plot(t, x, label="Original", lw=2)      # Señal con ruido
plt.plot(t, x_f, label="Filtrada", lw=2)    # Señal limpia (solo 2 Hz)
plt.grid(True, alpha=0.3); plt.legend()
plt.xlabel("Tiempo (s)"); plt.ylabel("Amplitud")
plt.title("Prueba (2 Hz + 5 Hz) en el tiempo")
savefig("fig01_prueba_tiempo.png")

plt.figure(figsize=(9,4))
plt.stem(f, np.abs(X), basefmt=" ", label="Antes")      # Espectro original
plt.stem(f, np.abs(X*H), basefmt=" ", label="Después")  # Espectro con el 0 en 5Hz
plt.xlim(0, 10); plt.grid(True, alpha=0.3); plt.legend()
plt.xlabel("Frecuencia (Hz)"); plt.ylabel("Magnitud")
plt.title("FFT prueba (antes vs después)")
savefig("fig02_prueba_espectro.png")

# Validación de energía (Teorema de Parseval)
print("=== Parseval prueba ===")
print("E tiempo:", E_t(x), " | E freq:", E_f(x))
print("E tiempo (f):", E_t(x_f), " | E freq (f):", E_f(x_f))

# ==========================================
# 3. EXPERIMENTO 2: AUDIO REAL (FILTRO PASA-BAJAS)
# ==========================================
print("\nIniciando procesamiento de audio...")

# Carga del archivo de audio
try:
    fs_a, xa = wavfile.read("audio.wav")
except FileNotFoundError:
    print("ERROR: No se encontró 'audio.wav'. Asegúrate de tener el archivo.")
    exit()

# Preprocesamiento del audio
# Si el audio es estéreo (tiene 2 canales), saca el promedio para hacerlo mono
xa = xa.mean(axis=1) if xa.ndim > 1 else xa 
xa = xa.astype(float) # Convierte a decimales
# Normalización: escala la amplitud entre -1 y 1 dividiendo por el valor máximo
xa /= np.max(np.abs(xa)) + 1e-12

# Vectores de tiempo y frecuencia para el audio
ta = np.arange(len(xa))/fs_a
fa, Xa = rfft_sig(xa, fs_a)

# Diseño del Filtro Pasa-Bajas (Low-Pass Filter)
cutoff = 1000  # Frecuencia de corte en 1000 Hz
# Crea una máscara booleana: 1 si la frecuencia es menor al corte, 0 si es mayor
Ha = (fa <= cutoff).astype(float)

# Filtrado y reconstrucción
xa_f = np.fft.irfft(Xa*Ha, n=len(xa))

# --- Gráficas del Experimento 2 ---
# 1. Señal original en el tiempo
plt.figure(figsize=(10,4))
plt.plot(ta, xa, lw=1)
plt.grid(True, alpha=0.3)
plt.xlabel("Tiempo (s)"); plt.ylabel("Amplitud")
plt.title("Audio original (tiempo)")
savefig("fig03_audio_tiempo_original.png")

# 2. Señal filtrada en el tiempo (se verá más suave)
plt.figure(figsize=(10,4))
plt.plot(ta, xa_f, lw=1)
plt.grid(True, alpha=0.3)
plt.xlabel("Tiempo (s)"); plt.ylabel("Amplitud")
plt.title("Audio filtrado (tiempo)")
savefig("fig04_audio_tiempo_filtrado.png")

# 3. Espectro del audio original en Decibeles
plt.figure(figsize=(10,4))
plt.plot(fa, db(Xa), lw=1)
plt.xlim(0, 5000); plt.grid(True, alpha=0.3) # Zoom a los primeros 5kHz
plt.xlabel("Frecuencia (Hz)"); plt.ylabel("Magnitud (dB)")
plt.title("Espectro audio (dB)")
savefig("fig05_audio_espectro_db.png")

# 4. Comparación de espectros (Antes vs Después)
plt.figure(figsize=(10,4))
plt.plot(fa, db(Xa), lw=1, alpha=0.7, label="Antes")
plt.plot(fa, db(Xa*Ha), lw=1.2, alpha=0.9, label="Después (Corte 1kHz)")
plt.xlim(0, 5000); plt.grid(True, alpha=0.3); plt.legend()
plt.xlabel("Frecuencia (Hz)"); plt.ylabel("Magnitud (dB)")
plt.title(f"Espectro antes vs después (LPF {cutoff} Hz)")
savefig("fig06_audio_antes_despues_db.png")

# Validación final y métricas de error
print("\n=== Parseval audio ===")
print("E tiempo:", E_t(xa), " | E freq:", E_f(xa))
print("E tiempo (f):", E_t(xa_f), " | E freq (f):", E_f(xa_f))

print("\n=== Métricas audio ===")
print("MSE (Error Cuadrático Medio):", mse(xa, xa_f))
print("SNR (Relación Señal-Ruido dB):", snr(xa, xa_f))

print("\n=== Guardando audio ===")
#convertimos los datos para que funcione el audio de salida
xa_f_norm = xa_f / np.max(np.abs(xa_f))
xa_int16 = (xa_f_norm * 32767).astype(np.int16)

# 3. Guardar el archivo convertido
wavfile.write("audio_filtrado_final.wav", fs_a, xa_int16)
print("Audio guardado exitosamente en formato PCM 16-bit.")