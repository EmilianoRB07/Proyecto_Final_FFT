# Denoising de Audio en el Dominio de la Frecuencia (FFT)

## Descripción del proyecto
Este proyecto implementa un sistema de reducción de ruido (denoising) de señales de audio
utilizando la Transformada Rápida de Fourier (FFT).  
El procesamiento se realiza en el dominio de la frecuencia mediante el diseño de filtros
ideales (pasa-bajas y notch), y posteriormente se reconstruye la señal en el dominio del tiempo
usando la transformada inversa (IFFT).

Además, se verifican numéricamente las propiedades de la energía de la señal mediante el
Teorema de Parseval y se evalúa el desempeño del filtrado utilizando métricas objetivas
como MSE y SNR.

---

## Objetivo
- Aplicar la FFT para analizar señales en el dominio de la frecuencia.
- Diseñar filtros simples en el dominio frecuencial.
- Reconstruir señales mediante la IFFT.
- Comparar señales originales y filtradas en tiempo y frecuencia.
- Verificar el Teorema de Parseval.
- Evaluar la calidad del filtrado mediante métricas numéricas.

---

## Requisitos
- Python 3.9 o superior
- Librerías:
  - numpy
  - matplotlib
  - scipy

---

## Instalación de dependencias
Ejecutar el siguiente comando en la terminal:

```bash
pip install numpy matplotlib scipy
