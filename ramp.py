# ramp.py
import math
import numpy as np
import matplotlib.pyplot as plt

def create_ideal_ramp(N, delta):
    cut_off = math.pow(2 * delta, -1)
    freq = np.fft.fftfreq(2 * N - 1, d=delta)
    w_abs = np.abs(freq) # Jacobian
    H = np.where(w_abs <= cut_off, w_abs, 0)
    
    return np.real(H)

def create_discrete_ramp(N, delta):
    n = np.arange(1 - N, N, 1)
    h = np.zeros_like(n , dtype=np.float32)
    # n = 0
    h[n == 0] = 1 / (4 * delta ** 2)
    # n is even -> h = 0
    # n is odd
    h[n % 2 != 0] = -1 / (n[n % 2 != 0] * np.pi * delta) ** 2
        
    h_shifted = np.fft.ifftshift(h)
    H = delta * np.fft.fft(h_shifted)

    return np.real(H)

if __name__ == "__main__":
    # Init
    N = 10
    delta = 1
    print("=== Start ramp.py ===")
    
    freq = freq = np.fft.fftfreq(2 * N - 1, d=delta)
    freq_sorted = np.fft.fftshift(freq)
    
    ideal_ramp = create_ideal_ramp(N, delta)
    ideal_ramp_sorted = np.fft.fftshift(ideal_ramp)

    discrete_ramp = create_discrete_ramp(N, delta)
    discrete_ramp_sorted = np.fft.fftshift(discrete_ramp)
    
    rmse = np.sqrt(np.mean(np.power(ideal_ramp - discrete_ramp, 2)))
    print(f"RMSE: {rmse}")
    
    # Save figure
    plt.title("Compare Ramp Filters")
    plt.plot(freq_sorted, ideal_ramp_sorted, label='Ideal')
    plt.plot(freq_sorted, discrete_ramp_sorted,  label='Discrete')
    plt.legend()
    plt.savefig("compare_ramp_filters.png")
    plt.clf() # Reset plt