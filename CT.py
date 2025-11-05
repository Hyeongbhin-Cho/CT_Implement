# CT.py
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def save_img(img, name, save_dir="./", type="clip"):
    if type == "clip":
        img = np.clip(img, 0, 1)
    elif type == "min-max":
        min = img.min()
        max = img.max()
        img = (img - min) / (max - min)
    elif type == "abs":
        img = np.abs(img)
        min = img.min()
        max = img.max()
        img = (img - min) / (max - min)
    else:
        raise ValueError("type must be one of 'clip', 'min-max', 'abs'.")
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"{name}.png"), img)

def create_source(x, y, center, A, B, coefficient=1):    
    return np.where((x - center[0]) ** 2 / A ** 2 + (y - center[1]) ** 2 / B ** 2 <= 1, coefficient, 0)

def compute_projection(source, bias_x, bias_y, thetas, ts, D, delta_t=0.5, delta_l= 0.5, quarter_offset=False):
    # Init
    H, W = source.shape
    P = np.zeros((len(thetas), len(ts)))
    t_shift = delta_t / 4 if quarter_offset else 0    # Detector quater-offset 
    
    for i, theta in enumerate(thetas):
        for j, t in enumerate(ts):
            l = np.arange(-D/2, D/2+delta_l, delta_l)
            
            l_x = (t + t_shift) * np.cos(theta) - l * np.sin(theta)
            l_y = (t + t_shift) * np.sin(theta) + l * np.cos(theta)

            # Line integral
            sample_sum = 0
            for k in range(len(l)):
                biased_x = l_x[k] - bias_x
                biased_y = l_y[k] - bias_y
                
                # Interpolation
                x_floor = int(np.floor(biased_x))
                y_floor = int(np.floor(biased_y))
                x_ceil = x_floor + 1
                y_ceil = y_floor + 1
            
                if (x_floor >= 0 and x_ceil < W and y_floor >= 0 and y_ceil < H):
                    dx = biased_x - x_floor
                    dy = biased_y - y_floor
                    
                    p_tr = source[y_ceil , x_ceil] 
                    p_tl = source[y_ceil , x_floor]
                    p_br = source[y_floor, x_ceil]
                    p_bl = source[y_floor, x_floor]
                    
                    # Weight is ratio of area
                    sample_sum +=  (1 - dy) * ((1 - dx) * p_bl + dx * p_br) + dy * ((1 - dx) * p_tl + dx * p_tr)

            P[i, j] = delta_l * sample_sum
            
    return P

def apply_detector_let(P, let_size):
    pre_pad = (let_size - 1) // 2
    post_pad = let_size // 2
    
    P = np.pad(P, ((pre_pad, post_pad), (0, 0)), mode="wrap")
    kernel = np.ones(let_size) / let_size
    
    conv1d = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), axis=0, arr=P)
    P_ = np.pad(conv1d, ((0, 0), (pre_pad, post_pad)), mode="wrap")
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"), axis=1, arr=P_)

def compute_fourier_slice(P):
    _, num_ts = P.shape
    
    length = int(math.pow(2, math.ceil(math.log2(2 * num_ts)))) # length =  2^n > 2 * det
    
    # Zero pad
    P = np.pad(P, ((0,0), (0, length- num_ts)), mode="constant", constant_values=0)
    
    # Fourier slice
    S = np.fft.fft(P, axis=1)
    
    return S

def compute_filtered_projection(S, delta_t=0.5, kernel="ramp", kernel_args={}):
    # Init
    _, len_S = S.shape
    
    # kernel
    H = None
    if kernel == "ramp":
        # 0이 중심인 n인덱스 생성        
        n_freq_order = np.fft.fftfreq(len_S) * len_S
        n = np.fft.fftshift(n_freq_order).astype(np.float32)
        
        h = np.zeros_like(n , dtype=np.float32)
        # n = 0
        h[n == 0] = 1 / (4 * delta_t ** 2)
        # n is even -> h = 0
        # n is odd
        h[n % 2 != 0] = -1 / (n[n % 2 != 0] * np.pi * delta_t) ** 2
        
        h_shifted = np.fft.ifftshift(h)

        H = delta_t * np.fft.fft(h_shifted)
        
        """
        # Save figure
        plt.plot(n, h)
        plt.title("Spatial Domain")
        plt.xlabel("n")
        plt.ylabel("h(n)")
        plt.savefig("ramp_spatial_domain.png")
        plt.clf() # Reset plt
        
        freq = np.fft.fftfreq(len(H), d=delta_t)
        freq_sorted = np.fft.fftshift(freq)
        H_sorted = np.real(np.fft.fftshift(H))
                
        plt.plot(freq_sorted, H_sorted)
        plt.title("Frequency Domain")
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.savefig("ramp_frequency_domain.png")
        plt.clf()"""
        
        """
        ## Calculating RMSE
        cut_off = kernel_args.get("cut_off", math.pow(2 * delta_t, -1)) 
        w_abs = np.abs(freq)
        H_ = np.where(w_abs <= cut_off, w_abs, 0)
        rmse = np.sqrt(np.mean(np.power(H - H_, 2)))
        print(f"RMSE: {rmse}")""" 
    
    elif kernel == "rect":
        cut_off = kernel_args.get("cut_off", math.pow(2 * delta_t, -1)) 
        freq = np.fft.fftfreq(len_S, d=delta_t)
        w_abs = np.abs(freq) # Jacobian
        H = np.where(w_abs <= cut_off, w_abs, 0)
        
        """
        # Save figure
        freq_sorted = np.fft.fftshift(freq)
        H_sorted = np.fft.fftshift(H)
        
        
        plt.plot(freq_sorted, H_sorted)
        plt.title("Ideal Ramp")
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.savefig("ideal_ramp.png")
        plt.clf()"""
    
    elif kernel == "none":
        H = np.ones_like(S, dtype=np.float32)
    
    else:
        raise ValueError("parameter 'kernel' should be one of 'ramp', 'rect', 'none'")
    
    # Multiple kernel
    S_ = S * H
    
    # Filtered projection
    Q = np.fft.ifft(S_, axis=1)
    
    return Q.real

def compute_convolution_projection(P, delta_t=0.5):
    # Init
    _, len_P = P.shape
    
    # 0이 중심인 n인덱스 생성        
    n_freq_order = np.fft.fftfreq(len_P) * len_P
    n = np.fft.fftshift(n_freq_order).astype(np.float32)
        
    h = np.zeros_like(n , dtype=np.float32)
    # n = 0
    h[n == 0] = 1 / (4 * delta_t ** 2)
    # n is even -> h = 0
    # n is odd
    h[n % 2 != 0] = -1 / (n[n % 2 != 0] * np.pi * delta_t) ** 2
    
    Q = np.apply_along_axis(lambda x: np.convolve(x, h, mode="same"), axis=1, arr=P)
    
    return Q * delta_t
    
def reconstruct_source(Q, x, y, thetas, t_min, delta_t=0.5, quarter_offset=False):
    H, W = x.shape
    len_t = Q.shape[1]
    sum_q = np.zeros((H, W))
    t_shift = delta_t / 4 if quarter_offset else 0
    
    for q, theta in zip(Q, thetas):
        t = np.cos(theta) * x + np.sin(theta) * y
        
        idx_float = (t - t_min - t_shift) / delta_t
        
        for i in range(H):
            for j in range(W):
                idx_floor = int(np.floor(idx_float[i, j]))
                idx_ceil = idx_floor + 1
                
                if (idx_floor >= 0 and idx_ceil < len_t):
                    q_floor = q[idx_floor]
                    q_ceil  = q[idx_ceil]
                    diff = idx_float[i, j] - idx_floor
                    
                    sum_q[i, j] += (1 - diff) * q_floor + diff * q_ceil
    
    return np.pi / len(thetas) * sum_q

def compute_spatial_slice(source, bias_x, bias_y, theta, r, D, delta_l= 0.5):
    # Init
    H, W = source.shape
    
    l = np.arange(-D/2, D/2+delta_l, delta_l)
    l_x = r * np.cos(theta) - l * np.sin(theta)
    l_y = r * np.sin(theta) + l * np.cos(theta)
    
    slice = []
    
    # Slice
    for k in range(len(l)):
        biased_x = l_x[k] - bias_x
        biased_y = l_y[k] - bias_y
        
        # Interpolation
        x_floor = int(np.floor(biased_x))
        y_floor = int(np.floor(biased_y))
        x_ceil = x_floor + 1
        y_ceil = y_floor + 1
        
        if (x_floor >= 0 and x_ceil < W and y_floor >= 0 and y_ceil < H):
            dx = biased_x - x_floor
            dy = biased_y - y_floor
                    
            p_tr = source[y_ceil , x_ceil] 
            p_tl = source[y_ceil , x_floor]
            p_br = source[y_floor, x_ceil]
            p_bl = source[y_floor, x_floor]
            
            slice.append((1 - dy) * ((1 - dx) * p_bl + dx * p_br) + dy * ((1 - dx) * p_tl + dx * p_tr))
        
    return np.array(slice)
        
if __name__ == "__main__":
    """Implement CT
    1. Source: f(x, y)
    2. Projection of source: P_theta(t)
    3. Fourier slice: S_theta(w)
    4. Filtered projection: Q_theta(t)
    5. Reconstruction
    """
    
    # Hyperparameter
    height, width = (128, 128)
    center = (70, 70)
    A, B = (50, 50)
    coefficient = 1
    range_angle = (0, 2 * np.pi)
    num_thetas = 360
    num_detectors = 512
    delta_t = 1
    delta_l = 0.5
    let_size = 5
    quarter_offset = False
    kernel_args = {}
    # Kernel
    kernel = "ramp"
    
    ## init
    # Diameter
    D = (height ** 2 + width ** 2) ** 0.5
    
    # Except 2pi(=360°) or pi(=180°)
    thetas = np.linspace(range_angle[0], range_angle[1], num_thetas + 1)[:-1]
    
    # W = 1 (1/pixel) -> delta = 1 / 2W = 0.5
    # Nyquist frequecy is 0.5 (1/pixel). But ,for better quality, it oversamples
    ts =  np.arange(-num_detectors//2, (num_detectors + 1) // 2, 1) * delta_t
    # ts = np.arange(-D/2 , D/2 + dleta_t, delta_t)

    # Grid: (x, y)
    R_x = np.arange(-width//2, (width + 1)//2 , 1)
    R_y = np.arange(-height//2, (height + 1)//2, 1)
    x, y = np.meshgrid(R_x, R_y)
    bias_x = x[0, 0]
    bias_y = y[0, 0]
    
    print("=== Start CT.py ===")
    
    ## 1.Source
    source = create_source(x, y, center, A, B, coefficient=coefficient)
    save_img(source, "source")
    
    ## 2.Projection of source: P_theta(t)
    P = compute_projection(source, bias_x, bias_y, thetas, ts, D, delta_t=delta_t, delta_l=delta_l, quarter_offset=quarter_offset)
    print(f"P shape: {P.shape}")
    save_img(P, "projection", type="min-max")
    
    # Apply detector let
    P = apply_detector_let(P, let_size)
    save_img(P, "projection__detector_lets", type="min-max")
    
    ## 3. Fourier slice: S_theta(w)
    S = compute_fourier_slice(P)
    print(f"S shape: {S.shape}")
    
    ## 4. Filtered projection: Q_theta(t)
    Q = compute_filtered_projection(S, delta_t=delta_t, kernel=kernel, kernel_args=kernel_args)
    Q = Q[:, :num_detectors] # Truncate pad
    print(f"Q shape: {Q.shape}")
    # print(f"Q min: {Q.min()}, Q max: {Q.max()}")
    save_img(Q, "filtered_projection", type="min-max")
    
    """
    ## 3, 4. Convolution Procjection
    Q = compute_convolution_projection(P, delta_t=delta_t)
    print(f"Q shape: {Q.shape}")
    save_img(np.clip(Q, P.min(), P.max()), "convolution_projection", type="min-max")
    
     ## 3, 4. Original Procjection
    Q = P
    """    
    
    ## 5. Reconstruction
    angle = 180
    recon = reconstruct_source(Q[:angle], x, y, thetas[:angle], t_min=ts[0], delta_t=delta_t)
    # print(f"Recon min: {recon.min()}, Recon max: {recon.max()}")
    save_img(recon, "reconstruction", type="min-max")
    
    ## 6. Slice Comparing
    theta = np.pi / 4 
    r = 35
    s_source = compute_spatial_slice(source, bias_x, bias_y, theta, r, D, delta_l=delta_l)
    s_recon  = compute_spatial_slice(recon,  bias_x, bias_y, theta, r, D, delta_l=delta_l)
    s_index = np.arange(0, len(s_source))
    
    # Save figure
    plt.title("Compare Spatial Slice")
    plt.plot(s_index, s_source, label='Source')
    plt.plot(s_index, s_recon,  label='Recon')
    plt.legend()
    plt.savefig("compare_spatial_slice.png")
    plt.clf() # Reset plt