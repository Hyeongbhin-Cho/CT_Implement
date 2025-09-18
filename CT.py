# CT.py
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def save_img(img, name, save_dir="./"):
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"{name}.png"), img)

def create_source(x, y, center, A, B, coefficient=1):    
    return np.where((x - center[0]) ** 2 / A ** 2 + (y - center[1]) ** 2 / B ** 2 <= 1, coefficient, 0)

def compute_projection(source, x, y, thetas, ts):
    # Init
    P = np.zeros((len(thetas), len(ts)))
    
    for i, theta in enumerate(thetas):
        for j, t in enumerate(ts):
            # Distance between ax+by+c=0 and (x_0, y_0) is |ax_0+by_0+c|/(a^2 + b^2) ** 0.5
            # Thus the distance between t and any point(x, y) is |x*cos + y*sin +-t|
            d = np.abs((np.cos(theta) * x + np.sin(theta) * y) - t)
            # Compute projeciton sweight according to d
            ray = np.maximum(np.zeros_like(d), np.ones_like(d) - d) 
            # Project
            P[i, j] = np.sum(source * ray, axis=(0, 1), keepdims=False) 
            
    return P

def compute_fourier_slice(P):
    _, num_ts = P.shape
    
    # Zero pad
    P = np.pad(P, ((0,0), (0, num_ts)), mode="constant", constant_values=0)
    
    # Fourier slice
    S = np.fft.fft(P, axis=1)
    
    return S

def compute_filtered_projection(S, delta=0.5, kernel="none", **kwargs):
    # Jacobian
    _, len_S = S.shape
    w_abs = np.abs(np.fft.fftfreq(len_S, d=delta))
    
    # kernel
    H = None
    if kernel == "none":
        H = np.ones_like(w_abs)
    elif kernel == "ramp":
        cut_off = kwargs["cut_off"]
        H = np.where(w_abs <= cut_off, 1, 0)
    elif kernel == "gaussian":
        sigma = kwargs["sigma"]
        H = np.exp(-0.5 * w_abs ** 2 / sigma ** 2)
    else:
        raise ValueError("parameter 'kernel' should be one of 'none', 'ramp', 'gaussian'")
    
    # Multiple filter
    S_ = S * w_abs * H
    
    # Filtered projection
    Q = np.fft.ifft(S_, axis=1)
    
    return Q.real

def reconstruct_source(Q, x, y, thetas, t_min, delta=0.5):
    H, W = x.shape
    len_t = Q.shape[1]
    sum_q = np.zeros((H, W))
    
    for q, theta in zip(Q, thetas):
        t = np.cos(theta) * x + np.sin(theta) * y
        
        n_hat = (t - t_min) / delta
        n_floor = np.floor(n_hat).astype(np.int32)
        n_ceil = n_floor + 1
        
        diff_n = n_hat - n_floor
        w_floor = 1 - diff_n
        w_ceil = diff_n
        
        for j in range(H):
            for i in range(W):
                t_floor = n_floor[j, i]
                t_ceil = n_ceil[j, i]
                
                q_floor = q[t_floor] if (t_floor >= 0 and t_floor < len_t) else 0.0
                q_ceil = q[t_ceil] if (t_ceil >= 0 and t_ceil < len_t) else 0.0
                
                sum_q[j, i] += (w_floor[j, i] * q_floor + 
                                    w_ceil[j, i] * q_ceil)

    return np.pi / len(thetas) * sum_q
        
        
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
    center = (0, 0)
    A, B = (50, 30)
    coefficient = 1
    range_angle = (0, 2 * np.pi)
    num_thetas = 360
    delta = 0.5
    # Kernel
    kernel = "ramp"
    cut_off = 0.5
    
    ## init
    # Diameter
    D = (height ** 2 + width ** 2) ** 0.5
    
    # Except 2pi(=360°) or pi(=180°)
    thetas = np.linspace(range_angle[0], range_angle[1], num_thetas + 1)[:-1]
    
    # W = 1 (1/pixel) -> delta = 1 / 2W = 0.5
    # Nyquist frequecy is 0.5 (1/pixel). But ,for better quality, it oversamples
    ts =  np.arange(-D/2, D/2 + delta, delta)
    
    # Grid: (x, y)
    R_x = np.linspace(-width//2, width//2 , width)
    R_y = np.linspace(-height//2, height//2, height)
    x, y = np.meshgrid(R_x, R_y)
    
    ## 1.Source
    source = create_source(x, y, center, A, B, coefficient=coefficient)
    save_img(source, "source")
    
    ## 2.Projection of source: P_theta(t)
    P = compute_projection(source, x, y, thetas, ts)
    
    ## 3. Fourier slice: S_theta(w)
    S = compute_fourier_slice(P)
    
    ## 4. Filtered projection: Q_theta(t)
    Q = compute_filtered_projection(S, delta=delta, kernel=kernel , cut_off=cut_off)
    
    # 5. Reconstruction
    recon = reconstruct_source(Q[:180], x, y, thetas[:180], t_min=-D/2, delta=delta)
    save_img(recon, "reconstruction")
