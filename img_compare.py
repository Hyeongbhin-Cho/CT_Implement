# img_compare.py
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os



if __name__ == "__main__":
    img1 = np.load("recon_wo.npy")
    img2 = np.load("recon_w.npy")
    
    img_diff = img1 - img2
    
    min = img_diff.min()
    max = img_diff.max()
    
    img_diff = (img_diff - min) / (max - min)
    
    plt.imshow(img_diff, cmap='gray')
    plt.show()