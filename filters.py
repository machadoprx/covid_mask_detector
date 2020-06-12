import cv2
import sys
import numpy as np
import imageio
from scipy.fftpack import fftn, ifftn, fftshift

def apply_clahe(g):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(g)

def gaussian_filter(k=3, sigma=1.0):
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp( -(1/2)*(np.square(x) + np.square(y))/np.square(sigma) )
    return filt/np.sum(filt)

def laplace_sharp(g, C):
    lap_op = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    del_g = fft_imagefilter(g, lap_op)
    return g - (C * del_g)

def padd_filter(w, size):
    w_size = w.shape[0]
    pad1 = (size//2)-w_size//2
    return np.pad(w, (pad1,pad1-1), "constant",  constant_values=0)

def fft_imagefilter(g, w):
    wp = padd_filter(w, g.shape[0])
    W = fftn(wp)
    G = fftn(g)
    R = np.multiply(W,G)
    
    r = np.real(fftshift(ifftn(R))) 
    return r

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def norm(a, min, max):
    n = (a - np.amin(a)) / (np.amax(a) - np.amin(a))
    return (n * (max - min) + min)