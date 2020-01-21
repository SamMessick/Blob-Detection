import numpy as np
from Low_Level_Processing import convolution
from Low_Level_Processing import padding

def gaussian_kernel(h, w, sigma):
    x, y = np.meshgrid(np.linspace(-(h-1) // 2, (h-1) // 2, h), np.linspace(-(w-1) // 2, (w-1) // 2, w))
    d = -(x * x + y * y)/(2*sigma**2)
    mu = 0.0
    g = np.exp(d)
    g = g*(g>np.finfo('double').eps*np.amax(g)).astype(int)
    sum = np.sum(g)
    if sum != 0:
        g = g / sum
    test = np.sum(g)
    return g

def LoG_kernel(h, w, sigma):
    x, y = np.meshgrid(np.linspace(-(h - 1) // 2, (h - 1) // 2, h), np.linspace(-(w - 1) // 2, (w - 1) // 2, w))
    d = -(x * x + y * y) / (2 * sigma ** 2)
    mu = 0.0
    g = np.exp(d)
    g = g*(g>np.finfo('double').eps*np.amax(g)).astype(int)
    test = (g>np.finfo('double').eps*np.amax(g)).astype(int)
    sum = np.sum(g)
    if sum != 0:
        g = g / sum
    l = g * (x * x + y * y - 2 * sigma ** 2) / (sigma ** 2)
    l = l - np.sum(l) / (h * w)
    return l

def gaussian_blur(img, sigma):
    if len(img.shape) > 2:
        num_channels = 3
    else:
        num_channels = 1

    kernel_dim = 9

    kernel = gaussian_kernel(kernel_dim, kernel_dim, sigma)
    pad_len = kernel_dim//2
    padded_img = padding.pad_image(img=img, d=num_channels, padding='wrap_around', d_pad=pad_len)
    result = convolution.convolve(padded_img, kernel, num_channels)
    return result

def LoG(img, sigma):
    if len(img.shape) > 2:
        num_channels = 3
    else:
        num_channels = 1

    kernel_dim = 5
    pad_len = kernel_dim // 2

    laplacian = LoG_kernel(kernel_dim, kernel_dim, sigma)

    padded_img = padding.pad_image(img=img, d=num_channels, padding='wrap_around', d_pad=pad_len)
    result = convolution.convolve(padded_img, laplacian, num_channels)
    return result