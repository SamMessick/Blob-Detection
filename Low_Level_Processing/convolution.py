import numpy as np

def convolve(padded_img, kernel, d):
    h_kernel = kernel.shape[0]
    w_kernel = kernel.shape[1]

    if (d == 1):
        # Perform 1-layer convolution for grayscale

        # Create a d-channel, h by w sized image to store the convolution results in
        convolved_img = np.zeros((padded_img.shape[0] - 2 * (h_kernel // 2), padded_img.shape[1] - 2 * (w_kernel // 2)))

        for i in range(0, padded_img.shape[1] - 2 * (w_kernel // 2)):
            for j in range(0, padded_img.shape[0] - 2 * (h_kernel // 2)):
                convolved_img[j, i] = np.sum(kernel * padded_img[j:j + h_kernel, i:i + w_kernel])

    else:
        # Perform multilayer convolution for multi-channel images

        # Create a d-channel, h by w sized image to store the convolution results in
        convolved_img = np.zeros((padded_img.shape[0] - 2 * (h_kernel // 2), padded_img.shape[1] - 2 * (w_kernel // 2), d))

        for i in range(0, padded_img.shape[1] - 2 * (w_kernel // 2)):
            for j in range(0, padded_img.shape[0] - 2 * (h_kernel // 2)):
                for channel in range(0, d):
                    convolved_img[j, i, channel] = np.sum( kernel * padded_img[j:j+h_kernel, i:i+w_kernel, channel] )

    return convolved_img