import numpy as np
import cv2
import time
from Mid_Level_Processing import scalespace, blobDetection

<<<<<<< HEAD
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
    test = np.sum(l)
    return l

def generateLaplacianScaleSpace(img, steps, layers, sigma):
    # Flags for tracking if an additional row or column was added
    layer = layers
    levels = []

    img_temp = img

    # Counter for printing out layers of pyramid
    i = 0

    while (layer > 0 and img_temp.shape[0] > 1):
        img_lapl = LoG(img_temp, sigma)
        img_blur = gaussian_blur(img_temp, sigma)
        img_down = cv2.resize(img_blur, dsize=(img_temp.shape[1] // 2, img_temp.shape[0] // 2),
                              interpolation=cv2.INTER_NEAREST)
        level = []
        level_temp = img_lapl
        level.append(img_lapl)
        for i in range(1, steps-1):
            level_temp = gaussian_blur(level_temp, sigma)
            level.append(level_temp)
        img_temp = img_down
        layer -= 1
        i = i + 1
        levels.append(level)
    i = -1
    j = -1
    for level in levels:
        i += 1
        j = 0
        for img in level:
            j += 1
            cv2.imwrite('Level %d Blurs %d.png' % (i, j), img)

def gaussian_blur(img, sigma):
    if len(img.shape) > 2:
        num_channels = 3
    else:
        num_channels = 1

    kernel_dim = 9

    kernel = gaussian_kernel(kernel_dim, kernel_dim, sigma)
    pad_len = kernel_dim//2
    padded_img = pad_image(img=img, d=num_channels, padding='wrap_around', d_pad=pad_len)
    convolution = convolve(padded_img, kernel, num_channels)
    return convolution

def LoG(img, sigma):
    if len(img.shape) > 2:
        num_channels = 3
    else:
        num_channels = 1

    kernel_dim = 5
    pad_len = kernel_dim // 2

    laplacian = LoG_kernel(kernel_dim, kernel_dim, sigma)

    padded_img = pad_image(img=img, d=num_channels, padding='wrap_around', d_pad=pad_len)
    convolution = convolve(padded_img, laplacian, num_channels)
    return convolution

def pad_image(img, d, padding, d_pad):

    # Obtain the size of the padding in the x and y directions
    if isinstance(d_pad, tuple):
        y_pad = d_pad[0]
        x_pad = d_pad[1]
    else:
        x_pad = y_pad = d_pad

    # Form padded image with original image inset
    if d == 1:
        padded_img = np.zeros((img.shape[0] + 2 * y_pad, img.shape[1] + 2 * x_pad))
    else:
        padded_img = np.zeros((img.shape[0] + 2 * y_pad, img.shape[1] + 2 * x_pad, d))

    if padding is 'wrap_around':
        padded_img[y_pad:img.shape[0] + y_pad, x_pad:img.shape[1] + x_pad] = img
        # Add wrap around elements
        if(y_pad != 0):
            if(x_pad != 0):
                # Add x and y padding: edges
                padded_img = add_y_copy_padding(padded_img, img, x_pad, y_pad)
                padded_img = add_x_copy_padding(padded_img, img, x_pad, y_pad)

                # Add x and y padding: corners
                padded_img = add_corner_padding(padded_img, img, x_pad, y_pad)

            else:
                # Add y padding: y-edges only
                padded_img = add_y_copy_padding(padded_img, img, x_pad, y_pad)

        elif(x_pad != 0):
            # Add x padding: x-edges only
            padded_img = add_y_copy_padding(padded_img, img, x_pad, y_pad)

    else:
        print('The padding you specified ' + padding + ' does not exist.')
        return

    return padded_img

def add_x_copy_padding(padded_img, img, x_pad, y_pad):
    # Left, Right
    padded_img[y_pad:img.shape[0] + y_pad, 0:x_pad] = img[:, img.shape[1] - x_pad:img.shape[1]]
    padded_img[y_pad:img.shape[0] + y_pad, img.shape[1] + x_pad:img.shape[1] + 2 * x_pad] = img[:, 0:x_pad]
    return padded_img


def add_y_copy_padding(padded_img, img, x_pad, y_pad):
    # Top, bottom
    padded_img[0:y_pad, x_pad:img.shape[1] + x_pad] = img[img.shape[0] - y_pad:img.shape[0], :]
    padded_img[img.shape[0] + y_pad:img.shape[0] + 2 * y_pad, x_pad:img.shape[1] + x_pad] = img[0:y_pad, :]
    return padded_img


def add_corner_padding(padded_img, img, x_pad, y_pad):
    # Upper left, upper right, lower left, lower right
    padded_img[0:y_pad, 0:x_pad] = img[img.shape[0]- y_pad:img.shape[0], img.shape[1]-x_pad:img.shape[1]]
    padded_img[0:y_pad, img.shape[1] + x_pad:img.shape[1] + 2 * x_pad] = img[img.shape[0]- y_pad:img.shape[0], 0:x_pad]
    padded_img[img.shape[0] + y_pad:img.shape[0] + 2 * y_pad, 0:x_pad] = img[0:y_pad, img.shape[1] - x_pad:img.shape[1]]
    padded_img[img.shape[0] + y_pad:img.shape[0] + 2 * y_pad, img.shape[1] + x_pad:img.shape[1] + 2 * x_pad] = img[0:y_pad, 0:x_pad]
    return padded_img

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
=======
# TODO Get rid of input print statements
>>>>>>> 81422c17a5216857c1d306b6c6be15f9d2dafd08

def main():
    awaiting_input = True
    while awaiting_input == True:

        name = input("Please enter the filename of what you would like to open: ")
        num_levels = int(input("Please also enter the number of image size levels you would like to create in your scale space: "))
        num_layers = int(input("Please input the number of image layers that you would like to generate per size level: "))
        awaiting_input = False
<<<<<<< HEAD
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        cv2.waitKey(0)
=======
        img = cv2.imread(name)
        gray_img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
>>>>>>> 81422c17a5216857c1d306b6c6be15f9d2dafd08
        if (img is None):
            awaiting_input = True
            print('Filename not found %s.' % (name))

    print('Generating Laplacian Scale Space for blob detection.')
<<<<<<< HEAD
    scaleSpace = generateLaplacianScaleSpace(img, 5, 4, 0.5)
=======
    t0 = time.time()
    sigma = 3
    laplacianScaleSpace = scalespace.generateLaplacianScaleSpace(gray_img, num_layers, num_levels, sigma)
    level0 = laplacianScaleSpace[0][:,:,0]
    level1 = laplacianScaleSpace[0][:,:,1]
    level2 = laplacianScaleSpace[0][:, :, 2]
    level3 = laplacianScaleSpace[0][:, :, 3]

    # Detect blobs and generate marked output image
    print('Generating Blob detection image.')

    level_counter = 0

    for level in laplacianScaleSpace:
        co_ordinates = list(set(blobDetection.detect_blob(level, level_counter, sigma)))
        blobDetection.generateBlobDetectionImage(img, co_ordinates)
        level_counter += 1

    tf = time.time()
    runtime = tf-t0
    print('Total runtime: %3.5f' %(runtime) )
>>>>>>> 81422c17a5216857c1d306b6c6be15f9d2dafd08

if __name__ == '__main__':
    main()