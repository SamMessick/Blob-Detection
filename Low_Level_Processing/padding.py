import numpy as np

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