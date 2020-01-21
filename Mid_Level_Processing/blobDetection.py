import cv2
import numpy as np
from matplotlib import pyplot as plt


def generateBlobDetectionImage(img, co_ordinates):
    print('Blobs detected. Generating output image.')
    for blob in co_ordinates:
        y, x, r = blob
        cv2.circle(img, (x,y), int(r * 1.414), color=(0,0,255), thickness=1)
    cv2.imwrite('BlobDetectionResult.png', img)


def detect_blob(log_image_np, level, sigma):
    co_ordinates = [] #to store co ordinates
    (h,w) = (log_image_np.shape[0], log_image_np.shape[1])
    for i in range(10,h-10):
        for j in range(10,w-10):
            slice_img = log_image_np[i-1:i+2,j-1:j+2,:] #9*3*3 slice
            result = np.amax(slice_img) #finding maximum
            #print(slice_img.shape)
            x,y,z = (np.unravel_index(slice_img.argmax(),slice_img.shape))
            if result >= 17:  # threshold from Lowe 2004
                #  Verify that the coordinate isn't close to any other found maxima
                if (len(co_ordinates) == 0):
                    k = np.power(2, ((z + level) / 2))
                    co_ordinates.append((i + x - 1, j + y - 1, k * sigma))  # finding co-rdinates
                else:
                    new_point_found = True
                    for co_ordinate in co_ordinates:
                        if (i+x-1 - co_ordinate[0] < 8) and (j+y-1 - co_ordinate[1] < 8):
                            new_point_found = False
                            break
                        else:
                            continue
                    if (new_point_found):
                        k = np.power(2, ((5-z + level) / 2))
                        co_ordinates.append((i + x - 1, j + y - 1, k * sigma))  # finding co-rdinates
    return co_ordinates