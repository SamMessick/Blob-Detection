import cv2
import numpy as np
from Low_Level_Processing import kernelops
import matplotlib.pyplot as plt

def generateLaplacianScaleSpace(img, steps, layers, sigma):
    # Flags for tracking if an additional row or column was added
    layer = layers
    levels = []

    img_temp = img

    # Counter for printing out layers of pyramid
    i = 0

    while (layer > 0 and img_temp.shape[0] > 1):
        print('Generating level %d' % (layers - layer))
        img_lapl = kernelops.LoG(img_temp, sigma)
        img_blur = kernelops.gaussian_blur(img_temp, sigma)
        img_down = cv2.resize(img_blur, dsize=(img_temp.shape[1] // 2, img_temp.shape[0] // 2),
                              interpolation=cv2.INTER_NEAREST)
        level = img_lapl
        level_temp = level
        print('Layer 0 generated')
        for i in range(1, steps):
            level_temp = kernelops.gaussian_blur(level_temp, sigma)
            level_temp += np.abs(np.amin(level_temp))
            level = np.dstack((level, level_temp))
            print('Layer %d generated' % (level.shape[2]-1))
        img_temp = img_down
        layer -= 1
        i = i + 1
        levels.append(level)
    i = -0
    #for level in levels:
    #    i += 1
    #    j = 0
    #    for img in level:
    #        j += 1
    #        cv2.imwrite('Level %d Blurs %d.png' % (i, j), img)

    return levels