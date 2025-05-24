import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# Histograms
img = cv2.imread('forest.jpg', 0)
img2 = cv2.imread('lane.png', 0)
plt.hist(img.flatten(), bins=32, range=(0, 255), density=True)
plt.show()
plt.hist(img2.flatten(), bins=32, range=(0, 255), density=True)
plt.show()

img_median_new1 = np.zeros([img.shape[0], img.shape[1]])
img_median_new2 = np.zeros([img2.shape[0], img2.shape[1]])

# Mean - 5x5
kernel = np.ones((5, 5), np.uint8) / 25
img_mean_new1 = cv2.filter2D(img, -1, kernel)
img_mean_new2 = cv2.filter2D(img2, -1, kernel)
cv2.imwrite("Mean_5x5_#1.png", img_mean_new1)
cv2.imwrite("Mean_5x5_#2.png", img_mean_new2)

# Mean - 81x81
kernel = np.ones((81, 81), np.uint8) / 6561
img_mean_new1 = cv2.filter2D(img, -1, kernel)
img_mean_new2 = cv2.filter2D(img2, -1, kernel)
cv2.imwrite("Mean_81x81_#1.png", img_mean_new1)
cv2.imwrite("Mean_81x81_#2.png", img_mean_new2)

pad1 = cv2.copyMakeBorder(img, 5 // 2, (5 // 2) + 1, 5 // 2, (5 // 2) + 1, cv2.BORDER_CONSTANT, None, value=0)
pad2 = cv2.copyMakeBorder(img2, 81 // 2, (81 // 2) + 1, 81 // 2, (81 // 2) + 1, cv2.BORDER_CONSTANT, None, value=0)

# Median - 5x5 and 81x81 just change temp values accordingly to accommodate 81x81
for i in range(1, img.shape[0] - 1):
    for j in range(1, img.shape[1] - 1):
        temp = np.zeros((5, 5))
        temp2 = np.zeros((81, 81))
        temp = pad1[i:i + 5, j:j + 5]
        temp2 = pad2[i:i + 81, j:j + 81]
        median1 = np.median(temp)
        median2 = np.median(temp2)
        img_median_new1[i, j] = median1
        img_median_new2[i, j] = median2

img_new1 = img_median_new1.astype(np.uint8)
img_new2 = img_median_new2.astype(np.uint8)
cv2.imwrite('new_median_filtered1.png', img_new1)
cv2.imwrite('new_median_filtered2.png', img_new2)

# Template - no cross
template = cv2.imread('beach.png', 0)
mural = cv2.imread('images/pic.jpeg', 0)
chad = cv2.matchTemplate(img, template, cv2.TM_CCORR)
chadNew = cv2.resize(chad, (600, 600))
chadder = cv2.rectangle(chadNew, (250, 250), (350, 350), (255, 225, 225), 2)
cv2.imshow('no_cross.png', chadder)
cv2.waitKey(10000)

# Template - with cross
template = cv2.imread('beach.png', 0)
mural = cv2.imread('images/pic.jpeg', 0)
chad = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
chadNew = cv2.resize(chad, (600, 600))
chadder = cv2.rectangle(chadNew, (245, 315), (345, 415), (255, 225, 225), 2)
cv2.imshow('with_cross.png', chadNew)
cv2.waitKey(10000)