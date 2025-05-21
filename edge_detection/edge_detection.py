import numpy as np
import cv2
import sys

img = cv2.imread('lane.png', 0)

#Smoothing
k= np.array([[2,4,5,6,2], [4,9,12,9,4], [5,12,15,12,5], [4,9,12,9,4], [2,4,5,4,2]])
k = k/159
img = cv2.filter2D(img,-1,k)
cv2.imshow("Image after Smoothing",img)
cv2.waitKey(10000)

#Finding Gradients
k_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
k_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
G_x = cv2.filter2D(img,-1,k_x)
G_y = cv2.filter2D(img,-1,k_y)

G = np.uint8(np.hypot(G_x,G_y))
G_theta = np.arctan2(G_y,G_x)

cv2.imshow("Original Gradient Magnitude Image",G)
cv2.waitKey(10000)

NMS = np.zeros((G.shape[0],G.shape[1]),dtype=np.uint8)
for i in range(1, G.shape[0] - 1):
    for j in range(1, G.shape[1] - 1):
            pixel_direction = 255
            pixel_process = 255
            # 0
            if (0 <= G_theta[i, j] < 22.5) or (157.5 <= G_theta[i, j] <= 180):
                pixel_direction = G[i, j + 1]
                pixel_process = G[i, j - 1]
            # 45
            elif (22.5 <= G_theta[i, j] < 67.5):
                pixel_direction = G[i - 1, j - 1]
                pixel_process = G[i + 1, j + 1]
            # 90
            elif (67.5 <= G_theta[i, j] < 112.5):
                pixel_direction = G[i + 1, j]
                pixel_process = G[i - 1, j]
            # 135
            elif (112.5 <= G_theta[i, j] < 157.5):
                pixel_direction = G[i - 1, j + 1]
                pixel_process = G[i + 1, j - 1]

            if (G[i, j] >= pixel_direction) and (G[i, j] >= pixel_process):
                NMS[i, j] = G[i, j]

            else:
                NMS[i, j] = 0

cv2.imshow("Image after NMS",NMS)
cv2.waitKey(10000)

#Thresholding
np.set_printoptions(threshold=sys.maxsize)
highThreshold = NMS.max() * 0.145;

res = np.zeros((NMS.shape[0], NMS.shape[1]),dtype=np.uint8)

strong = np.uint8(255)
strong_i, strong_j = np.where(NMS >= highThreshold)
res[strong_i, strong_j] = strong

cv2.imshow("Image after Thresholding",res)
cv2.waitKey(10000)