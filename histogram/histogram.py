import numpy as np
import matplotlib.pyplot as plt
import cv2


def computeNormGrayHistogram(image):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    bins = np.zeros(32)
    count = np.bincount(img.flatten())
    for i in range(len(count)):
        index = i + 1
        index = index * 8
        index = index - 1
        if i == 31:
            break
        bins[i] = np.sum(count[index - 8 + 1:index + 1])
    bins = bins / np.sum(bins)
    x = np.arange(0, 256, 8)
    plt.bar(x, bins, width=8)
    plt.title("Gray Histogram")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")
    plt.show()


def computeNormRGBHistogram(image):
    color_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)
    for channel_id, c in zip(channel_ids, colors):
        histogram, bins = np.histogram(
            color_img[:, :, channel_id], bins=32, range=(0, 256),density=True
        )
        plt.bar(bins[0:-1], histogram, color=c,width=8)
    plt.title("Color Histogram")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")
    plt.show()

def AHE(img,winSize):
    size = img.shape
    output = np.zeros((size[0],size[1]))
    image = cv2.copyMakeBorder(img,winSize//2,(winSize//2) + 1,winSize//2,(winSize//2)+1,cv2.BORDER_REPLICATE)
    for i in range(size[0]):
        for j in range(size[1]):
            rank=0
            contextual_region = image[i:i + winSize, j:j + winSize]
            for x in range(winSize):
                for y in range(winSize):
                    if image[i][j] > contextual_region[x][y]:
                        rank = rank + 1
            output[i][j] = ((rank * 255) / winSize**2)
    return output

# Comment out the following as needed

# image = cv2.imread('forest.jpg')
image = cv2.imread('images/pic.jpeg')
cv2.imshow("Normal",image)
flipped_image = cv2.flip(image,1)
cv2.imshow("Flipped",flipped_image)
b, g, r = cv2.split(image)
computeNormGrayHistogram(cv2.merge([b,g,2*r]))
computeNormRGBHistogram(cv2.merge([b,g,2*r]))
cv2.imshow('Doubled red channel',cv2.merge([b,g,2*r]))
cv2.waitKey(10000)
beach = cv2.imread('images/beach.png',0)
output=AHE(beach,129)
cv2.imwrite("images/Regular_HE.png",cv2.equalizeHist(beach))
cv2.imwrite("images/Ahe_Image_129.png",output)