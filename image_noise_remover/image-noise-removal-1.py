# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import os

# def load_image(path, grayscale=True):
#     """Load image with error handling"""
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Image not found: {path}")
#     flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
#     img = cv2.imread(path, flag)
#     if img is None:
#         raise ValueError(f"Could not read image: {path}")
#     return img

# def save_image(image, filename):
#     """Save image with validation"""
#     if image is None or image.size == 0:
#         raise ValueError("Cannot save empty image")
#     cv2.imwrite(filename, image)
#     print(f"Saved: {filename}")

# def plot_histogram(image, title="Histogram", bins=32):
#     """Plot normalized histogram"""
#     plt.figure()
#     plt.hist(image.flatten(), bins=bins, range=(0, 255), density=True)
#     plt.title(title)
#     plt.xlabel("Pixel Value")
#     plt.ylabel("Frequency")
#     plt.show()

# def apply_mean_filter(image, kernel_size):
#     """Apply mean filter with given kernel size"""
#     kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
#     return cv2.filter2D(image, -1, kernel)

# def apply_median_filter(image, kernel_size):
#     """Optimized median filter implementation"""
#     pad_size = kernel_size // 2
#     padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size,
#                               cv2.BORDER_CONSTANT, value=0)

#     result = np.zeros_like(image)
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             neighborhood = padded[i:i+kernel_size, j:j+kernel_size]
#             result[i,j] = np.median(neighborhood)
#     return result

# def template_matching(image, template, method):
#     """Perform template matching with visualization"""
#     result = cv2.matchTemplate(image, template, method)
#     result_display = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

#     # Find template location
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#     top_left = max_loc if method in [cv2.TM_CCORR, cv2.TM_CCORR_NORMED] else min_loc
#     h, w = template.shape
#     bottom_right = (top_left[0] + w, top_left[1] + h)

#     # Draw rectangle
#     result_rgb = cv2.cvtColor(result_display, cv2.COLOR_GRAY2BGR)
#     cv2.rectangle(result_rgb, top_left, bottom_right, (0, 0, 255), 2)

#     return result_rgb

# def main():
#     try:
#         # Load images
#         img1 = load_image("images/pic.jpeg")
#         img2 = load_image("forest.jpg")

#         # Plot histograms
#         plot_histogram(img1, "Histogram - Image 1")
#         plot_histogram(img2, "Histogram - Image 2")

#         # Mean filtering
#         for kernel_size in [5, 81]:
#             mean1 = apply_mean_filter(img1, kernel_size)
#             mean2 = apply_mean_filter(img2, kernel_size)
#             save_image(mean1, f"Mean_{kernel_size}x{kernel_size}_#1.png")
#             save_image(mean2, f"Mean_{kernel_size}x{kernel_size}_#2.png")

#         # Median filtering
#         for kernel_size in [5, 81]:
#             median1 = apply_median_filter(img1, kernel_size)
#             median2 = apply_median_filter(img2, kernel_size)
#             save_image(median1, f"Median_{kernel_size}x{kernel_size}_#1.png")
#             save_image(median2, f"Median_{kernel_size}x{kernel_size}_#2.png")

#         # Template matching
#         template = load_image("lane.png")
#         mural = load_image("beach.png")

#         methods = [
#             ("TM_CCORR", cv2.TM_CCORR),
#             ("TM_CCORR_NORMED", cv2.TM_CCORR_NORMED)
#         ]

#         for name, method in methods:
#             result = template_matching(mural, template, method)
#             cv2.imshow(f"Template Matching - {name}", result)
#             cv2.waitKey(3000)  # Display for 3 seconds
#             cv2.destroyAllWindows()
#             save_image(result, f"template_match_{name.lower()}.png")

#     except Exception as e:
#         print(f"Error: {e}")
#         print("Please ensure:")
#         print("1. All image files exist in the directory")
#         print("2. Images are in correct format (jpg/png)")
#         print("3. OpenCV is properly installed")

# if __name__ == "__main__":
#     main()