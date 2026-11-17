import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read the image from the command line
img = cv2.imread(sys.argv[1])

if img is None:
    print("Could not read image")
    sys.exit(1)


# Part 2

# Method 1: Convert to float before adding
gray1 = (
    img[:, :, 0].astype(np.float32)
    + img[:, :, 1].astype(np.float32)
    + img[:, :, 2].astype(np.float32)
) / 3


# Method 2: Use np.mean with a floating-point type
gray2 = np.mean(img, axis=2, dtype=np.float32)


# Part 3

# Convert to grayscale using the formula
weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)

gray_formula = np.sum(
    img.astype(np.float32) * weights,
    axis=2
)


# Part 4

gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert both images to float before subtraction.
# This prevents uint8 subtraction from wrapping around.
difference = np.abs(
    gray_formula - gray_cv.astype(np.float32)
)

print("Maximum difference:", np.max(difference))


# Part 5

# Display simple average and weighted average
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray1, cmap="gray")
plt.title("Simple Average")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray_formula, cmap="gray")
plt.title("Weighted Average")
plt.axis("off")

plt.tight_layout()
plt.show()