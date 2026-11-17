import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read the color image
img = cv2.imread(sys.argv[1])

if img is None:
    print("Could not read image")
    sys.exit(1)


# Part 1

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create masks for bright and dark pixels
bright_mask = gray > 200
dark_mask = gray < 50

# Count the pixels
bright_count = np.sum(bright_mask)
dark_count = np.sum(dark_mask)

print("Pixels brighter than 200:", bright_count)
print("Pixels darker than 50:", dark_count)


# Part 2

# Method 1: assignment with a mask

bw_mask = gray > 128

bw1 = np.zeros_like(gray)
bw1[bw_mask] = 255


# Method 2: using np.where

bw2 = np.where(gray > 128, 255, 0)


# Part 3

# Create a mask with shape (H, W)
red_mask = gray > 200

# Make a copy of the original color image
red_img = img.copy()

# OpenCV uses BGR order, so red is (0, 0, 255)
red_img[red_mask] = [0, 0, 255]


# Display original image and image with red pixels

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB))
plt.title("Bright Pixels in Red")
plt.axis("off")

plt.tight_layout()
plt.show()