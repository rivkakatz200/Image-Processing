import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read image
img = cv2.imread(sys.argv[1])

if img is None:
    print("Could not read image")
    sys.exit(1)


# Print shapes
expressions = [
    ("img[10, 20]", img[10, 20]),
    ("img[:, :, 0]", img[:, :, 0]),
    ("img[100:200, 50:150]", img[100:200, 50:150]),
    ("img[::2, ::2]", img[::2, ::2]),
    ("img[::-1, :]", img[::-1, :]),
    ("img[:, ::-1, :]", img[:, ::-1, :]),
    ("img[0]", img[0]),
    ("img[:, 0]", img[:, 0]),
]

for expression, result in expressions:
    print(f"{expression}: shape: {result.shape}")


# Slicing, reduction and reflections
crop = img[100:200, 50:150]
small = img[::2, ::2]
flip_vertical = img[::-1, :]
flip_horizontal = img[:, ::-1, :]


# Convert BGR to RGB for matplotlib
crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
flip_vertical_rgb = cv2.cvtColor(flip_vertical, cv2.COLOR_BGR2RGB)
flip_horizontal_rgb = cv2.cvtColor(flip_horizontal, cv2.COLOR_BGR2RGB)


# Display results side by side
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

axes[0].imshow(crop_rgb)
axes[0].set_title("img[100:200, 50:150]")

axes[1].imshow(small_rgb)
axes[1].set_title("img[::2, ::2]")

axes[2].imshow(flip_vertical_rgb)
axes[2].set_title("img[::-1, :]")

axes[3].imshow(flip_horizontal_rgb)
axes[3].set_title("img[:, ::-1, :]")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# Create a black 10-pixel border
img[:10, :] = 0
img[-10:, :] = 0
img[:, :10] = 0
img[:, -10:] = 0


# Display image with border
cv2.imshow("Image with black border", img)
cv2.waitKey(0)
cv2.destroyAllWindows()