import sys
import cv2
import matplotlib.pyplot as plt

# Read image filename from command line argument
if len(sys.argv) != 2:
    print("Usage: python ex04.py <image file>")
    sys.exit(1)

filename = sys.argv[1]

# ─────────────────────────────────────────
# Part 1 – Read image and print basic info
# ─────────────────────────────────────────
img = cv2.imread(filename)

# Part 2 – Check for invalid filename (cv2.imread returns None on failure)
if img is None:
    print(f"Error: could not read image '{filename}'. Check that the file exists.")
    sys.exit(1)

print("type :", type(img))       # should be <class 'numpy.ndarray'>
print("dtype:", img.dtype)       # should be uint8
print("shape:", img.shape)       # (height, width, 3)  – channels are BGR
print("top-left pixel (BGR):", img[0, 0])  # value of the upper-left pixel

# ─────────────────────────────────────────────────────────────────────────────
# Part 3 – Show the image twice: once without conversion, once after BGR→RGB
# ─────────────────────────────────────────────────────────────────────────────

# OpenCV reads images in BGR order, but matplotlib expects RGB order.
# Without conversion the red and blue channels are swapped → colours look wrong.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img)                  # wrong colours: blue ↔ red are swapped
axes[0].set_title("No conversion (BGR interpreted as RGB)")
axes[0].axis("off")

axes[1].imshow(img_rgb)              # correct colours after BGR → RGB
axes[1].set_title("After BGR → RGB conversion")
axes[1].axis("off")

# ─────────────────────────────────────────────────────────────────────────────
# Part 4 – Show each colour channel separately as a grayscale image
# ─────────────────────────────────────────────────────────────────────────────

# In OpenCV the channel order is Blue=0, Green=1, Red=2
b_channel = img[:, :, 0]   # Blue channel
g_channel = img[:, :, 1]   # Green channel
r_channel = img[:, :, 2]   # Red channel

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

# cmap='gray' tells matplotlib to display a single-channel array as grayscale.
# vmin/vmax are set explicitly so the brightness is comparable across channels.
axes2[0].imshow(b_channel, cmap='gray', vmin=0, vmax=255)
axes2[0].set_title("Blue channel")
axes2[0].axis("off")

axes2[1].imshow(g_channel, cmap='gray', vmin=0, vmax=255)
axes2[1].set_title("Green channel")
axes2[1].axis("off")

axes2[2].imshow(r_channel, cmap='gray', vmin=0, vmax=255)
axes2[2].set_title("Red channel")
axes2[2].axis("off")

# Save the channel figure BEFORE calling plt.show() because show() clears it
plt.savefig("ex04.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Part 5 – Read the same image in grayscale and print the shape
# ─────────────────────────────────────────────────────────────────────────────
img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
print("Grayscale shape:", img_gray.shape)   # (height, width) – no channel dimension
