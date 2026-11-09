import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read image filename from command line argument
if len(sys.argv) != 2:
    print("Usage: python ex06.py <image file>")
    sys.exit(1)

filename = sys.argv[1]

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: could not read image '{filename}'.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Part 1 – img * 1.5 : dtype and maximum value
# ─────────────────────────────────────────────────────────────────────────────

# Multiplying a uint8 array by a float causes NumPy to upcast the result to
# float64 automatically so that the values are not lost.
result_float = img * 1.5
print("dtype of img * 1.5 :", result_float.dtype)   # float64
print("max value           :", result_float.max())   # can be > 255

# ─────────────────────────────────────────────────────────────────────────────
# Part 2 – Convert back to uint8 and display next to the original
# ─────────────────────────────────────────────────────────────────────────────

# np.clip first to make sure no value exceeds 255, then cast to uint8.
# Without clip, values > 255 would wrap around (e.g. 300 -> 44).
result_k1_5 = np.clip(result_float, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(result_k1_5, cmap='gray', vmin=0, vmax=255)
axes[1].set_title("k=1.5 (float, clipped)")
axes[1].axis("off")

# ─────────────────────────────────────────────────────────────────────────────
# Part 3 – Compare img * 1.5 (float) with img * 2 (integer)
# ─────────────────────────────────────────────────────────────────────────────

# img * 2 stays in uint8 because 2 is a Python int, and NumPy keeps the
# original dtype. This causes modulo-256 wrapping: 200*2=400 -> 400%256=144.
result_k2_int = img * 2           # uint8, wraps around
result_k2_clip = np.clip(img.astype(np.uint16) * 2, 0, 255).astype(np.uint8)

print("\ndtype of img * 2 (int) :", result_k2_int.dtype)   # uint8 – wraps!
print("max value of img * 2   :", result_k2_int.max())

axes[2].imshow(result_k2_int, cmap='gray', vmin=0, vmax=255)
axes[2].set_title("k=2 (int, wraps!)")
axes[2].axis("off")

# ─────────────────────────────────────────────────────────────────────────────
# Part 4 – k=0.5 : image becomes darker
# ─────────────────────────────────────────────────────────────────────────────

# k=0.5 halves every pixel value. The result is float64 (same as k=1.5),
# so we clip and cast back. The image gets darker because all values shrink.
result_k0_5 = np.clip(img * 0.5, 0, 255).astype(np.uint8)

axes[3].imshow(result_k0_5, cmap='gray', vmin=0, vmax=255)
axes[3].set_title("k=0.5 (darker)")
axes[3].axis("off")

plt.suptitle("Contrast scaling", fontsize=13)
plt.tight_layout()
plt.savefig("ex06.png")   # save BEFORE show()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Part 5 (bonus) – Stretch contrast around the mean
# ─────────────────────────────────────────────────────────────────────────────
# Formula:  result = (img - m) * k + m
# where m is the mean pixel value.
#
# Pixels ABOVE the mean get pushed further up (brighter).
# Pixels BELOW the mean get pushed further down (darker).
# The mean itself stays unchanged.
#
# img - m produces a float64 array (because m is a float), so there is NO
# risk of uint8 overflow/underflow at the subtraction step.
# However, the final result CAN go outside [0,255], so we still need clip.

m = img.mean()
print(f"\nImage mean: {m:.1f}")

k_values = [0.5, 1.5, 2.5]
fig2, axes2 = plt.subplots(1, len(k_values) + 1, figsize=(18, 4))

axes2[0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes2[0].set_title("Original")
axes2[0].axis("off")

for i, k in enumerate(k_values):
    stretched = (img.astype(np.float64) - m) * k + m
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    axes2[i + 1].imshow(stretched, cmap='gray', vmin=0, vmax=255)
    axes2[i + 1].set_title(f"k={k} (around mean)")
    axes2[i + 1].axis("off")

plt.suptitle("Contrast stretch around mean", fontsize=13)
plt.tight_layout()
plt.show()
