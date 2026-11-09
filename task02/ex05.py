import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read image filename from command line argument
if len(sys.argv) != 2:
    print("Usage: python ex05.py <image file>")
    sys.exit(1)

filename = sys.argv[1]

# Load the image in grayscale – brightness operations make most sense on a
# single-channel image so we can clearly see overflow / clamp effects.
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: could not read image '{filename}'.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Helper from exercise 3 – safe addition that clamps to [0, 255]
# ─────────────────────────────────────────────────────────────────────────────
def add_safe(x, b):
    """Add scalar b to uint8 array x, clamping results to [0, 255]."""
    y = x.astype(np.int16) + b   # widen to int16 so no overflow happens yet
    y = np.clip(y, 0, 255)       # clamp every value to the valid uint8 range
    return y.astype(np.uint8)    # convert back to uint8 for display

# ─────────────────────────────────────────────────────────────────────────────
# Part 1 – brighten() function
# ─────────────────────────────────────────────────────────────────────────────
def brighten(img, b, method):
    """
    Add b to every pixel in a grayscale uint8 image.

    method="np"   -> plain NumPy addition  (wraps on overflow)
    method="cv2"  -> cv2.add              (saturates on overflow)
    method="clip" -> add_safe from ex03   (clips to [0, 255])
    """
    if method == "np":
        # Regular NumPy addition on uint8 wraps around modulo 256.
        # E.g.  250 + 10  ->  4  (not 255).
        # We cast to int16 first so that a negative b is accepted, then cast
        # back to uint8 which applies the modulo-256 wrapping behaviour.
        return (img.astype(np.int16) + b).astype(np.uint8)

    elif method == "cv2":
        # cv2.add performs saturating arithmetic: values above 255 become 255,
        # values below 0 become 0 – exactly what we want visually.
        # cv2.add requires both arguments to be arrays of the same shape and
        # dtype, so we create a constant array filled with b (clamped to uint8).
        b_clamped = int(np.clip(b, 0, 255))
        scalar_img = np.full(img.shape, b_clamped, dtype=np.uint8)
        return cv2.add(img, scalar_img)

    elif method == "clip":
        return add_safe(img, b)

    else:
        raise ValueError(f"Unknown method '{method}'")

# ─────────────────────────────────────────────────────────────────────────────
# Part 2 – Apply brighten with b=+100 in all three methods, display side by side
# ─────────────────────────────────────────────────────────────────────────────
b_pos = 100

result_np   = brighten(img, b_pos, "np")
result_cv2  = brighten(img, b_pos, "cv2")
result_clip = brighten(img, b_pos, "clip")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
images = [img, result_np, result_cv2, result_clip]
titles = ["Original", 'method="np" (wrap)', 'method="cv2" (saturate)', 'method="clip" (clamp)']

for ax, image, title in zip(axes, images, titles):
    # vmin/vmax are set explicitly so matplotlib does NOT rescale the
    # brightness automatically – we want to see the actual pixel values.
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")

plt.suptitle(f"Brighten by b={b_pos}", fontsize=13)
plt.tight_layout()
plt.savefig("ex05.png")   # save BEFORE show() – after show() the figure is cleared
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Part 4 – Repeat with b=-100 to observe darkening behaviour
# ─────────────────────────────────────────────────────────────────────────────
b_neg = -100

result_np_neg   = brighten(img, b_neg, "np")
# cv2.add/subtract: for negative b we should use cv2.subtract so that
# saturation goes to 0 instead of wrapping.  cv2.add with a negative
# np.uint8 value would wrap (e.g. -100 becomes 156 in uint8).
scalar_neg = np.full(img.shape, abs(b_neg), dtype=np.uint8)
result_cv2_neg  = cv2.subtract(img, scalar_neg)
result_clip_neg = brighten(img, b_neg, "clip")

fig3, axes3 = plt.subplots(1, 4, figsize=(18, 4))
images_neg = [img, result_np_neg, result_cv2_neg, result_clip_neg]
titles_neg = [
    "Original",
    'method="np" (wrap, b=-100)',
    'cv2.subtract (saturate, b=-100)',
    'method="clip" (clamp, b=-100)',
]

for ax, image, title in zip(axes3, images_neg, titles_neg):
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")

plt.suptitle(f"Darken by b={b_neg}", fontsize=13)
plt.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────
# Part 5 – Display without vmin/vmax so matplotlib auto-scales the brightness
# ─────────────────────────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 4, figsize=(18, 4))
images_auto = [img, result_np, result_cv2, result_clip]
titles_auto = [
    "Original (auto scale)",
    'method="np" (auto)',
    'method="cv2" (auto)',
    'method="clip" (auto)',
]

for ax, image, title in zip(axes4, images_auto, titles_auto):
    # WITHOUT vmin/vmax: matplotlib maps the darkest pixel to black and the
    # brightest pixel to white, regardless of the actual 0-255 range.
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis("off")

plt.suptitle("Without vmin/vmax – matplotlib auto-scales each image", fontsize=13)
plt.tight_layout()
plt.savefig("ex05_no_vmin.png")
plt.show()
