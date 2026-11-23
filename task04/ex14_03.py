import cv2
import numpy as np

img = cv2.imread("ex14.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

# pink flower
mask = (h >= 160) & (h <= 179) & (s >= 100) & (v >= 100)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

out[mask] = img[mask]

cv2.imshow("Original", img)
cv2.imshow("Result", out)

cv2.waitKey(0)
cv2.destroyAllWindows()