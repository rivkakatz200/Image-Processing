import numpy as np

# Part 1
black = np.zeros((100, 200), dtype=np.uint8)

# Part 2
white = np.full((100, 200), 255, dtype=np.uint8)

# Part 3
color = np.full((100, 200, 3), (153, 102, 255), dtype=np.uint8)

# Part 4
print(f"BLACK: size = {black.size}, ndim = {black.ndim}, shape = {black.shape}, nbytes = {black.nbytes}")
print(f"WHITE: size = {white.size}, ndim = {white.ndim}, shape = {white.shape}, nbytes = {white.nbytes}")
print(f"COLOR: size = {color.size}, ndim = {color.ndim}, shape = {color.shape}, nbytes = {color.nbytes}")

# Part 5
print(np.zeros((1080, 1920, 3), dtype=np.uint8).nbytes)   
print(np.zeros((3000, 4000, 3), dtype=np.uint8).nbytes)   

# Part 6
b = np.arange(12).reshape(3, 4)
print(b)
print(b.sum())
print(b.sum(axis=0))
print(b.sum(axis=1))
print(b[2, 3], b[-1, -1]) 