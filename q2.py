import numpy as np

black = np.zeros((100, 200), dtype=np.uint8)

white = np.full((100, 200), 255, dtype=np.uint8)

color = np.full((100, 200, 3), (153, 102, 255), dtype=np.uint8)

print(f"BLACK: size = {black.size}, ndim = {black.ndim}, shape = {black.shape}, nbytes = {black.nbytes}")
print(f"WHITE: size = {white.size}, ndim = {white.ndim}, shape = {white.shape}, nbytes = {white.nbytes}")
print(f"COLOR: size = {color.size}, ndim = {color.ndim}, shape = {color.shape}, nbytes = {color.nbytes}")

print(np.zeros((1080, 1920, 3), dtype=np.uint8).nbytes)   
print(np.zeros((3000, 4000, 3), dtype=np.uint8).nbytes)   

b = np.arange(12).reshape(3, 4)
print(b)
print(b.sum()) # Sum of the entire matrix
print(b.sum(axis=0)) # An array containing the sums of the matrix columns (0 – by columns)
print(b.sum(axis=1)) # An array containing the sums of the matrix rows (1 – by rows)
print(b[2, 3], b[-1, -1]) # b[2,3]: row index 2 (3rd row), column index 3 (4th column) → 11. b[-1,-1]: last row, last column → also 11.