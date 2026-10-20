import numpy as np

a = np.array([[10, 100, 200],
              [50, 150, 250]], dtype=np.uint8)


a_dtype = a.dtype
a_shape = a.shape
a_ndim = a.ndim
a_size = a.size
a_nbytes = a.nbytes
a_1_2 = a[1, 2]
a_0_1 = a[0, 1]
a_pluse_10 = a + 10
a_minuse_20 = a - 20
a_multi_2 = a * 2
a_max = a.max()
a_max_0 = a.max(axis=0)
a_max_1 = a.max(axis=1)

print(f"a.dtype = {a_dtype}")
print(f"a.shape = {a_shape}")
print(f"a.ndim = {a_ndim}")
print(f"a.size = {a_size}")
print(f"a.nbytes = {a_nbytes}")
print(f"a[1, 2] = {a_1_2}")
print(f"a[0, 1] = {a_0_1}")
print(f"a + 10 = {a_pluse_10}")
print(f"a - 20 = {a_minuse_20}")
print(f"a * 2 = {a_multi_2}")
print(f"a.max = {a_max}")
print(f"a.max(axis=0) = {a_max_0}")
print(f"a.max(axis=1) = {a_max_1}")
