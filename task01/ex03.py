import numpy as np

# Part 1
x = np.array([100, 200, 250], dtype=np.uint8)
print(x + 10)
print(x + 100)
print(x - 150)
print(x * 2)
print(x.astype(np.int16) + 100)
print(x * 1.5)
print((x * 1.5).dtype)

# Part 2
print(x.astype(np.int32) + 300)   

# Part 3
def add_safe(x, b):
    y = x.astype(np.int16) + b
    y = np.clip(y, 0, 255)
    return y.astype(np.uint8)
print(add_safe(x, 100))
print(add_safe(x, -150))

# Part 5
#   We use values that fit in int8: 100, 50, 127.
y = np.array([100, 50, 127], dtype=np.int8)
print(y + 10)    
print(y + 100)   
print(y - 60)    
print(y * 2)    
print(y.astype(np.int16) + 100)  
print(y * 1.5)                  
print((y * 1.5).dtype)          