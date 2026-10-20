import numpy as np

# 1. The guess is in the MD.
x = np.array([100, 200, 250], dtype=np.uint8)
print(x + 10)
print(x + 100)
print(x - 150)
print(x * 2)
print(x.astype(np.int16) + 100)
print(x * 1.5)
print((x * 1.5).dtype)

#2. x + 300: In NumPy 2.x, adding an integer literal that is out of range for the array's
#   dtype (uint8 range: 0-255) raises an OverflowError at the Python level.
#   NumPy no longer silently upcasts — the programmer must cast explicitly if needed.
#   Example of correct approach: x.astype(np.int32) + 300 → [400, 500, 550]
print(x.astype(np.int32) + 300)   # explicit cast to avoid OverflowError

#3. 
def add_safe(x, b):
    y = x.astype(np.int16) + b
    y = np.clip(y, 0, 255)
    return y.astype(np.uint8)
print(add_safe(x, 100))
print(add_safe(x, -150))

#4. Since x is of type np.uint8, the addition takes place within the uint8 range before np.clip receives the result.

#5. The guess is in the MD.
#   We use values that fit in int8: 100, 50, 127.
#   127 + 1 overflows to -128 (two's complement wrap).
y = np.array([100, 50, 127], dtype=np.int8)
print(y + 10)    
print(y + 100)   
print(y - 60)    
print(y * 2)    
print(y.astype(np.int16) + 100)  
print(y * 1.5)                  
print((y * 1.5).dtype)          
# Demonstrating 127+1 overflow:
z = np.array([127], dtype=np.int8)
print(z + 1)   # [-128] — wraps to minimum int8 value