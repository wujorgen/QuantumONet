import pennylane as qml
import jax
import time
import sys

print(sys.argv)

print("this is printing from inside the python script")

for i in range(30):
    print(i)
    time.sleep(1.5)

