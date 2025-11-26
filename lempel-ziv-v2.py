#
# lempel-ziv-v2.py : Lempel-Ziv complexity index with Numba and Numpy
#

import time
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def complexityLempelZiv(s):
    complexity = 1
    prefix_length = 1
    length_component = 1
    max_length_component = 1
    pointer = 0
    n = len(s)

    while prefix_length + length_component <= n:
        if s[pointer + length_component - 1] == s[prefix_length + length_component - 1]:
            length_component += 1
        else:
            if length_component > max_length_component:
                max_length_component = length_component

            pointer += 1

            if pointer == prefix_length:
                complexity += 1
                prefix_length += max_length_component
                pointer = 0
                max_length_component = 1

            length_component = 1

    if length_component != 1:
        complexity += 1

    return complexity


def read_string():
    print("Menù:\n")
    print("Read from [F]ile\n")
    print("Read from [K]eyboard\n")
    x = input("> ").strip()

    if x.upper() == "F":
        file_name = input("What file: ").strip()
        try:
            with open(file_name, encoding="utf-8") as f:
                s = f.read()
        except FileNotFoundError:
            print(f"File {file_name} not found.")
            quit()
    else:
        s = input("give me a string: ")

    return s


def main():
    s = read_string()

    if len(s) == 0:
        print("Empty input, complexity = 0")
        return

    # Convert string to array of uint8 for maximum speed with numba
    arr = np.frombuffer(s.encode("utf-8"), dtype=np.uint8)

    t0 = time.perf_counter()
    c = complexityLempelZiv(arr)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    
    print(f"\n\nLempel-Ziv complexity index for\n\n{s[:256]}...\n\nis {c}")
    print(f"\nElapsed time: {elapsed:.6f} seconds\n")

main()
