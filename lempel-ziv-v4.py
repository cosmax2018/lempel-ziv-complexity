# LZ complexity in O(N) using a Suffix Automaton (Numba-friendly implementation)
# - Works on bytes
# - Maps alphabet to only used symbols (reduces memory)
# - Core routines are njit-compiled for speed
# - Measures elapsed time with time.perf_counter()

import sys
import time
import numpy as np
from numba import njit

# ---------------------------
# Helper: build mapping of bytes -> dense alphabet [0..m-1]
# ---------------------------
def build_byte_mapping(data: bytes):
    present = np.zeros(256, dtype=np.int32)
    for b in data:
        present[b] = 1
    symbols = np.nonzero(present)[0].tolist()
    m = len(symbols)
    map256 = np.full(256, -1, dtype=np.int32)
    for i, b in enumerate(symbols):
        map256[b] = i
    return map256, m

# ---------------------------
# Suffix Automaton core (Numba-compatible)
# We store:
#   next_arr: int32 array shape (max_states, m) filled with -1
#   link: int32 array length max_states
#   length: int32 array length max_states
#   size_last: int32 array length 2 -> [size, last_state] (mutable container)
# ---------------------------

@njit
def sa_extend(ch, next_arr, link, length, size_last):
    """
    Extend suffix automaton with character index `ch`.
    Mutates next_arr, link, length and size_last in-place.
    """
    size = size_last[0]
    last = size_last[1]

    cur = size
    size += 1
    length[cur] = length[last] + 1

    p = last
    # add transition p --ch--> cur for all p that don't have it
    while p != -1 and next_arr[p, ch] == -1:
        next_arr[p, ch] = cur
        p = link[p]

    if p == -1:
        link[cur] = 0
    else:
        q = next_arr[p, ch]
        if length[p] + 1 == length[q]:
            link[cur] = q
        else:
            clone = size
            size += 1
            length[clone] = length[p] + 1
            # copy transitions q -> clone
            # note: this copies m entries; amortized cost is fine
            m = next_arr.shape[1]
            for cc in range(m):
                next_arr[clone, cc] = next_arr[q, cc]

            link[clone] = link[q]
            while p != -1 and next_arr[p, ch] == q:
                next_arr[p, ch] = clone
                p = link[p]
            link[q] = clone
            link[cur] = clone

    last = cur
    size_last[0] = size
    size_last[1] = last

@njit
def lz_factor_count(arr_mapped, next_arr, link, length, size_last):
    """
    Compute number of LZ factors (Lempel-Ziv factorization where a factor
    is the longest prefix that appeared before, plus the next char).
    Works by:
      - for each position i, greedily match as long as automaton from state 0 has transitions
      - consume matched length; if match didn't reach end, also consume the next char
      - extend automaton by each consumed char (sa_extend)
    """
    n = arr_mapped.shape[0]
    i = 0
    count = 0

    while i < n:
        v = 0
        j = i
        # follow transitions while possible (matching substring present in processed prefix)
        while j < n:
            ch = arr_mapped[j]
            nxt = next_arr[v, ch]
            if nxt == -1:
                break
            v = nxt
            j += 1

        if j == n:
            consumed = j - i   # matched to end — consume all matched chars
            if consumed == 0:
                # no match and at end -> consume one (should not usually happen)
                consumed = 1
        else:
            consumed = (j - i) + 1  # take matched prefix plus next new char

        # extend automaton by consumed characters
        k = 0
        while k < consumed and i + k < n:
            sa_extend(arr_mapped[i + k], next_arr, link, length, size_last)
            k += 1

        i += consumed
        count += 1

    return count

# ---------------------------
# Main wrapper: prepare arrays, check memory, call njit routines
# ---------------------------
def compute_lz_complexity_bytes(raw: bytes, memory_limit_bytes=1_000_000_000):
    """
    raw: input bytes
    memory_limit_bytes: threshold to avoid allocating huge (next_arr) tables
    Returns: (complexity_count, elapsed_seconds)
    """
    n = len(raw)
    if n == 0:
        return 0, 0.0

    map256, m = build_byte_mapping(raw)
    # map raw bytes to dense alphabet
    arr = np.frombuffer(raw, dtype=np.uint8)
    arr_mapped = map256[arr]  # int32 vector

    # estimate memory for next_arr: 2*n * m * 4 bytes (int32)
    max_states = 2 * n
    est_bytes = max_states * m * 4
    if est_bytes > memory_limit_bytes:
        # Memory would be too large; informative error to user.
        raise MemoryError(
            f"Estimated memory for transition table is {est_bytes/1e9:.3f} GB "
            f"(2*N*m*4). Reduce input size or increase memory_limit_bytes "
            f"(current {memory_limit_bytes/1e9:.3f} GB)."
        )

    # allocate arrays
    next_arr = np.full((max_states, m), -1, dtype=np.int32)
    link = np.full(max_states, -1, dtype=np.int32)
    length = np.zeros(max_states, dtype=np.int32)
    # size_last: [size, last]
    size_last = np.zeros(2, dtype=np.int32)
    size_last[0] = 1  # one initial state (0)
    size_last[1] = 0
    link[0] = -1
    length[0] = 0

    # time and compute
    t0 = time.perf_counter()
    count = lz_factor_count(arr_mapped, next_arr, link, length, size_last)
    t1 = time.perf_counter()
    return int(count), float(t1 - t0)

# ---------------------------
# CLI / demo
# ---------------------------
def read_input_bytes():
    print("Menù:\n")
    print("[F] Read from file")
    print("[K] Read from keyboard (string)\n")
    x = input("> ").strip().upper()
    if x == "F":
        fn = input("File name: ").strip()
        try:
            with open(fn, "rb") as f:
                return f.read()
        except FileNotFoundError:
            print(f"File {fn} not found.")
            sys.exit(1)
    else:
        s = input("give me a string: ")
        return s.encode("utf-8")


def main():
    raw = read_input_bytes()
    print(f"Input length = {len(raw)} bytes")

    try:
        c, elapsed = compute_lz_complexity_bytes(raw, memory_limit_bytes=1_000_000_000)
    except MemoryError as e:
        print("MemoryError:", e)
        print("You can:")
        print("- give a smaller input, or")
        print("- increase memory_limit_bytes in compute_lz_complexity_bytes(), or")
        print("- ask me to run the O(N log N) fallback version.")
        sys.exit(1)

    print(f"\nLempel-Ziv complexity (number of factors) = {c}")
    print(f"Elapsed time = {elapsed:.6f} seconds\n")


if __name__ == "__main__":
    main()
