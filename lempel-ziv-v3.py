# Fast Lempel-Ziv Complexity (LZ76) using Suffix Array + LCP
# Complexity: O(N log N)  --> suitable for millions of characters

import sys, time
import numpy as np


# -----------------------------------------------------------
#   SUFFIX ARRAY (Skew algorithm / prefix doubling)
# -----------------------------------------------------------
def build_suffix_array(s):
    """
    Builds a suffix array in O(N log N)
    """
    n = len(s)
    k = 1
    rank = np.array(s, dtype=np.int64)
    tmp = np.zeros(n, dtype=np.int64)
    sa = np.arange(n, dtype=np.int64)

    while True:
        # Sort by (rank[i], rank[i+k])
        sa = sorted(sa, key=lambda x: (rank[x], rank[x + k] if x + k < n else -1))

        tmp[sa[0]] = 0
        for i in range(1, n):
            prev = sa[i - 1]
            cur = sa[i]
            tmp[cur] = tmp[prev] + (
                (rank[prev], rank[prev + k] if prev + k < n else -1)
                < (rank[cur], rank[cur + k] if cur + k < n else -1)
            )

        rank[:] = tmp[:]
        k *= 2
        if k >= n:
            break

    return np.array(sa, dtype=np.int64)


# -----------------------------------------------------------
#   LCP ARRAY (Kasai algorithm)
# -----------------------------------------------------------
def build_lcp_array(s, sa):
    """
    Kasai’s LCP construction in O(N)
    """
    n = len(s)
    rank = np.zeros(n, dtype=np.int64)
    for i in range(n):
        rank[sa[i]] = i

    lcp = np.zeros(n, dtype=np.int64)
    h = 0

    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1

    return lcp


# -----------------------------------------------------------
#   LZ76 COMPLEXITY USING SUFFIX ARRAY + LCP
# -----------------------------------------------------------
def lz76_complexity(s):
    """
    Compute Lempel-Ziv complexity using the LZ76 factorization.
    Complexity: O(N log N)
    """
    if len(s) == 0:
        return 0

    sa = build_suffix_array(s)
    lcp = build_lcp_array(s, sa)

    n = len(s)
    c = 1
    i = 0
    while i < n:
        # The longest match of substring s[i:]
        longest_match = 0
        for j in range(1, n - i):
            longest_match = max(longest_match, lcp[j])
        if longest_match == 0:
            i += 1
        else:
            i += longest_match
        c += 1

    return c


# -----------------------------------------------------------
#   INPUT HANDLING
# -----------------------------------------------------------
def read_string():
    print("Menù:\n")
    print("Read from [F]ile\n")
    print("Read from [K]eyboard\n")
    x = input("> ").strip()

    if x.upper() == "F":
        file_name = input("What file: ").strip()
        try:
            with open(file_name, "rb") as f:
                data = f.read()
                return data
        except FileNotFoundError:
            print(f"File {file_name} not found.")
            sys.exit(1)

    else:
        return input("give me a string: ").encode("utf-8")


def main():
    raw = read_string()

    # Convert to numpy array of bytes
    arr = np.frombuffer(raw, dtype=np.uint8)

    print("\nComputing LZ complexity (fast O(N log N) version)...\n")

    # -----------------------------------------
    #   MISURAZIONE TEMPO DI ELABORAZIONE
    # -----------------------------------------
    t0 = time.perf_counter()
    c = lz76_complexity(arr)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    # -----------------------------------------

    print(f"Lempel-Ziv complexity index = {c}")
    print(f"\nElapsed time: {elapsed:.6f} seconds\n")

main()
