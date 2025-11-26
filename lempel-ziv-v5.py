import time
import os
from numba import njit

# ================================================================
#   VERSIONE 1 — VELOCE (O(N)) — LZ76 ottimizzata (Numba JIT)
# ================================================================
@njit
def lz_complexity_fast_numba(s):
    n = len(s)
    i, k, l, c = 0, 1, 1, 1

    while True:
        if i + k - 1 < n and l + k - 1 < n and s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > 1:
                i += 1
                if i == l:
                    c += 1
                    l += k
                    if l >= n:
                        break
                    i = 0
                k = 1
            else:
                i += 1
                if i == l:
                    c += 1
                    l += 1
                    if l >= n:
                        break
                    i = 0

    return c


# ================================================================
#   VERSIONE 2 — FALLBACK (O(N log N)) — Python puro
# ================================================================
def lz_complexity_fallback(s: str) -> int:
    n = len(s)
    substrings = set()
    c = 0
    i = 0

    while i < n:
        max_match = 1
        # scanning until mismatch
        for l in range(1, n - i + 1):
            sub = s[i:i + l]
            if sub in substrings:
                max_match = l
            else:
                break

        substrings.add(s[i:i + max_match])
        c += 1
        i += max_match

    return c


# ================================================================
#   LOGICA DI FALLBACK AUTOMATICO
# ================================================================
def compute_lz_complexity(s: str):
    MAX_TIME_FAST = 1.0   # limite per trigger fallback

    # prima chiamata numba = compilazione → più lenta
    start = time.time()

    try:
        result = lz_complexity_fast_numba(s)
        elapsed = time.time() - start

        # la prima esecuzione include il tempo di compilazione!
        if elapsed > MAX_TIME_FAST:
            print(f"[!] Versione veloce troppo lenta ({elapsed:.2f}s). Compilazione Numba? Attivo fallback…")
            raise TimeoutError

        return result, elapsed, "FAST-NUMBA"

    except Exception:
        # fallback garantito
        start = time.time()
        result = lz_complexity_fallback(s)
        elapsed = time.time() - start
        return result, elapsed, "FALLBACK"


# ================================================================
#   LETTURA DA FILE E AVVIO
# ================================================================
if __name__ == "__main__":

    filename = input("Inserisci nome file: ")

    if not os.path.exists(filename):
        print("ERRORE: file non trovato.")
        exit(1)

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    print("Calcolo complessità LZ76 (Numba JIT)…")

    c, t, mode = compute_lz_complexity(data)

    print(f"\nRisultato:")
    print(f"  → Complessità Lempel-Ziv: {c}")
    print(f"  → Tempo impiegato: {t:.4f} s")
    print(f"  → Metodo utilizzato: {mode}")
