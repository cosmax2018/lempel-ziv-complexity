#
# diff-iit-zip-v1.py :  esempio Python pratico per vedere la differenza tra ψ (integrazione causale)
#                       e Lempel-Ziv (compressibilità del segnale).
#

import numpy as np
from itertools import product
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
    
# --------------------------
# Sistema binario 3 unità
# --------------------------
# Stato iniziale: 3 unità, generiamo tutti gli stati possibili
states = list(product([0, 1], repeat=3))

# Funzione di evoluzione: esempio semplice
# A(t+1) = B(t) XOR C(t)
# B(t+1) = A(t)
# C(t+1) = A(t) AND B(t)
def evolve_v1(state):
    A, B, C = state
    A_new = B ^ C
    B_new = A
    C_new = A & B
    return (A_new, B_new, C_new)
    
def evolve_v2(state):
    A, B, C = state
    A_new = (B + C) % 2      # somma modulo 2
    B_new = (A + C) % 2
    C_new = (A ^ B)           # XOR
    return (A_new, B_new, C_new)
    

# --------------------------
# Genera sequenza di stati
# --------------------------
steps = 8
sequence = []
current_state = (1, 0, 0)

for _ in range(steps):
    sequence.append(current_state)
    current_state = evolve_v2(current_state)

# Converti in stringa binaria per LZ
seq_str = ''.join(''.join(str(bit) for bit in s) for s in sequence)

print("Sequenza di stati:")
print(sequence)
print("\nSequenza binaria concatenata:")
print(seq_str)

# --------------------------
# Calcolo Lempel-Ziv
# --------------------------
lz_index = complexityLempelZiv(np.array(tuple(seq_str)))
print("\nLempel-Ziv complexity:", lz_index)

# --------------------------
# Calcolo ψ semplificato
# --------------------------
# Entropia del sistema completo
def H(prob):
    # entropia in bit
    prob_nonzero = prob[prob > 0]
    return -np.sum(prob_nonzero * np.log2(prob_nonzero))

# Distribuzione dei singoli elementi
states_array = np.array(sequence)
probs = np.mean(states_array, axis=0)  # probabilità media di 1 per ogni unità

# Entropia individuale (somma entropie delle parti)
H_parts = np.sum([-p*np.log2(p) - (1-p)*np.log2(1-p) if p>0 and p<1 else 0 for p in probs])

# Entropia totale (approssimata come somma dei bit nel sistema)
flat_states = [tuple(s) for s in sequence]
unique, counts = np.unique(flat_states, axis=0, return_counts=True)
probs_total = counts / counts.sum()
H_total = H(probs_total)

# ψ semplificato = H_total - H_parts
psi_simplified = H_total - H_parts

print("\nψ semplificato:", psi_simplified)
