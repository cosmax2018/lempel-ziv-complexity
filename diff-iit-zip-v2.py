#
# diff-iit-zip-v2.py :  esempio Python pratico per vedere la differenza tra ψ (integrazione causale)
#                       e Lempel-Ziv (compressibilità del segnale).
#
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# --------------------------
# Funzione Lempel-Ziv
# --------------------------
def complexityLempelZiv(s):
    complexity = 1
    prefix_length = 1
    length_component = 1
    max_length_component = 1
    pointer = 0
    while prefix_length + length_component <= len(s):
        if s[pointer + length_component - 1] == s[prefix_length + length_component - 1]:
            length_component += 1
        else:
            max_length_component = max(length_component, max_length_component)
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
def evolve(state):
    A, B, C = state
    return (B ^ C, A, A & B)

steps = 16
sequence = []
current_state = (0, 0, 0)
for _ in range(steps):
    sequence.append(current_state)
    current_state = evolve(current_state)

# --------------------------
# Converti in stringa binaria
# --------------------------
seq_str = ''.join(''.join(str(bit) for bit in s) for s in sequence)

# --------------------------
# Calcolo Lempel-Ziv e ψ passo passo
# --------------------------
lz_values = []
psi_values = []

for i in range(1, len(sequence)+1):
    subseq = sequence[:i]
    subseq_str = seq_str[:i*3]  # ogni stato = 3 bit

    # Lempel-Ziv
    lz_index = complexityLempelZiv(list(subseq_str))
    lz_values.append(lz_index)

    # ψ semplificato
    states_array = np.array(subseq)
    probs = np.mean(states_array, axis=0)
    H_parts = np.sum([-p*np.log2(p)-(1-p)*np.log2(1-p) if 0<p<1 else 0 for p in probs])
    flat_states = [tuple(s) for s in subseq]
    unique, counts = np.unique(flat_states, axis=0, return_counts=True)
    probs_total = counts / counts.sum()
    H_total = -np.sum(probs_total * np.log2(probs_total))
    psi_values.append(H_total - H_parts)

# --------------------------
# Grafico
# --------------------------
plt.figure(figsize=(10,5))
plt.plot(range(1, len(sequence)+1), lz_values, marker='o', label="Lempel-Ziv complexity")
plt.plot(range(1, len(sequence)+1), psi_values, marker='s', label="ψ semplificato")
plt.xlabel("Step (numero di stati)")
plt.ylabel("Valore")
plt.title("Confronto Lempel-Ziv e ψ semplificato")
plt.grid(True)
plt.legend()
plt.show()
