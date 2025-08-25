import numpy as np
def binary_matrix(N):
    """
    Enumerate N-bit binaries as a (2**N, N) matrix.

    Parameter:
    - N: number of (qu)bit.

    Return:

    np.ndarry of the shape (2**N, N) and dtype int.

    Example:

    >> print(binary_matrix(3))
    [[0 0]
     [0 1]
     [1 0]
     [1 1]]

    """
    return np.array([list(np.binary_repr(i, width=N)) for i in range(2**N)], dtype=int)


def dicts_to_matrix(dict_list, default_value=0.0, normalize=False):
    """
    Convert a list of dictionaries with 6-bit binary string keys into an M x N matrix.

    Parameter:
    - dict_list: list of dicts with keys as 6-bit binary strings and values as numbers.
    - default_value: value to fill in if a key is missing in a dictionary (default: 0.0).
    - normalize: if True, normalize values in each dictionary so they sum to 1.

    Return:
    - matrix: np.ndarray of shape (M, N) with values filled from the dictionaries.
    - sorted_keys: list of binary keys sorted in binary numeric order.
    """
    # Step 1: Collect all unique keys
    all_keys = set()
    for d in dict_list:
        all_keys.update(d.keys())

    # Step 2: Sort keys by binary numeric value
    sorted_keys = sorted(all_keys, key=lambda x: int(x, 2))
    key_to_index = {key: i for i, key in enumerate(sorted_keys)}

    M = len(sorted_keys)
    N = len(dict_list)
    matrix = np.full((M, N), default_value, dtype=float)

    # Step 3: Fill matrix, optionally normalize each dictionary
    for j, d in enumerate(dict_list):
        if normalize:
            total = sum(d.values())
            if total > 0:
                d = {k: v / total for k, v in d.items()}
        for key, value in d.items():
            i = key_to_index[key]
            matrix[i, j] = value

    # Step 4: Convert keys to binary arrays
    binary_key_arrays = [
        np.array([int(bit) for bit in key], dtype=int) for key in sorted_keys
    ]

    return matrix, sorted_keys, binary_key_arrays


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
        
def hamming_distance_1_neighbors(binary_str: str) -> list[str]:
    """Generate all binary strings with Hamming distance 1 from the input binary string."""
    neighbors = []
    for i in range(len(binary_str)):
        # Flip the i-th bit
        flipped_bit = '1' if binary_str[i] == '0' else '0'
        neighbor = binary_str[:i] + flipped_bit + binary_str[i+1:]
        neighbors.append(neighbor)
    return neighbors

def hamming1_scatter(prob_baseline: np.ndarray, prob_best: np.ndarray, num_qubits: int) -> np.ndarray:
    """
    For each basis index b (0..2^n-1), find the index in {b} ∪ {b xor 2^k | k=0..n-1}
    with the largest baseline prob and scatter-add prob_best[b] into that index.
    """
    N = 1 << num_qubits                      # 2**num_qubits
    idxs = np.arange(N, dtype=np.uint64)     # basis indices as integers

    # Build neighbor matrix: [N, num_qubits] via XOR with each single-bit mask
    flips = (1 << np.arange(num_qubits, dtype=np.uint64))           # [Q]
    neighbors = idxs[:, None] ^ flips[None, :]                      # [N, Q]

    # Include self as a neighbor at column 0 → concat to shape [N, Q+1]
    neighbors = np.concatenate([idxs[:, None], neighbors], axis=1)  # [N, Q+1]

    # Gather baseline probabilities for each candidate
    nb_base = prob_baseline[neighbors]                              # [N, Q+1]

    # Pick argmax candidate per row
    best_cols = np.argmax(nb_base, axis=1)                          # [N]
    best_targets = neighbors[np.arange(N), best_cols]               # [N]

    # Scatter-add prob_best[b] into prob_hamming1[best_targets[b]]
    out = np.zeros(N, dtype=prob_best.dtype)
    np.add.at(out, best_targets, prob_best)
    return out
