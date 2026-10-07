from collections import Counter, defaultdict
from pathlib import Path
import random

import numpy as np
from ase.io import read, write

# =========================
# Parameters
# =========================
input_path = Path("/home/energy/s234441/mlip_bsc9/ML_workflow_BSc/Example_data/sac_training_filtered.xyz")
val_fraction = 0.2
tol = 0.3
seed = 42

# Output files are written to the same directory as the input file
train_path = input_path.with_name(input_path.stem + "_train.xyz")
val_path = input_path.with_name(input_path.stem + "_val.xyz")


# =========================
# Helper functions
# =========================
def get_composition(atoms):
    return Counter(atoms.get_chemical_symbols())


def get_pairwise_distances_by_type(atoms):
    symbols = atoms.get_chemical_symbols()
    dmat = atoms.get_all_distances(mic=True)

    pair_dists = defaultdict(list)

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            pair = tuple(sorted((symbols[i], symbols[j])))
            pair_dists[pair].append(dmat[i, j])

    for pair in pair_dists:
        pair_dists[pair] = np.sort(np.array(pair_dists[pair]))

    return dict(pair_dists)


def same_structure(a, b, tol=0.08):
    if get_composition(a) != get_composition(b):
        return False

    da = get_pairwise_distances_by_type(a)
    db = get_pairwise_distances_by_type(b)

    if da.keys() != db.keys():
        return False

    for pair in da:
        if len(da[pair]) != len(db[pair]):
            return False

        if not np.allclose(da[pair], db[pair], atol=tol, rtol=0):
            return False

    return True


def build_similarity_groups(structures, tol=0.08):
    """
    Build groups of similar structures.
    If A is similar to B and B is similar to C,
    all three are placed in the same group.
    """
    n = len(structures)
    visited = [False] * n
    groups = []

    for i in range(n):
        if visited[i]:
            continue

        group = []
        stack = [i]
        visited[i] = True

        while stack:
            current = stack.pop()
            group.append(current)

            for j in range(n):
                if not visited[j] and same_structure(structures[current], structures[j], tol=tol):
                    visited[j] = True
                    stack.append(j)

        groups.append(group)

    return groups


def grouped_train_val_split(structures, val_fraction=0.2, tol=0.08, seed=42):
    rng = random.Random(seed)
    groups = build_similarity_groups(structures, tol=tol)
    rng.shuffle(groups)

    n_total = len(structures)
    target_val = n_total * val_fraction

    val_indices = []
    current_val_count = 0

    for group in groups:
        if current_val_count < target_val:
            val_indices.extend(group)
            current_val_count += len(group)

    val_index_set = set(val_indices)
    train_indices = [i for i in range(n_total) if i not in val_index_set]

    train_structures = [structures[i] for i in train_indices]
    val_structures = [structures[i] for i in val_indices]

    return train_structures, val_structures, train_indices, val_indices, groups


# =========================
# Run the split
# =========================
def main():
    structures = read(str(input_path), index=":")
    print(f"Total number of structures: {len(structures)}")

    train_structures, val_structures, train_indices, val_indices, groups = grouped_train_val_split(
        structures,
        val_fraction=val_fraction,
        tol=tol,
        seed=seed,
    )

    print(f"Number of groups found: {len(groups)}")
    print(f"Train: {len(train_structures)} structures")
    print(f"Validation: {len(val_structures)} structures")
    print(f"Actual validation fraction: {len(val_structures)/len(structures):.4f}")

    write(str(train_path), train_structures)
    write(str(val_path), val_structures)

    print(f"Train file written to: {train_path}")
    print(f"Validation file written to: {val_path}")


if __name__ == "__main__":
    main()
