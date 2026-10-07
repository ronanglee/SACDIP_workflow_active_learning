import os
import numpy as np
from collections import Counter, defaultdict
from ase.io import read, write
from ase import Atoms
from ase.visualize import view
from ase.data import covalent_radii, atomic_numbers

traj_path = "/home/energy/s234441/mlip_bsc8/ML_workflow_BSc/Example_data/A.traj"

target_metals = ["Fe", "Co", "Ni", "Pd", "Pt"]

# -----------------------------
# Duplicate helpers
# -----------------------------

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


def remove_duplicates(structures, tol=0.08):
    unique = []

    for atoms in structures:
        duplicate = False
        for ref in unique:
            if same_structure(atoms, ref, tol=tol):
                duplicate = True
                break

        if not duplicate:
            unique.append(atoms)

    return unique


# -----------------------------
# Find metal atom
# -----------------------------

def find_metal_index(atoms):
    for atom in atoms:
        if atom.symbol not in ["C", "N", "O", "H"]:
            return atom.index
    return None


# -----------------------------
# Remove adsorbates (O, H)
# -----------------------------

def remove_adsorbates(atoms):
    atoms = atoms.copy()
    remove = [atom.index for atom in atoms if atom.symbol in ["O", "H"]]

    if remove:
        del atoms[remove]

    return atoms


# -----------------------------
# N-N prevention
# -----------------------------

N_RADIUS = covalent_radii[atomic_numbers["N"]]
NN_SCALE = 1.15
NN_CUTOFF = NN_SCALE * (2 * N_RADIUS)


def would_create_nn_bond(atoms, candidate_idx, chosen_new_n=None, cutoff=NN_CUTOFF):
    if chosen_new_n is None:
        chosen_new_n = []

    existing_n = [atom.index for atom in atoms if atom.symbol == "N"]

    for n_idx in existing_n + list(chosen_new_n):
        if n_idx == candidate_idx:
            continue

        d = atoms.get_distance(candidate_idx, n_idx, mic=True)
        if d < cutoff:
            return True

    return False


def has_any_nn_bond(atoms, cutoff=NN_CUTOFF):
    n_indices = [atom.index for atom in atoms if atom.symbol == "N"]

    for i in range(len(n_indices)):
        for j in range(i + 1, len(n_indices)):
            d = atoms.get_distance(n_indices[i], n_indices[j], mic=True)
            if d < cutoff:
                return True

    return False


# -----------------------------
# Metal substitution
# -----------------------------

def substitute_metal(atoms, metal_index):
    new_structures = []

    for metal in target_metals:
        new_atoms = atoms.copy()
        new_atoms[metal_index].symbol = metal
        new_structures.append(new_atoms)

    return new_structures


# -----------------------------
# Local C/N shell around the metal
# -----------------------------

def get_cn_indices_within_radius(atoms, metal_index, radius=2.2):
    candidates = []

    for atom in atoms:
        if atom.index == metal_index:
            continue

        if atom.symbol in ["C", "N"]:
            dist = atoms.get_distance(atom.index, metal_index, mic=True)
            if dist <= radius:
                candidates.append((dist, atom.index))

    candidates.sort(key=lambda x: x[0])

    return [idx for _, idx in candidates]


def generate_nmax_to_n0_candidates(atoms, shell_indices):
    candidates = []
    n_shell = len(shell_indices)

    # shell_indices are sorted from closest -> farthest
    for n_inner in range(n_shell, -1, -1):
        candidate = atoms.copy()

        for i, idx in enumerate(shell_indices):
            candidate[idx].symbol = "N" if i < n_inner else "C"

        if not has_any_nn_bond(candidate):
            candidates.append(candidate)

    return candidates


# -----------------------------
# Add one random N near the metal
# -----------------------------

def add_one_random_n_within_radius(atoms, metal_index, max_dist=4.5, rng=None):
    atoms = atoms.copy()

    if rng is None:
        rng = np.random.default_rng()

    valid_candidates = []

    for atom in atoms:
        if atom.symbol != "C":
            continue

        dist = atoms.get_distance(atom.index, metal_index, mic=True)
        if dist > max_dist:
            continue

        if would_create_nn_bond(atoms, atom.index):
            continue

        valid_candidates.append(atom.index)

    if not valid_candidates:
        return None

    chosen_idx = int(rng.choice(valid_candidates))
    atoms[chosen_idx].symbol = "N"

    if has_any_nn_bond(atoms):
        return None

    return atoms


# -----------------------------
# Add adsorbates
# -----------------------------

def add_OOH(atoms, metal_index):
    atoms = atoms.copy()
    metal_pos = atoms[metal_index].position

    O1 = metal_pos + np.array([0, 0, 1.8])
    O2 = O1 + np.array([0, 0, 1.4])
    H = O2 + np.array([0.8, 0, 0.6])

    OOH = Atoms(
        ["O", "O", "H"],
        positions=[O1, O2, H]
    )

    atoms += OOH
    return atoms


def add_O(atoms, metal_index):
    atoms = atoms.copy()
    metal_pos = atoms[metal_index].position

    O = metal_pos + np.array([0, 0, 1.8])

    O_ads = Atoms(
        ["O"],
        positions=[O]
    )

    atoms += O_ads
    return atoms


def add_OH(atoms, metal_index):
    atoms = atoms.copy()
    metal_pos = atoms[metal_index].position

    O = metal_pos + np.array([0, 0, 1.8])
    H = O + np.array([0.8, 0, 0.6])

    OH = Atoms(
        ["O", "H"],
        positions=[O, H]
    )

    atoms += OH
    return atoms


# -----------------------------
# Load training structures
# -----------------------------

training_structures = read(traj_path, ":")

clean_structures = []
rng = np.random.default_rng()

# -----------------------------
# Generate clean candidates
# -----------------------------

for atoms in training_structures:
    atoms = remove_adsorbates(atoms)

    metal_index = find_metal_index(atoms)
    if metal_index is None:
        continue

    metal_variants = substitute_metal(atoms, metal_index)

    for struct in metal_variants:
        struct_metal_index = find_metal_index(struct)
        if struct_metal_index is None:
            continue

        shell_indices = get_cn_indices_within_radius(
            struct,
            struct_metal_index,
            radius=2.2
        )

        base_candidates = generate_nmax_to_n0_candidates(struct, shell_indices)

        # Base candidates: Nmax, N(max-1), ..., N1, N0
        clean_structures.extend(base_candidates)

        # Extra candidates: add one random N to each base structure
        for base in base_candidates:
            extra = add_one_random_n_within_radius(
                base,
                struct_metal_index,
                max_dist=4.5,
                rng=rng
            )
            if extra is not None:
                clean_structures.append(extra)

# -----------------------------
# Remove duplicates from clean structures only
# -----------------------------

clean_structures = remove_duplicates(clean_structures, tol=0.3)

print("Unique clean structures:", len(clean_structures))

# -----------------------------
# Add adsorbates after deduplication
# -----------------------------

o_structures = []
oh_structures = []
ooh_structures = []

for atoms in clean_structures:
    metal_index = find_metal_index(atoms)
    if metal_index is None:
        continue

    o_structures.append(add_O(atoms, metal_index))
    oh_structures.append(add_OH(atoms, metal_index))
    ooh_structures.append(add_OOH(atoms, metal_index))

print("Unique O structures:", len(o_structures))
print("Unique OH structures:", len(oh_structures))
print("Unique OOH structures:", len(ooh_structures))

# -----------------------------
# Save files
# -----------------------------

output_dir = "kandidaterA"
os.makedirs(output_dir, exist_ok=True)

write(f"{output_dir}/clean_SAC_candidates.xyz", clean_structures)
write(f"{output_dir}/O_SAC_candidates.xyz", o_structures)
write(f"{output_dir}/OH_SAC_candidates.xyz", oh_structures)
write(f"{output_dir}/OOH_SAC_candidates.xyz", ooh_structures)

# -----------------------------
# View structures
# -----------------------------

view(clean_structures)
view(ooh_structures)