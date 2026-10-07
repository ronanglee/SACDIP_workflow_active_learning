import random
from pathlib import Path
from ase.io import read, write

# =========================
# Settings
# =========================
BASE_DIR = Path("/home/energy/s234441/mlip_bsc9/ML_workflow_BSc/Example_data")
CANDIDATE_GROUPS = ["Z", "C10", "C8", "C12", "A"]
ADSORBATES = ["clean", "O", "OH", "OOH"]

N_PER_ADSORBATE = 100   # 100 structures from each adsorbate
BATCH_SIZE = 100        # 400 structures per training batch
RANDOM_SEED = 67        # For reproducible random sampling

random.seed(RANDOM_SEED)


def find_xyz_file(group: str, adsorbate: str) -> Path:
    """
    Find the correct xyz file for a given group and adsorbate.
    Expected format, for example:
      /.../Example_data/kandidaterC10/clean_SAC_candidates.xyz
    """
    folder = BASE_DIR / f"kandidater{group}"
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    expected = folder / f"{adsorbate}_SAC_candidates.xyz"
    if expected.exists():
        return expected

    # Fallback: try to find any xyz file starting with the adsorbate name
    matches = list(folder.glob(f"{adsorbate}*.xyz"))
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise RuntimeError(
            f"Multiple possible files found for {group}/{adsorbate}: "
            + ", ".join(str(m) for m in matches)
        )
    else:
        raise FileNotFoundError(
            f"Could not find xyz file for {group}/{adsorbate} in {folder}"
        )


def sample_structures_from_file(xyz_file: Path, n: int):
    """
    Read all structures from an xyz file and return n random structures.
    """
    structures = read(str(xyz_file), index=":")
    if len(structures) < n:
        raise ValueError(
            f"The file {xyz_file} contains only {len(structures)} structures, "
            f"but {n} were requested."
        )
    return random.sample(structures, n)


def main():
    all_selected_structures = []

    # Collect 100 structures from each adsorbate in each group
    for group in CANDIDATE_GROUPS:
        print(f"\nProcessing group: {group}")
        for adsorbate in ADSORBATES:
            xyz_file = find_xyz_file(group, adsorbate)
            selected = sample_structures_from_file(xyz_file, N_PER_ADSORBATE)
            all_selected_structures.extend(selected)
            print(
                f"  {adsorbate:>5}: selected {len(selected)} structures from {xyz_file.name}"
            )

    total = len(all_selected_structures)
    print(f"\nTotal number of selected structures: {total}")

    # Shuffle all 2000 structures before splitting into batches
    random.shuffle(all_selected_structures)

    # Write structures to training batch files
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        batch_structures = all_selected_structures[start:end]

        output_file = BASE_DIR / f"training_batch{i+1}.xyz"
        write(str(output_file), batch_structures)
        print(f"Wrote {len(batch_structures)} structures to {output_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()