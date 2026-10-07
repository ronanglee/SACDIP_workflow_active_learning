from ase.io import read, write
from ase import units

import pandas as pd
import numpy as np
import torch
import os, json
import sys
import argparse
import logging
from cPaiNN.relax import ML_Relaxer
from cPaiNN.utils import setup_seed


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Run NEB simulation", fromfile_prefix_chars="+"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Where to find the models",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model. Default is cpainn",
        default='cpainn',
    )

    parser.add_argument(
        "--structure_path",
        type=str,
        help="Path to all to be optimized structures",
    )

    parser.add_argument(
        "--opt_steps",
        type=int,
        default=500,
        help="Number of optimization steps",
    )

    parser.add_argument(
        "--opt_fmax",
        type=int,
        help="The maximum force for the optimization",
    )

    parser.add_argument(
        "--opt_algo",
        type=str,
        help="Algorithm used for optimization",
    )

    parser.add_argument(
        "--opt_cell",
        type=bool,
        default=True,
        help="Optimize the cell or not",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for this run",
    )

    parser.add_argument(
        "--max_force_sd",
        type=float,
        default=0.5,
        help="Stop optimization run if force standard deviation is too large",
    )

    parser.add_argument(
        "--force_sd_threshold",
        type=float,
        default=0.2,
        help="Threshold to write structure to warning_opt.xyz instead of pool_set",
    )

    parser.add_argument(
        "--max_force_threshold",
        type=float,
        default=20.0,
        help="Discard structures with predicted max force above this value (eV/A)",
    )

    return parser.parse_args(arg_list)


def update_namespace(ns: argparse.Namespace, d: dict) -> None:
    for k, v in d.items():
        ns.__dict__[k] = v


def optimize_run(params: dict, run_dir: str = '.') -> None:

    # Create device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Put a tensor on the device before loading data
    # This way the GPU appears to be in use when other users run gpustat
    torch.tensor([0], device=device)

    # Get arguments
    args = get_arguments()

    # Update namespace with the parameters from the config file
    update_namespace(args, params)

    # Save parsed arguments
    with open(os.path.join(run_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    setup_seed(args.random_seed)

    # set logger
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    runHandler = logging.FileHandler(os.path.join(run_dir, 'optimize.log'), mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))

    logger.addHandler(runHandler)
    logger.addHandler(logging.StreamHandler())

    logging.info(f"Using device: {device}")

    # Load MLP model
    # Set up the MLP calculator
    relaxer = ML_Relaxer(
        calc_name=args.model_name,
        optimizer=args.opt_algo,
        calc_paths=args.model_path,
        device=device,
        relax_cell=args.opt_cell
    )

    # Load all structures
    structures = read(args.structure_path, index=':')
    logger.info(f"Found {len(structures)} structures in {args.structure_path}")

    pool_set = []

    max_uncertainty_in_batch = 0.0

    # Loop over all structures
    for i, atoms in enumerate(structures):

        traj_path = os.path.join(run_dir, f"opt_{i}.xyz")
        log_path = os.path.join(run_dir, f"opt_{i}.log")

        logger.info(f"Optimizing structure {i} with {args.opt_algo} algorithm")

        relax_result = relaxer.relax(
            atoms,
            fmax=args.opt_fmax,
            steps=args.opt_steps,
            traj_file=traj_path,
            log_file=log_path,
        )

        final_atoms = relax_result['final_structure']

        # Force screening

        forces = final_atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))

        logger.info(f"Structure {i} max predicted force: {max_force:.2f} eV/A")
        
        # Chech maxforce of final structure (convergence)
        if max_force > args.max_force_threshold:

            logger.error(
                f"Structure {i} discarded: max force {max_force:.2f} > {args.max_force_threshold}"
            )

            force_error_path = os.path.join(run_dir, "force_error_opt.xyz")

            write(force_error_path, final_atoms, append=True)

            # if max_force too large discard entire trajectory
            if os.path.exists(traj_path):
                os.remove(traj_path)

            continue

        ensemble = final_atoms.calc.results.get('ensemble')

        if ensemble is not None:

            # Check force uncertainty
            forces_sd = np.mean(np.sqrt(ensemble['forces_var']))

            logger.info(f"Structure {i} optimized. ML Force SD: {forces_sd:.4f}")

            # save largest uncertainty
            if forces_sd > max_uncertainty_in_batch:
                max_uncertainty_in_batch = forces_sd

            if forces_sd > args.max_force_sd:

                logger.error(
                    f"Structure {i} has fatal uncertainty: {forces_sd:.4f} > {args.max_force_sd}."
                )

                error_path = os.path.join(run_dir, 'fatal_error_opt.xyz')

                if 'bader_charge' in final_atoms.arrays and 'bader_charge' in final_atoms.calc.results:
                    final_atoms.arrays['bader_charge'] = final_atoms.calc.results['bader_charge']

                write(error_path, final_atoms, append=True) # fatal error
                
                # if forces_sd > max_force_sd discard entire trajectory
                if os.path.exists(traj_path):
                    os.remove(traj_path)

                continue

            elif forces_sd > args.force_sd_threshold:

                warning_path = os.path.join(run_dir, 'warning_opt.xyz')

                if 'bader_charge' in final_atoms.arrays and 'bader_charge' in final_atoms.calc.results:
                    final_atoms.arrays['bader_charge'] = final_atoms.calc.results['bader_charge']

                write(warning_path, final_atoms, append=True) # non fatal warning

                logger.warning(
                    f"Structure {i} has high uncertainty ({forces_sd:.4f})"
                )
                # if forces_sd > force_sd_threshold discard entire trajectory
                if os.path.exists(traj_path):
                    os.remove(traj_path)

                continue

        else:
            logger.warning(
                f"No ensemble results found for structure {i}. Cannot check uncertainty."
            )

        pool_set.append(traj_path)

        logger.info(f"Optimized structure {i} saved to {traj_path}")
        logger.info(f"Final energy: {final_atoms.get_potential_energy()}")

    accuracy_threshold = 0.02

    logger.info(
        f"Highest unc for the batch was: {max_uncertainty_in_batch:.4f} eV/A"
    )
    
    
    # if all structures reach 0.02 eV/A terminate training
    if max_uncertainty_in_batch > 0.0 and max_uncertainty_in_batch < accuracy_threshold:

        success_path = os.path.join(run_dir, "SUCCESS_EARLY_STOP.txt")

        with open(success_path, "w") as f:
            f.write(f"Workflow stoped deliberately\n")
            f.write(
                f"Max unc in this batch was {max_uncertainty_in_batch:.4f} eV/A, which is under the limit of {accuracy_threshold} eV/A.\n"
            )
            f.write(f"The model is in: {args.model_path}\n")

        logger.info("The model is now accurate enough.")

        raise RuntimeError(
            f"EARLY STOP: The model is precise enough (Max usikkerhed: {max_uncertainty_in_batch:.4f} < {accuracy_threshold})."
        )
    # Save the pool set
    pool_set_path = os.path.join(run_dir, "pool_set.json")

    with open(pool_set_path, "w") as f:
        json.dump(pool_set, f)

    logger.info(f"Pool set saved to {pool_set_path}")