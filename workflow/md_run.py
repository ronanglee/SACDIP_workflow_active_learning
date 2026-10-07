from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, Trajectory, write

import numpy as np
import torch
import sys, os, json
import argparse
import logging
from cPaiNN.relax import ML_Relaxer
from cPaiNN.utils import setup_seed

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="MD simulations drive by graph neural networks", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--init_traj",
        type=str,
        help="Path to start configurations",
    )
    parser.add_argument(
        "--start_indice",
        type=int,
        help="Indice of the start configuration",
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
        "--time_step",
        type=float,
        #default=0.5,
        help="Time step of MD simulation",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum steps of MD",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=100000,
        help="Minimum steps of MD, raise error if not reached",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Maximum time steps of MD",
    )
    parser.add_argument(
        "--dump_step",
        type=int,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--print_step",
        type=int,
        default=1,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--num_uncertain",
        type=int,
        #default=1000,
        help="Stop MD when too many structures with large uncertainty are collected",
    )
    parser.add_argument(
        "--max_force_sd",
        type=float,
        default=0.5,
        help="Stop MD when the force standard deviation is too large",
    )
    parser.add_argument(
        "--force_sd_threshold",
        type=float,
        default=0.25,
        help="Stop MD when it has been over the force standard deviation threshold too many times",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.003,
        help="Setting the friction term in the Langevin dynamics",
    )
    parser.add_argument(
        "--rattle",
        type=float,
        help="Randomly displace atoms within the given atom length in Å",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for this run",
    )

    return parser.parse_args(arg_list)

def update_namespace(ns:argparse.Namespace, d:dict) -> None:
    """

    Update the namespace with the dictionary.

    Args:
        ns: The namespace to update
        d: The dictionary to update the namespace with
    
    """
    for k, v in d.items():
        
        ns.__dict__[k] = v

class CallsCounter:
    def __init__(self, func):
        self.calls = 0
        self.func = func
    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.func(*args, **kwargs)

def MD(params:dict,run_dir:str='.') -> None:
    # Create device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Put a tensor on the device before loading data
    # This way the GPU appears to be in use when other users run gpustat
    torch.tensor([0], device=device)
    
    # Get arguments
    args = get_arguments()
    # Update arguments with config file
    update_namespace(args, params)

    # Save parsed arguments
    with open(os.path.join(run_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    setup_seed(args.random_seed)

    # set logger
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    runHandler = logging.FileHandler(os.path.join(run_dir,'md.log'), mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    errorHandler = logging.FileHandler(os.path.join(run_dir,'error.log'), mode='w')
    errorHandler.setLevel(logging.WARNING)
    errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))

    logger.addHandler(runHandler)
    logger.addHandler(errorHandler)
    logger.addHandler(logging.StreamHandler())
    logger.warning = CallsCounter(logger.warning)
    logger.info = CallsCounter(logger.info)
    logging.info(f"Using device: {device}")

    # set up md start configuration
    images = read(args.init_traj, ':')
    start_indice = np.random.choice(len(images)) if args.start_indice == None else args.start_indice
    logger.debug(f'MD starts from No.{start_indice} configuration in {args.init_traj}')
    atoms = images[start_indice]
    atoms.wrap() #Wrap positions to unit cell.
    
    if args.rattle:
        atoms.rattle(args.rattle)
    
    # Set up the MLP calculator
    ML_class = ML_Relaxer(calc_name=args.model_name,calc_paths=args.model_path,device=device)
    ml_calc = ML_class.calculator
    if not ML_class.ensemble:
        raise NotImplementedError("Only ensemble training is supported at the moment")

    atoms.calc = ml_calc
    atoms.get_potential_energy()

    def write_warning_xyz(a=atoms,traj_name=os.path.join(run_dir,'warning_struct.xyz')):
        if 'bader_charge' in a.arrays:
            a.arrays['bader_charge'] = a.calc.results['bader_charge']
        write(traj_name, a, append=True)
    
    @CallsCounter
    def printenergy(a=atoms):  # store a reference to atoms in the definition.
        """Function to print the potential, kinetic and total energy."""
        # Calculate energy and temperature
        epot = a.get_potential_energy()
        ekin = a.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB) / a.get_global_number_of_atoms()

        # Calculate ensemble properties
        ensemble = a.calc.results['ensemble'].copy()
        ensemble['forces_var_mean'] = np.mean(ensemble['forces_var'])
        ensemble['forces_sd'] = np.mean(np.sqrt(ensemble['forces_var']))
        ensemble['forces_l2_var'] = np.mean(ensemble['forces_l2_var'])

        # Format ensemble for logging
        ensemble_formatted = ", ".join(
                    ["{}={:.5f}".format(k, np.mean(v)) for (k, v) in ensemble.items()]
                )

        # Log the results and check if the uncertainty is too large
        if ensemble['forces_sd'] > args.max_force_sd: 
            logger.error("Too large uncertainty!")
            if logger.info.calls + logger.warning.calls > args.min_steps:
                raise AssertionError("Done")
            else:
                raise ValueError(f"Too large uncertainty!, info calls: {logger.info.calls}, warning calls: {logger.warning.calls}, min calls: {args.min_steps}")
        elif ensemble['forces_sd'] > args.force_sd_threshold: 
            write_warning_xyz(a)
            logger.warning("Steps={:10d} Epot={:12.3f} Ekin={:12.3f} temperature={:8.2f} {} ".format(
                printenergy.calls * args.print_step,
                epot,
                ekin,
                temp,
                ensemble_formatted,
            ))
            if logger.warning.calls > args.num_uncertain:
                logger.error(f"More than {args.num_uncertain} uncertain structures are collected!")
                if logger.info.calls + logger.warning.calls > args.min_steps:
                    raise AssertionError("Done")
                else:
                    raise ValueError(f"More than {args.num_uncertain} uncertain structures are collected!")
        else:
            logger.info("Steps={:10d} Epot={:12.3f} Ekin={:12.3f} temperature={:8.2f} {} ".format(
                printenergy.calls * args.print_step,
                epot,
                ekin,
                temp,
                ensemble_formatted,
            ))

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    dyn = Langevin(atoms, args.time_step * units.fs, temperature_K=args.temperature, friction=args.friction)
    dyn.attach(printenergy, interval=args.print_step)

    #traj = Trajectory(os.path.join(run_dir,'MD.traj'), 'w', atoms)

    def write_xyz(a=atoms,traj_name=os.path.join(run_dir,'MD.xyz')):
        if 'bader_charge' in a.arrays:
            a.arrays['bader_charge'] = a.calc.results['bader_charge']
        write(traj_name, a, append=True)

    dyn.attach(write_xyz, interval=args.dump_step)

    try:
        dyn.run(args.max_steps)
    
    # Simulations stops but is accepted
    except AssertionError as e:
        logger.info("Simulation stops and is accepted")
        pass
    
    # Simulation stops and is rejected
    except ValueError as e:
        raise e

    return
