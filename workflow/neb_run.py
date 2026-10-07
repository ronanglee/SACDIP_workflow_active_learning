from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write
from ase import units
from ase.mep import NEB, NEBTools

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
        "--initial_image",
        type=str,
        help="Path to initial structure directory",
    )
    parser.add_argument(
        "--final_image",
        type=str,
        help="Path to final structure directory",
    )
    parser.add_argument(
        "--opt_steps",
        type=int,
        #default=100,
        help="Number of optimization steps",
    )
    parser.add_argument(
        "--opt_fmax",
        type=int,
        #default=0.03,
        help="The maximum force for the optimization",
    )
    parser.add_argument(
        "--num_img",
        type=int,
        #default=50,
        help="Number of images for NEB pathway",
    )
    parser.add_argument(
        "--neb_fmax",
        type=float,
        #default=0.03,
        help="The maximum force along the NEB path",
    )
    parser.add_argument(
        "--neb_steps",
        type=int,
        #default=1000,
        help="Number of steps for the NEB calculation",
    )
    parser.add_argument(
        "--climb",
        type=bool,
        #default=False,
        help="If True, the climbing image method is used",
    )
    parser.add_argument(
        "--num_MD",
        type=int,
        #default=5,
        help="Number of NEB images (sorted by max energies) to do small MD sim. upon. If None a small MD is done for all images",
    )
    parser.add_argument(
        "--time_MD",
        type=int,
        #default=5, #fs
        help="The time step of the small MD sim.",
    )
    parser.add_argument(
        "--time_step_MD",
        type=float,
        default=1,
        help="Time step of small MD simulation",
    )
    parser.add_argument(
        "--temperature_MD",
        type=float,
        #default=400,
        help="Temperture of small MD simulation",
    )
    parser.add_argument(
        "--friction_MD",
        type=float,
        #default=5,
        help="Setting the friction term in the Langevin dynamics",
    )
    parser.add_argument(
        "--print_step",
        type=int,
        default=1,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for this run",
    )

    return parser.parse_args(arg_list)

def load_images(img_IS, img_FS):
    initial = read(img_IS)
    final = read(img_FS)
    return initial, final

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

def NEB_run(params:dict,run_dir:str='.') -> None: 
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

    runHandler = logging.FileHandler(os.path.join(run_dir,'neb.log'), mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    errorHandler = logging.FileHandler(os.path.join(run_dir,'neb_error.log'), mode='w')
    errorHandler.setLevel(logging.WARNING)
    errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))

    logger.addHandler(runHandler)
    logger.addHandler(errorHandler)
    logger.addHandler(logging.StreamHandler())
    logger.warning = CallsCounter(logger.warning)
    logger.info = CallsCounter(logger.info)
    logging.info(f"Using device: {device}")
    # Load MLP model
    # Set up the MLP calculator
    ML_class = ML_Relaxer(calc_name=args.model_name,calc_paths=args.model_path,device=device,relax_cell=False)
    ml_calc = ML_class.calculator
    if not ML_class.ensemble:
        raise NotImplementedError("Only ensemble training is supported at the moment")
    
    # Read initial and final images from the given paths
    initial_image = args.initial_image
    final_image = args.final_image
    initial_dft, final_dft = load_images(initial_image, final_image)


    # Perform geometry optimization for initial and final images
    logger.info('Optimizing images with the MLP')
    initial_results=ML_class.relax(initial_dft, fmax=args.opt_fmax, steps=args.opt_steps, 
                                    traj_file=os.path.join(run_dir,f'initial.xyz'), 
                                    log_file=os.path.join(run_dir,f'initial.log'), interval=1)
    initial_ml = initial_results['final_structure']
    
    final_results=ML_class.relax(final_dft, fmax=args.opt_fmax, steps=args.opt_steps,
                                    traj_file=os.path.join(run_dir,f'final.xyz'), 
                                    log_file=os.path.join(run_dir,f'final.log'), interval=1)
    final_ml = final_results['final_structure']

    # Check if the systems converged:
    fmax_init = pd.read_csv(os.path.join(run_dir,'initial.log'),delimiter="\s+")['fmax'].values.astype(float)[-1]
    fmax_final = pd.read_csv(os.path.join(run_dir,'final.log'),delimiter="\s+")['fmax'].values.astype(float)[-1]
    if fmax_init >args.opt_fmax or fmax_final > args.opt_fmax:
        sys.exit(f'Maximum optimization step reached. System did not converged. Check your system or increase the number of optimization steps from {args.opt_steps} steps')
    
    # Make a band consisting of N images
    images = [initial_ml]
    images += [initial_ml.copy() for i in range(args.num_img)]
    images += [final_ml]

    # Setup NEB
    neb_path = os.path.join(run_dir,f'NEB.xyz')
    neb = NEB(images,allow_shared_calculator=True, climb=args.climb)#, parallel=parallel)#, method="improvedtangent")
    neb.interpolate(mic=True)
    write(os.path.join(run_dir,'neb_initial.xyz'), images)

    # Set up the calculators
    for i, image in enumerate(images):
        image.calc = ml_calc
        image.get_potential_energy()

    # Run the NEB
    logger.info('Running NEB calculation')
    optimizer = ML_class.opt_class(neb,trajectory=neb_path,logfile=neb_path.replace('xyz','log'))
        
    optimizer.run(fmax=args.neb_fmax, steps=args.neb_steps)
    logger.info('NEB calculation done')

    
    ### Do small MD sim on a few NEB images
    # Finding the last structures of NEB.
    traj_old = read(os.path.join(run_dir,'NEB.xyz'),':')
    ind = args.num_img+2 #last x images, including initial and final image
    traj_NEB = traj_old[-ind:]
    write(os.path.join(run_dir,'NEB_MD.xyz'),traj_NEB)
    
    #Finding the largest energies
    if args.num_MD:
        nebtools = NEBTools(traj_NEB)
        E = nebtools.get_fit().energies
        sort_max = np.argsort(np.abs(E))[::-1]
        image_ind = sort_max[:args.num_MD] 
    else:
        image_ind = np.arange(0,ind)
    
    # Run MD simulation on the selected images
    logger.info('random indices:')
    logger.info(image_ind)
    for i in image_ind:
        atoms = traj_NEB[i]
        atoms.calc = ML_class.calculator
        atoms.get_potential_energy()
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_MD)
        dyn = Langevin(atoms, args.time_step_MD * units.fs, temperature_K=args.temperature_MD, friction=args.friction_MD)

        def printenergy_MD(a=atoms):  # store a reference to atoms in the definition.
            """Function to print the potential, kinetic and total energy."""
            ensemble = a.calc.results['ensemble']
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

            logger.info(f"Image={i} Epot={epot} Ekin={ekin} temperature={temp} {ensemble_formatted}")
        
        dyn.attach(printenergy_MD, interval=1)

        def write_xyz(a=atoms,traj_name=os.path.join(run_dir,'NEB_MD.xyz')):
            if 'bader_charge' in a.arrays:
                a.arrays['bader_charge'] = a.calc.results['bader_charge']
            write(traj_name, a, append=True)

        logger.debug(f'MD starts for Image: {i} for {args.time_step_MD} fs')
        dyn.attach(write_xyz, interval=args.time_step_MD)
        dyn.run(args.time_MD)
    return
