import numpy as np
import os 
import json
import argparse, toml
from pathlib import Path
import logging
from ase.io import read, write
from perqueue.constants import DYNAMICWIDTHGROUP_KEY,CYCLICALGROUP_KEY, ITER_KW, INDEX_KW

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="General Active Learning", fromfile_prefix_chars="+"
    )
    parser.add_argument("--method", type=str)
    parser.add_argument("--neb_init", type=bool, default=True)
    parser.add_argument("--kernel", type=str)
    parser.add_argument("--selection", type=str)
    parser.add_argument("--n_random_features", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_name", type=str, default='cpainn')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split_file", type=str)
    parser.add_argument("--pool_set", type=str)
    parser.add_argument("--train_set", type=str)
    parser.add_argument("--random_seed", type=int)
    return parser.parse_args(arg_list)

def update_namespace(ns:argparse.Namespace, d:dict) -> None:
    for k, v in d.items():
        ns.__dict__[k] = v

# an O-O (OOH) or O-H (OH, OOH) bond present in the starting frame counts as broken beyond these.
# In the iter_0 MACE relaxations intact OOH reached 1.66 A and a splitting one went 1.62 -> 1.79 -> 2.03 A;
# the largest O-H was 1.19 A.
BOND_BREAK = {('O', 'O'): 1.7, ('H', 'O'): 1.3}


def intact_adsorbate_frames(frames):
    """Mask of frames whose O-O and O-H bonds are those of the trajectory's first frame.

    The first frame is the starting structure, so a frame where OOH has split into O + OH, or
    where H has left its O (to the carbon, an N or the metal), is masked out. The reference is
    taken again whenever the atoms change, so a file holding several trajectories also works.
    """
    keep, ref, bonds = [], None, []
    for atoms in frames:
        syms = atoms.get_chemical_symbols()
        if ref != syms:
            ref = syms
            oh = [i for i, s in enumerate(syms) if s in ('O', 'H')]
            bonds = [(i, j, BOND_BREAK[tuple(sorted((syms[i], syms[j])))])
                     for k, i in enumerate(oh) for j in oh[k + 1:]
                     if tuple(sorted((syms[i], syms[j]))) in BOND_BREAK
                     and atoms.get_distance(i, j, mic=True) < BOND_BREAK[tuple(sorted((syms[i], syms[j])))]]
        keep.append(all(atoms.get_distance(i, j, mic=True) < d for i, j, d in bonds))
    return keep


def load_pool(pool_set, pool_out_path):
    """Read the pool set and remove structures with max force > 50 eV/A or a broken O-O/O-H bond"""
    if isinstance(pool_set, str):
        pool_set = [pool_set]
    dataset = []
    broken = 0
    for traj in pool_set:
        if not os.path.exists(traj):
            logging.info(f"File {traj} does not exist!")
            continue
        if Path(traj).stat().st_size > 0:
            frames = read(traj, index=':')
            keep = intact_adsorbate_frames(frames)
            dataset += [a for a, k in zip(frames, keep) if k]
            broken += len(frames) - sum(keep)
            if not all(keep):
                logging.info(f"{traj}: {len(frames) - sum(keep)}/{len(frames)} frames with a broken O-O/O-H bond removed")
    logging.info(f"Removed {broken} frames with a broken O-O/O-H bond")

    dataset_new = []
    max_force_threshold = 50
    skipped = 0
    for atoms in dataset:
        try:
            atoms.get_potential_energy()
            forces = atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force <= max_force_threshold:
                dataset_new.append(atoms)
            else:
                skipped += 1
                logging.info(f"Skipping structure with max force {max_force:.2f}")
        except Exception as e:
            logging.info(f"Skipping structure due to error: {e}")
    logging.info(
        f"Loaded {len(dataset_new)} valid structures out of {len(dataset)} "
        f"(skipped {skipped} high-force structures)"
    )
    if broken or len(dataset) != len(dataset_new):
        write(pool_out_path, dataset_new)
        logging.info(f"Filtered structures written to {pool_out_path}")
        return dataset_new, pool_out_path
    return dataset_new, pool_set


def select_mace(args, system_name, system_dir, run_path, iter_idx, device):
    """Active learning selection with an ensemble of MACE models (features are MACE descriptors)"""
    from cPaiNN.active_learning import GeneralActiveLearning
    from cPaiNN.mace_tools import load_mace_calculator

    calc = load_mace_calculator(args.model_path, device=device)
    assert args.pool_set and args.train_set

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(system_dir,"al_log.log"), mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Selected system: {system_name}")
    logging.info(f"Selected method: {args.method}")
    logging.info(f"Using {calc.num_models} MACE model(s) from {args.model_path}")
    logging.info(f"Loding the dataset from {args.pool_set} and {args.train_set}")

    pool, args.pool_set = load_pool(
        args.pool_set,
        os.path.join(run_path,'simulate', f'iter_{iter_idx}',system_name,'filtered_pool_set.xyz'),
    )
    datasets = {'pool': pool, 'train': read(args.train_set, index=':')}
    logging.info(f"Train set size: {len(datasets['train'])}")
    logging.info(f"Pool set size: {len(datasets['pool'])}")

    if args.method == 'MD' and len(datasets['pool']) < args.batch_size*10:
        raise RuntimeError(f"""The pool data set ({len(datasets['pool'])}) is not large enough for selection!
        It should be larger than 10 times batch size ({args.batch_size*10}).
        Check your simualtion!""")
    if args.method not in ['MD', 'optimize']:
        raise NotImplementedError(f"Selection for method {args.method} is not implemented for MACE")

    al = GeneralActiveLearning(
        kernel=args.kernel,
        selection=args.selection,
        n_random_features=args.n_random_features,
    )
    logging.info(f"Selecting {args.batch_size} structures")
    indices = al.select_mace(calc, datasets, al_batch_size=min(args.batch_size, len(datasets['pool'])))

    al_info = {
        'kernel': args.kernel,
        'selection': args.selection,
        'dataset': args.pool_set,
        'selected': indices,
    }
    with open(os.path.join(system_dir,'selected.json'), 'w') as f:
        json.dump(al_info, f)

    return True, {'system_name':system_name}


def main(cfg,system_name,**kwargs):

    from cPaiNN.active_learning import GeneralActiveLearning
    from cPaiNN.data import AseDataset
    from cPaiNN.model import PainnModel
    from cPaiNN.relax import ML_Relaxer
    import torch
    from cPaiNN.utils import setup_seed

    #return True, {'system_name':system_name}
    # Create device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Put a tensor on the device before loading data
    # This way the GPU appears to be in use when other users run gpustat
    torch.tensor([0], device=device)

    # Load iteration index
    iter_idx,*_ = kwargs[ITER_KW]

    # Load all parameters from config file
    with open(cfg, 'r') as f:
        main_params = toml.load(f)

    # Load local parameters
    task_name = 'select'
    params_select = main_params[task_name]

    # Add random seed and model path to the trained MLPs
    params_select['random_seed'] = main_params['global']['random_seed']
    run_path = main_params['global']['run_path']
    params_select['model_path'] = os.path.join(run_path,'train', f'iter_{iter_idx}')
    # Use the same model type as the train step ('cpainn' or 'mace')
    params_select.setdefault('model_name', main_params['train'].get('model', 'cpainn'))

    
    # Load the system parameters
    system_params = params_select['runs'][system_name]
    method = system_params['method']
    params_select['method'] = method

    # remove runs from params
    params_select.pop('runs')

    # Find the system dataset and update the parameters
    sytem_path = os.path.join(run_path,'simulate', f'iter_{iter_idx}',system_name)

    if method == 'MD':
        pool_set = [os.path.join(sytem_path,'MD.xyz'),
                    os.path.join(sytem_path,'warning_struct.xyz')]

    elif method == 'NEB':
        pool_set = [os.path.join(sytem_path,'NEB_MD.xyz')]
        args_system_simulate = json.load(open(os.path.join(sytem_path,'arguments.json')))
        neb_img = args_system_simulate['num_img'] + 2

    else:
        with open(os.path.join(sytem_path,'pool_set.json')) as f:
            pool_set = json.load(f)

    params_select['pool_set'] = pool_set
    # Update the main parameters with the system parameters
    params_select.update(system_params)

    # Get the Namespace and update the parameters
    args = get_arguments()
    update_namespace(args, params_select)

    setup_seed(args.random_seed)

    # System directory
    run_path = main_params['global']['run_path']
    system_dir = os.path.join(run_path,task_name, f'iter_{iter_idx}',system_name)
    # Create the iteration directory
    if not os.path.exists(system_dir):
        os.makedirs(system_dir)

    # Save parsed arguments
    with open(os.path.join(system_dir,"arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # Save toml parameters
    with open(os.path.join(system_dir, "params.toml"), "w") as f:
        toml.dump(params_select, f)

    if args.model_name == 'mace':
        return select_mace(args, system_name, system_dir, run_path, iter_idx, device)

    # Load models
    if args.model_name != 'cpainn':
        raise NotImplementedError("Only cpainn and mace models are supported")

    models = []
    model_pth = Path(args.model_path).rglob('*best_model.pth')

    for each in model_pth:
        state_dict = torch.load(each, map_location=torch.device(device))

        model = PainnModel(
            num_interactions=state_dict["num_layer"],
            hidden_state_size=state_dict["node_size"],
            cutoff=state_dict["cutoff"],
            compute_forces=state_dict["compute_forces"],
            compute_stress=state_dict["compute_stress"],
            compute_magmom=state_dict["compute_magmom"],
            compute_bader_charge=state_dict["compute_bader_charge"],
        )

        model.to(device)
        model.load_state_dict(state_dict["model"])
        models.append(model)

    # Test if models is a list and there is a pool set and train set
    assert isinstance(models, list)
    assert args.pool_set and args.train_set

    # set logger
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(system_dir,"al_log.log"), mode="w"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Selected system: {system_name}")
    logging.info(f"Selected method: {args.method}")

    logging.info(f"Loding the dataset from {args.pool_set} and {args.train_set}")
    # Load pool set (drops high-force frames and frames with a broken O-O/O-H bond)
    dataset, args.pool_set = load_pool(
        args.pool_set,
        os.path.join(run_path,'simulate', f'iter_{iter_idx}',system_name,'filtered_pool_set.xyz'),
    )


    if models[0].compute_magmom and models[0].compute_bader_charge:
        charge_key = ['magmom','bader']
    elif models[0].compute_magmom:
        charge_key = 'magmom'
    elif models[0].compute_bader_charge:
        charge_key = 'bader_charge'
    else:
        charge_key = None

    data_dict = {
        'pool': AseDataset(
            dataset,
            cutoff=models[0].cutoff,
            compute_forces=models[0].compute_forces,
            compute_stress=models[0].compute_stress,
            charge_key=charge_key
        ),

        'train': AseDataset(
            args.train_set,
            cutoff=models[0].cutoff,
            compute_forces=models[0].compute_forces,
            compute_stress=models[0].compute_stress,
            charge_key=charge_key
        )
    }

    logging.info(f"Train set size: {len(data_dict['train'])}")
    logging.info(f"Pool set size: {len(data_dict['pool'])}")

    # raise error if the pool dataset is not large enough
    if args.method =='MD':
        if len(data_dict['pool']) < args.batch_size*10 :
            raise RuntimeError(f"""The pool data set ({len(data_dict['pool'])}) is not large enough for selection!
            It should be larger than 10 times batch size ({args.batch_size*10}).
            Check your simualtion!""")

    # Select structures
    al = GeneralActiveLearning(
        kernel=args.kernel,
        selection=args.selection,
        n_random_features=args.n_random_features,
    )

    logging.info(f"Selecting {args.batch_size} structures")
    # Manually choose N NEB images to be labeled.
    if args.method == 'NEB':
        indices_neb = np.arange(0,neb_img)
        if neb_img > args.batch_size:
            raise RuntimeError(f"""The pool data set is not large enough for selection! 
            Choose a batch size ({args.batch_size}) larger then NEB selection ({neb_img})!""")

        if neb_img == args.batch_size:
            indices = indices_neb.tolist()

        else:
            data_dict['pool'] = AseDataset(dataset[neb_img:], cutoff=models[0].cutoff)
            indices_al = al.select(
                models,
                data_dict,
                al_batch_size=args.batch_size-neb_img
            )
            indices = np.concatenate(
                (indices_neb,np.array(indices_al)+neb_img),
                dtype=int
            ).tolist()
            
    # Choose N MD images to be labeled.
    elif args.method == 'MD' or args.method == 'optimize':
        indices = al.select(models, data_dict, al_batch_size=args.batch_size)

    else:
        raise RuntimeError("Please give valid method!")

    al_idx = indices

    al_info = {
        'kernel': args.kernel,
        'selection': args.selection,
        'dataset': args.pool_set,
        'selected': al_idx,
    }

    with open(os.path.join(system_dir,'selected.json'), 'w') as f:
        json.dump(al_info, f)

    return True, {'system_name':system_name}


if __name__ == "__main__":
    main()
