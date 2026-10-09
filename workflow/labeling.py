from ase.io import read, write, Trajectory
from ase.db import connect
from shutil import copy
import os, subprocess
import numpy as np
import argparse
import json
import toml
from pathlib import Path
from ase.calculators.calculator import CalculationFailed
import re

from perqueue.constants import DYNAMICWIDTHGROUP_KEY,CYCLICALGROUP_KEY, ITER_KW, INDEX_KW

# Any SCF line in OSZICAR, e.g. "DAV:  12", "RMM:  3", "CG :  5", "DMP:  1"
SCF_LINE_RE = re.compile(r"^\s*[A-Za-z]{2,4}\s*:\s*\d+")
MAX_FORCE = 5.0  # eV/A, frames above this are treated as unphysical


def read_incar_value(incar_path, key, default=None, cast=float):
    """Read a single tag from INCAR (ignores comments)."""
    try:
        with open(incar_path) as fh:
            for line in fh:
                line = line.split('#')[0].split('!')[0]
                if '=' in line:
                    k, v = line.split('=', 1)
                    if k.strip().upper() == key.upper():
                        return cast(v.split()[0])
    except FileNotFoundError:
        pass
    return default


def scf_steps_per_ionic(oszicar_path):
    """Number of SCF iterations for each ionic step, from OSZICAR."""
    counts, current = [], 0
    with open(oszicar_path) as fh:
        for line in fh:
            if SCF_LINE_RE.match(line):
                current += 1
            elif 'F=' in line:      # end of an ionic step
                counts.append(current)
                current = 0
    return counts


def select_by_fmax(fmaxs, fmin=0.01, fmax_cap=MAX_FORCE, nbins=12, per_bin=2):
    """Pick up to `per_bin` frames per log-spaced fmax bin, plus first and last."""
    bins = np.logspace(np.log10(fmin), np.log10(fmax_cap), nbins)
    counts, keep = {}, []
    for i, f in enumerate(fmaxs):
        if np.isnan(f):
            continue
        b = int(np.digitize(f, bins))
        if counts.get(b, 0) < per_bin:
            keep.append(i)
            counts[b] = counts.get(b, 0) + 1
    return sorted(set(keep + [0, len(fmaxs) - 1]))


def label_vasp_trajectory(system_dir, db, al_ind, iter_idx, warning_traj):
    """Write a force-binned subset of the ionic steps in vasprun.xml to `db`.

    Returns (n_written, n_selected, scf_ok_all, ionic_ok).
    """
    incar = os.path.join(system_dir, 'INCAR')
    nelm = read_incar_value(incar, 'NELM', default=60, cast=int)
    nsw = read_incar_value(incar, 'NSW', default=0, cast=int)
    ediffg = read_incar_value(incar, 'EDIFFG', default=-0.05)
    f_target = abs(ediffg) if ediffg < 0 else None   # force criterion only if EDIFFG < 0

    frames = read(os.path.join(system_dir, 'vasprun.xml'), index=':')
    n = len(frames)
    scf_counts = scf_steps_per_ionic(os.path.join(system_dir, 'OSZICAR'))

    if len(scf_counts) != n:
        print(f"WARNING [{al_ind}]: {n} frames in vasprun.xml but "
              f"{len(scf_counts)} ionic steps in OSZICAR; skipping SCF check.")
        scf_ok = [True] * n
    else:
        scf_ok = [c < nelm for c in scf_counts]

    # fmax over free atoms (what EDIFFG acts on) for selection/convergence,
    # over all atoms for the unphysical-force filter
    fmaxs = np.array([np.max(np.linalg.norm(a.get_forces(), axis=1)) for a in frames])
    fmaxs_all = np.array([np.max(np.linalg.norm(a.get_forces(apply_constraint=False), axis=1))
                          for a in frames])

    # ionic convergence: final force below EDIFFG target, or finished before NSW
    final_fmax = fmaxs[-1]
    if nsw <= 1:
        ionic_ok = True     # single point, nothing to converge
    elif f_target is not None:
        ionic_ok = bool(final_fmax < f_target)
    else:
        ionic_ok = n < nsw
    if not ionic_ok:
        print(f"WARNING [{al_ind}]: ionic relaxation not converged "
              f"(final fmax = {final_fmax:.3f} eV/Å, {n} steps).")

    keep = select_by_fmax(fmaxs, fmin=0.01, fmax_cap=MAX_FORCE, nbins=12, per_bin=2)

    n_written = 0
    for step in keep:
        a = frames[step]
        a.set_pbc(True)
        fmax = float(fmaxs_all[step])

        # 'fmax' is a reserved ASE db key, hence 'max_force'
        meta = dict(al_ind=al_ind, iter=iter_idx, step=step, nsteps=n,
                    max_force=fmax, is_initial=(step == 0), is_final=(step == n - 1),
                    ionic_converged=ionic_ok)

        # SCF not converged at this ionic step -> forces unreliable
        if not scf_ok[step]:
            db.write(a, converged=False, reason='scf', **meta)
            continue

        # unphysical forces
        if fmax > MAX_FORCE or np.isnan(fmax):
            print(f"Skipping {al_ind} step {step}: large force {fmax:.2f} eV/Å")
            a.info.update(meta)
            write(warning_traj, a, append=True)
            db.write(a, converged=False, reason='large_force', **meta)
            continue

        db.write(a, converged=True, **meta)
        n_written += 1

    print(f"[{al_ind}] kept {n_written}/{len(keep)} selected frames "
          f"from {n} ionic steps (final fmax = {final_fmax:.3f} eV/Å)")

    return n_written, len(keep), all(scf_ok[s] for s in keep), ionic_ok


def get_processed_al_inds(db):
    """
    Return all al_ind values already present in the database,
    regardless of converged=True/False.
    """
    processed = []
    for row in db.select():
        try:
            processed.append(row.al_ind)
        except Exception:
            pass
    return processed


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="DFT labeling of the selected datset", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--label_set",
        type=str,
        help="Path to trajectory to be labeled by DFT",
    )
    parser.add_argument(
        "--train_set",
        type=str,
        help="Path to existing training data set",
    )
    parser.add_argument(
        "--pool_set", 
        type=str, 
        help="Path to MD trajectory obtained from machine learning potential",
    )
    parser.add_argument(
        "--al_info", 
        type=str, 
        help="Path to json file that stores indices selected in pool set",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="arguments.toml",
        help="Path to config file. e.g. 'arguments.toml'"
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

def main(cfg,system_name,**kwargs):

    # Load iteration index
    iter_idx,*_ = kwargs[ITER_KW]

    with open(cfg, 'r') as f:
        main_params = toml.load(f)

    # Find how many models to train and return to the run directory
    #params_train = main_params['train']
    #if 'ensemble' in params_train:
    #    dmkey = len(list(params_train['ensemble'].keys()))
    #else:
    #    dmkey = 1

    #return True, {DYNAMICWIDTHGROUP_KEY: dmkey}
    
    # Load local parameters
    task_name = 'labeling'
    params = main_params[task_name]

    # Labeling harvests ionic steps from vasprun.xml/OSZICAR/INCAR, so only VASP is supported
    if params.get('method') != 'VASP':
        raise NotImplementedError(
            f"labeling.py only supports method='VASP' (got {params.get('method')!r}); "
            "the trajectory-harvesting logic relies on VASP output files."
        )

    #ronan
    # Load the system parameters
    system_params = params['runs'][system_name]
    #     
    # # Update the main parameters with the system parameters
    # params.update(system_params)
    #ronan

    # Set the data set selected by the active learning 
    run_path = main_params['global']['run_path']
    al_path = os.path.join(run_path,'select', f'iter_{iter_idx}',system_name)
    params['al_info'] = os.path.join(al_path,'selected.json')

    # Get the pool set and the selected indices
    with open(params['al_info']) as f:
        al_data = json.load(f)
    
    params['pool_set'] = al_data['dataset']  
    al_indices = al_data["selected"]  
    
    # Get namepsace and update it with the parameters
    args = get_arguments()
    update_namespace(args, params)

    # System directory
    run_path = main_params['global']['run_path']
    system_dir = os.path.join(run_path,task_name, f'iter_{iter_idx}',system_name)
    print(f"Running DFT calculations in {system_dir}")
    # Create the iteration directory
    if not os.path.exists(system_dir):
        os.makedirs(system_dir)

    # Save parsed arguments
    with open(os.path.join( system_dir,"arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # Save the total parameters
    with open(os.path.join(system_dir, "params.toml"), "w") as f:
        toml.dump(params, f)
    
    # Set up dataframe and load all already processed data id's
    db = connect(os.path.join(system_dir,'dft_structures.db'))
    db_al_ind = get_processed_al_inds(db)
    
    # Get images and set parameters
    if args.label_set:
        images = read(args.label_set, index = ':')
    elif args.pool_set:
        if isinstance(args.pool_set, list):
            pool_traj = []
            for pool_path in args.pool_set:
                # Check if the file exists
                try: 
                    os.path.exists(pool_path)
                except FileNotFoundError:
                    print(f"File {pool_path} does not exist!")
                    continue
                if not os.path.exists(pool_path):
                    print(f"File {pool_path} does not exist!")
                    continue
                
                # Check if the file is empty and read it if not
                if Path(pool_path).stat().st_size > 0:
                    pool_traj += read(pool_path, index=':')
        else:
            pool_traj = read(args.pool_set,index=':')
        
        if db_al_ind:
            _,rm,_ = np.intersect1d(al_indices, db_al_ind,return_indices=True)
            al_indices = np.delete(al_indices,rm)
        images = [pool_traj[i] for i in al_indices]        
    else:
        raise RuntimeError('Valid configarations for DFT calculation should be provided!')
    
    vasp_params = params['VASP']

    check_result = False
    check_result_elec = False
    check_result_ionic = False
    failed_idx, scf_idx, ionic_idx = [], [], []
    warning_traj = os.path.join(system_dir, 'large_force.xyz')

    from ase.calculators.vasp import Vasp
    # set environment variables
    #os.putenv('ASE_VASP_VDW', '/home/energy/modules/software/VASP/vasp-potpaw-5.4')
    #os.putenv('VASP_PP_PATH', '/home/energy/modules/software/VASP/vasp-potpaw-5.4')
    #os.putenv('ASE_VASP_COMMAND', 'mpirun vasp_std')

    # Update the parameters
    vasp_params.update(system_params)

    vasp_params["lcharg"] = False
    vasp_params["lwave"] = False

    calc = Vasp(**vasp_params,directory=system_dir)
    for i, atoms in enumerate(images):
        al_ind = al_indices[i]
        atoms.set_pbc([True,True,True])
        atoms.calc = calc
        try:
            atoms.get_potential_energy()
            n_written, n_selected, scf_ok, ionic_ok = label_vasp_trajectory(
                system_dir, db, al_ind, iter_idx, warning_traj
            )
        except (CalculationFailed, ValueError, IndexError, FileNotFoundError) as e:
            print(f"VASP calculation failed or output unparseable (e.g. lost atoms) for index {al_ind}: {e}")
            check_result = True
            db.write(atoms,al_ind=al_ind,iter=iter_idx,converged=False,reason='failed')
            failed_idx.append(int(al_ind))
            try: # If it did not save the CHGCAR
                copy(os.path.join(system_dir,'OSZICAR'), os.path.join(system_dir,f'OSZICAR_{i}_{al_ind}'))
                copy(os.path.join(system_dir,'CHGCAR'), os.path.join(system_dir,f'CHGCAR_{i}_{al_ind}'))
                os.remove(os.path.join(system_dir,'CHGCAR'))
            except FileNotFoundError as e:
                print(e)
                pass
            continue

        if not scf_ok:
            check_result_elec = True
            scf_idx.append(int(al_ind))
        if not ionic_ok:
            check_result_ionic = True
            ionic_idx.append(int(al_ind))
        try: # If you do not save CHGCAR
            copy(os.path.join(system_dir,'OSZICAR'), os.path.join(system_dir,f'OSZICAR_{i}_{al_ind}'))
            copy(os.path.join(system_dir,'CHGCAR'), os.path.join(system_dir,f'CHGCAR_{i}_{al_ind}') )
            copy(os.path.join(system_dir,'OUTCAR'), os.path.join(system_dir,f'OUTCAR_{i}_{al_ind}') )
        except FileNotFoundError as e:
            print(e)
        try: # Same for WAVECAR
            os.remove(os.path.join(system_dir,'WAVECAR'))
            os.remove(os.path.join(system_dir,'CHGCAR'))
        except FileNotFoundError as e:
            print(e)


    if check_result:
        print("WARNING: Some DFT calculations failed.")
        print(f"Failed indices: {failed_idx}")
        print(f"Total failed: {len(failed_idx)}")
    if check_result_elec:
        print("WARNING: Some selected ionic steps did not converge electronically (saved with converged=False).")
        print(f"Affected indices: {scf_idx}")
        print(f"Total affected: {len(scf_idx)}")
    if check_result_ionic:
        print("WARNING: Some DFT calculations did not converge on ionic steps. Still saved the structures to the database.")
        print(f"Unconverged indices: {ionic_idx}")
        print(f"Total unconverged: {len(ionic_idx)}")


    # write to training set
    if args.train_set:
        # train_traj = Trajectory(args.train_set, mode = 'a')
        train_traj = read(args.train_set, index = ':')
        database = connect(os.path.join(system_dir,'dft_structures.db'))
        for row in database.select([('converged','=','True')]):
            atom = row.toatoms()
            atom.info['path'] = str(Path('dft_structures.db').resolve())
            train_traj.append(atom)
        write(args.train_set, train_traj)
    
    # Find how many models to train and return to the run directory
    # Load them to the main parameters again to give the opertunity to change the number of MLP doing the labeling step
    with open(cfg, 'r') as f:
        main_params = toml.load(f)
    params_train = main_params['train']
    if 'ensemble' in params_train:
        dmkey = len(list(params_train['ensemble'].keys()))
    else:
        dmkey = 1

    return True, {DYNAMICWIDTHGROUP_KEY: dmkey}
if __name__ == "__main__":
    main()
