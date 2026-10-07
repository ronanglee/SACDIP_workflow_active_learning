from pathlib import Path
import toml, os
import numpy as np
from perqueue import PersistentQueue
from perqueue.task_classes.task import Task
from perqueue.task_classes.task_groups import (
    CyclicalGroup,
    Workflow,
    DynamicWidthGroup,
    StaticWidthGroup,
)
from perqueue.constants import DYNAMICWIDTHGROUP_KEY

### Load config file ###
config_file_path = "config.toml"  # Absolute path to Config.toml file
config = toml.load(config_file_path)

# Setup autocPaiNN workflow path
workflow = Path(config["global"]["workflow_path"])

### Setup folders at the run directory ###
run_path = Path(config["global"]["run_path"])
if not os.path.isdir(run_path):
    os.makedirs(run_path)


# Setup task directory accoring to their names in config file
if not os.path.isdir(run_path / "train"):
    os.makedirs(run_path / "train")
if not os.path.isdir(run_path / "simulate"):
    os.makedirs(run_path / "simulate")
if not os.path.isdir(run_path / "select"):
    os.makedirs(run_path / "select")
if not os.path.isdir(run_path / "labeling"):
    os.makedirs(run_path / "labeling")

### Task resources ###
# resources = "8:1:1:sm3090_devel:1h"
resources = "8:1:sm3090_devel:4h"
resources2 = "24:1:xeon24el8:25h"
Train_resources = resources
Simulation_resources = resources
Activelearning_resources = resources
Labeling_resources = resources2

### Set Tasks ###
main_argument = {"cfg": str(config_file_path)}

Train = Task(
    workflow / "workflow/train.py",
    name="Train MLP model",
    args=main_argument,
    resources=Train_resources,
)
Simulation = Task(
    workflow / "workflow/simulation.py",
    name="Run ML simulation",
    args=main_argument,
    resources=Simulation_resources,
)
Activelearning = Task(
    workflow / "workflow/al_select.py",
    name="Run active learning",
    args=main_argument,
    resources=Activelearning_resources,
)
Labeling = Task(
    workflow / "workflow/labeling.py",
    name="Run DFT labeling",
    args=main_argument,
    resources=Labeling_resources,
)

### Set Task Groups ###
# If you perform ensemble training more models are train or else one main model is trained
if "ensemble" in config["train"]:
    swg_width = len(config["train"]["ensemble"])
else:
    swg_width = 1

# Initial training of the MLP model
swg = StaticWidthGroup(Train, width=swg_width)

# The MLP labeling step where each structure follows each other independently
dwg_labeling = DynamicWidthGroup([Simulation, Activelearning, Labeling])

# The MLP training step where the model is trained after labeling. Here all structure needs to be finsihsed before the next training step
dwg_train = DynamicWidthGroup(Train)

# Combine the steps into a cyclical group
cg = CyclicalGroup(
    [dwg_labeling, dwg_train], max_tries=config["global"]["max_iteration"]
)

### Set Workflow ###
wf = Workflow({swg: [], cg: [swg]})

### Submit Workflow ###
with PersistentQueue() as pq:
    pq.submit(wf)
