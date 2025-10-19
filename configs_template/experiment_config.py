from pathlib import Path

# Global config for the experiment
from src.transforms import get_transform

RANDOM_SEED = 1  # If None, random seed will be randomly sampled
SAVE_RESULTS = True
OVERWRITE = True  # If True, will erase all previous content of SAVE_DIR

# Check this Before experiments
MODELNAME_LIST = ["ProtoNet", "MatchingNet", "TransPropNet", "TransFineTune", "TP"]

# BATCH
PGADA = True
GPU_ID = 0
MODEL = 0
N_SHOT = 5
N_TARGET = 16
BATCH = "ERM_test_2"
GROUP = f"{BATCH}"
EXP_INFO = f"{MODELNAME_LIST[MODEL]}:{N_SHOT}shot-{N_TARGET}target"
EXP_NAME = "META_DATASET"

# ablation setting
PROPORTION = 0.5
MULTI_PERTUBATION = -1
ENCODING_IN_PHI = 0
FIX_G = 0

# TEST
QUERY_RESIZE = True
INVERSE_RESIZE = True
SMART_RESIZER = True
FORCE_OT = 1

# Batch save
DESCRIBE = ""
if EXP_NAME != "":
    SAVE_DIR = Path(
        f"outputs/{BATCH}/{EXP_NAME}/{N_TARGET}target-{N_SHOT}shot/{MODELNAME_LIST[MODEL]}/"
    )
    ERM_MODEL_DIR = Path(f"outputs/{BATCH}/{EXP_NAME}/erm_model")
else:
    SAVE_DIR = Path(
        f"outputs/{BATCH}/{N_TARGET}target-{N_SHOT}shot/{MODELNAME_LIST[MODEL]}/"
    )
    ERM_MODEL_DIR = Path(f"outputs/{BATCH}/erm_model")
# Additional parameters, only used for polyaxon
USE_POLYAXON = True
SECOND_EVAL_SAVE_DIR = SAVE_DIR / "extra_eval"

# Detail
CELOSS_R = False
CELOSS_R_IN_R = False
R_KLD = False
