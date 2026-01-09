from pprint import pprint
import sys
import platform
from pathlib import Path
import torch

# Asegurar que el repositorio raiz esté en sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import config
from utils.search_space import HS_SEARCH_SPACE


def summarize_space(name, space):
    return {
        "name": name,
        "keys": list(space.keys()),
        "values": space,
    }


def env_info():
    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main():
    info = {
        "env": env_info(),
        "dataset": config.DATASET_QUECHUA_PATH,
        "out_dir": str(config.OUT_DIR),
        "checkpoint_dir": str(config.CHECKPOINT_DIR),
        "hs_params": {
            "HMS": 12,
            "HMCR": 0.95,
            "PAR": 0.30,
            "BW": "por parametro (bw en search_space)",
            "NI": 6,
            "resume_state": str(config.OUT_DIR / "state_hs.json"),
        },
        "search_spaces": [
            summarize_space("HS_SEARCH_SPACE", HS_SEARCH_SPACE),
        ],
    }
    pprint(info)


if __name__ == "__main__":
    main()
