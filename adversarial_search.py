from pathlib import Path

import numpy as np

import numpy as np
from adversarial import find_noise_for_images
from grid import get_combinations


def grid_search():
    params_grid = {
        "lr": np.logspace(0.0001, 10, 10, endpoint=True),
    }
    params_configurations = get_combinations(params_grid)

    results = []
    for e, d in enumerate(params_configurations):
        r = find_noise_for_images(output_dir=Path(f"IMAGES_{e}"), **d)
        results.append(r)

    print(params_configurations[np.argmin(results)])


grid_search()