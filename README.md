# BZK Address Parsing

# Coming from the HIP 26 Paper

> A Prompt Optimization Framework for Parsing Historical Addresses

Paper under review for [HIP 26](https://blog.sbb.berlin/hip2026/#cfp)

- **Dataset (BZK Open Addresses)** The dataset is found in the `open_data` directory.
    - Statistics can be found in `dataset_statistics.ipynb`
- **Section 4.1** Experiments with Optuna and results are located in the `optuna_llms.ipynb`. This includes all details about the search space
- **Section 4.2** Cross val evaluation is found in `cross_val_evaluation.ipynb`
    - **Error analysis** is based on prediction errors listed in `optuna_llms.ipynb`
- **LLM prompts** are found in the `prompts` directory
- **Predictions** for the experiments reported in the paper are provided in `experiments_data` and will automatically be loaded when running the notebooks. **Replicating the experiments** requires deleting this directory first, and then runnning the notebooks
- `compare.ipynb` and `error_analysis.ipynb` pertain to early experiments with manual prompt design, only vaguely described in the paper.
- `modules` directory contains utility functions and classes for the rest of the project

Other files are not relevant to the paper.


# Running

## Hardware Requirements

The high computation requirements of the experiments require a GPU or other hyperparallelization hardware compatible with pytorch. Experiments should be able to take advantage of any pytorch compatible hardware as long as the correct pytorch binaries are installed.

## Software Dependencies

Every python dependency can be installed using `uv sync`

Aside these dependencies, docker and docker-compose are also required to test the libpostal library.
