# SGPA-in-SR

## Cloning the repository

Use the following command to clone the repository including the `gplearn-sgpa` submodule.
```
git clone --recurse-submodules -j8 https://github.com/krzysztof-kacprzyk/SGPA-in-SR.git
```

## Running the experiments

### Install dependencies

Run the following to create a new conda environment with all dependencies.
```
conda env create --file=environment.yml
```

Use `conda activate sgpa-in-sr` to activate the new environment.

Then navigate to gplearn_sgpa folder and run the following command to install gplearn with sgpa constraints.
```
pip install -e .
```

Then navigate to pmlb folder and run the following command to install pmlb.
```
pip install -e .
```

### Run the gplearn experiment

Navigate and to `gplearn_sgpa/experiments` and run:
```
python run_scripts/run_all.py
```

The results will be saved in `gplearn_sgpa/experiments/results`.

### Notebooks with examples

Example usage of SGPA module can be found in `Examples.ipynb`.

Comparison with XAI techniques can be found in `Comparison.ipynb`.

### SimplEq experiments

You can run the SimplEq experiments by running `SimplEqExperiments.ipynb`.

### Correlation experiments

You can run the correlation experiments by running `Correlation.ipynb`.



