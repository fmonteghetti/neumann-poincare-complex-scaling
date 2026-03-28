# Embedded eigenvalues and complex resonances of the Neumann-Poincaré operator on domains with corners

This repository contains the code producing the plots of:

[1] "A complex-scaled boundary integral equation for the embedded
eigenvalues and complex resonances of the Neumann-Poincaré
operator on domains with corners."
Luiz M. Faria, Florian Monteghetti.
[postprint](https://hal.science/hal-04970403)
[article](https://www.sciencedirect.com/science/article/abs/pii/S0898122125003402)

# Description

The work [1] introduces a complex-scaled boundary integral equation that enables to compute the embedded eigenvalues and complex resonances of the Neumann-Poincaré operator. The code in this repository performs two things:

1. Discretizing the complex-scaled Neumann-Poincaré operator using a Nyström method. The implementation is done using the Julia library `Intil.jl`.

2. Discretizing the (quasi-static) plasmonic eigenvalue problem using a finite element method. This is done using the C++/python library `fenicsx`.

# Content

`ComputationalPlasmonicsBEM`: Julia module implementing the Nyström method using `Inti.jl`. 

`computational_plasmonics_FEM`: Python module implementing the discretization of the plasmonic eigenvalue problem using `fenicsx`.

`np_complex_scaling`: Python module that relies on `ComputationalPlasmonicsBEM` and `computational_plasmonics_FEM` to discretize the Neumann-Poincaré operator and the plasmonic eigenvalue problem. The Julia code is called using the `juliacall` package.

`scripts`: Python scripts producing the plots of [1]. 

# Usage

The code can be run in either an Anaconda or a docker environment.

## Anaconda 

A new conda environment suited to run the scripts can be obtained with:

```console
conda env create --file environment.yml 
```

Run the scripts with
```
conda activate article_np
cd scripts
python3 script_name.py
```

## Docker (devcontainer)

A devcontainer environement is provided in `.devcontainer`. To use it, in the command palette of VS code: 

- `Dev Containers: Open Folder in Container` and select the root of this repository.
- `Python: Select Interpreter` and select the python environment.
- Run the scripts using `Python: Run Current File in Interactive Window`.  
