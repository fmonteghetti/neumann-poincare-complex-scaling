# Embedded eigenvalues and complex resonances of the Neumann-Poincaré operator on domains with corners

This repository contains the code producing the plots of:

[1] "A complex-scaled boundary integral equation for the embedded
eigenvalues and complex resonances of the Neumann-Poincaré
operator on domains with corners."
Luiz M. Faria, Florian Monteghetti.
TODO: link

# Description

The work [1] introduces a complex-scaled boundary integral equation that enables to compute the embedded eigenvalues and complex resonances of the Neumann-Poincaré operator. The code in this repository performs two things:

1. Discretizing the complex-scaled Neumann-Poincaré operator using a Nyström method. The implementation is done using the Julia library `Intil.jl`.

2. Discretizing the (quasi-static) plasmonic eigenvalue problem using a finite element method. This is done using the C++/python library `fenicsx`.


# Content

`bem`: Julia module implementing the Nyström method using `Inti.jl`. 

`fem`: Python module implementing the discretization of the plasmonic eigenvalue problem using `fenicsx`.

`scripts`: Python scripts producing the plots of [1]. The Julia code is called using the `juliacall` package.

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
