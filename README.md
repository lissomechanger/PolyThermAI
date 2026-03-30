# PolyThermAI: Bottom-up Polymer Thermal Conductivity Tool via Machine Learning

This repository provides a tool (named PolyThermAI) for calculating the thermal conductivity (*κ*) of amorphous polymers using the Green-Kubo (GK) formalism based on [DMFF](https://github.com/deepmodeling/DMFF) platform, as detailed in the work by Tu et al., *J. Phys. Chem. B.* 2025, 129, 33, 8593–8602. PolythermAI is designed to work with hybrid machine learning potentials (MLPs) which integrates physics-driven long-range and data-driven short-range interactions, such as [PhyNEO](https://github.com/junminchen/PhyNEO).

## Key Concepts

The workflow follows a three-step process:

1. **Dynamics Sampling** (`PhyNEO or other polymer MLPs`): Generate accurate molecular dynamics trajectories (positions and velocities) using ab initio hybrid potential.
2. **Heat Flux Calculation** (`compt_dp_J.py`, `compute_eann_J.py`, `compute_gnn_J.py`): Compute the instantaneous heat flux vector ***J***  for each frame of the trajectory.
3. **Thermal Conductivity Calculation** (`calc_corr.py`, `integrate.py`): Compute the heat flux autocorrelation function (HFACF) and integrate it using GK relation to obtain *κ*.

A central idea from the paper is the **ML-facilitated heat flux calculation**: The local heat flux terms are directly computed using automatic diffrentiable (AD) technique, and the nonlocal long-range polarizable heat flux is computed using the ML technique. As the analytical heat flux of complex polarizable interaction is difficult to derive, a localized ML model (e.g., Deep Potential) is trained to fit the non-bonding energy and virial, which are then used to compute the heat flux. This approach is validated by comparing with exact results from classical polymer force fields (eg., PCFF, OPLS).

## Repository Structure

```
├── README.md                            # This file
├── Docs
    ├── Theory.md                        # Detailed theoretical background (PolyThermAI)
    ├── ML_heatflux.md                   # Documentation for compt_dp_J.py (DeepMD)
    ├── EANN_heatflux.md                 # Documentation for compute_eann_J.py (EANN)
    ├── SGNN_heatflux.md                 # Documentation for compute_gnn_J.py (sGNN)
    ├── TCF.md                           # Documentation for calc_corr.py (Autocorrelation)
    ├── Integration.md                   # Documentation for integrate.py (Integration)
├── Heatflux
    ├── compute_dp_J.py                  # DeepMD heat flux calculator
    ├── compute_eann_J.py                # EANN heat flux calculator
    ├── compute_gnn_J.py                 # SGNN heat flux calculator
├── Thermal_conductivity
    ├── tc.sh                            # Automated integration script
    ├── calc_corr.py                     # Autocorrelation calculator
    └── integrate.py                     # Integration script
```

## Theory

See ***Docs*** for theoretical backgrouds and code analysis.

## Workflow Overview

1. **Prepare Inputs** : Ensure you have sampled initial configurations in NPT+ NVT simulations and run dynamics in NVE ensembles. You need 20 (or more) NVE trajectories for ensemble average. Each trajectory documents positions and velocities. 
2. **Compute Heat Flux** : Choose the appropriate script for your potential and run it to generate a `J_*.txt` file.
3. **Merge Heat Flux**: Add heat flux data for all individual potentials (DP, EANN, sGNN) to obtain the total heat flux for each NVE trajectory.
4. **Compute Autocorrelation** : Run `calc_corr.py` to generate the HFACF.
5. **Compute Thermal Conductivity** : Run `integrate.py` to obtain the final *κ* value and its statistical uncertainty.

## Usage

The project consists of three main computation modules and two analysis scripts:

### Computation Module

1. **`compt_dp_J.py`**: Calculates kinetic term combining nonbonding heat flux using DeepMD potential

   - Input: `pos.npy`, `vel.npy`, `box.npy, atype.npy (dp element type file)`, `graph.pb (dp model)`
   - Output: J_dp.txt
   - Usage:
2. **`compute_eann_J.py`**: Calculates heat flux using EANN (Embedded Atom Neural Network) potential from DMFF

   - Input: initial PDB file (for topology), trajectory (xyz format), velocities (npy format), EANN parameters
   - Output: `J_eann.txt`
   - Usage:
3. **`compute_gnn_J.py`**: Calculates heat flux using sGNN (Subgraph Neural Network) potential from DMFF

   - Input: initial PDB file (for topology), trajectory (xyz format), velocities (npy format), sGNN parameters
   - Output: `J_gnn.txt`
   - Usage:

### Analysis Module

- **`calc_corr.py`**: Computes heat flux autocorrelation function from multiple trajectories

  - Input: Directory with J data files, output filename
  - Output: Autocorrelation data in `corr.txt` and specified output file
- **`integrate.py`**: Integrates autocorrelation function to obtain thermal conductivity

  - Input: Correlation data file
  - Output: Thermal conductivity value and `hcf.txt`
- **`tc.sh`**: Automated bash script for TCF calculation and integration. (This script is not necessary)

  Usage: bash tc.sh xxx.pdb (the pdb is the topology file for reading box volume)

## **Attention**

- **Please be careful of the units used in this project.**

    Input Unit: Positions: Å;  Velocity: m/s

    Output Unit: *(LAMMPS Real )*  Energy: Kcal/mol; Distance: Å; Time: fs;   Heat Flux: Kcal/mol·Å/fs

    And the unit of final thermal conductivity is W/m/K.

* **This project uses JIT acceleration.**

  The first run of each code is slow, due to the JIT compilation.
* **Notice the sampling time interval of NVE trajectory!**

  This dertermines the correlation length of TCF.
* **Environment**

  If you want to use GPU acceleration, you need to  match versions of Cuda, GPU, DMFF, JAX and DeepMD!

## Dependencies

- Python 3.10+
- NumPy, SciPy, MDAnalysis
- Cuda (If GPU is used)
- For `compt_dp_J.py`: deepmd-kit
- For `compute_eann_J.py`: jax, openmm, dmff
- For `compute_gnn_J.py`: jax, openmm, dmff

## References

1. Tu, C.; Li, X.; Chen, J.; Sun, B.; Yu, K. Enhancing Thermal Conductivity Computation of Polymers via Machine Learning Techniques. *J. Phys. Chem. B* . 2025 129, 33, 8593–8602. DOI: [10.1021/acs.jpcb.5c03656](https://doi.org/10.1021/acs.jpcb.5c03656).
2. Chen J,; Yu, K. PhyNEO: A Neural-Network-Enhanced Physics-Driven Force Field Development Workflow for Bulk Organic Molecule and Polymer Simulations. *J. Chem. Theory Comput.* 2024, 20, 1, 253–265. DOI:[10.1021/acs.jctc.3c01045](https://doi.org/10.1021/acs.jctc.3c01045 "DOI URL").
3. Wang, X.; Li, J.; Yang, L.; Chen, F.; Wang, J.; Chang, J.; Chen, J.; Feng, W.; Zhang, L.; Yu, K. DMFF: An Open-Source Automatic Differentiable Platform for Molecular Force Field Development and Molecular Dynamics Simulation. *J. Chem. Theory Comput.* 2023, 19, 17, 5897–5909. DOI: [https://doi.org/10.1021/acs.jctc.2c01297](https://doi.org/10.1021/acs.jctc.2c01297 "DOI URL").
