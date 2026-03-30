# EANN Heat Flux Calculator (`compute_eann_J.py`)

This script calculates the heat flux using an Embedded Atom Neural Network (EANN) potential. It uses JAX.grad for automatic differentiation and is designed for potentials where atomic energy can be expressed as a sum over atoms, E = ∑_i e_i.

## Command-Line Arguments

**text**

```
python compute_eann_J.py <pdb> <pxyz> <velfile> <psr3> <xml>
```

| Argument | Description                                                                                                      |
| -------- | ---------------------------------------------------------------------------------------------------------------- |
| pdb      | PDB file with system topology and initial coordinates.                                                           |
| pxyz     | Trajectory file (e.g., XYZ or any format supported by MDAnalysis).                                               |
| velfile  | Numpy file with atomic velocities, shape `(n_frames, n_atoms, 3)`. Units: m/s (converted to Å/fs internally). |
| psr3     | Pickle file containing the trained EANN model parameters.                                                        |
| xml      | XML file defining the non-bonded interactions (e.g., for neighbor list generation with OpenMM).                  |

## Parameter Settings

| Parameter          | Value     | Unit | Description                                 |
| ------------------ | --------- | ---- | ------------------------------------------- |
| rc                 | 4.0       | Å   | Cutoff radius for neighbor list generation. |
| kj_2_kcal          | 0.2390032 | -    | Conversion from kJ/mol to kcal/mol.         |
| m_per_s_2_A_per_fs | 1e-5      | -    | Velocity conversion.                        |

## Heat Flux Calculation Methodology

### 1. Initialization and Setup

* The atomic types (H, C, O) are read from the PDB file.
* The periodic box vectors are extracted from the PDB.
* The EANN model (`EANNForce2`) is initialized. A key feature of `EANNForce2` is its `get_atomic_energy` method, which returns a per-atom energy array {e_iᵖᵒᵗ}.

### 2. Loop over Frames

For each frame:

* **Get Positions and Velocities** : Read coordinates from the trajectory and velocities from the `.npy` file.
* **Generate Neighbor List** : Use OpenMM to generate a neighbor list for the EANN model.
* **Calculate Atomic Potential Energy** : `E_a = pot_eann2.get_atomic_energy(...)`. This returns an array of per-atom energies in kJ/mol, which is then converted to kcal/mol.
* **Calculate Atomic Forces via Automatic Differentiation (AD)** :
* A function `individual_energy` is defined to return the energy of a single atom (index i).
* `grad` is applied to this function with respect to positions, yielding the force on atom i.
* This is done for all atoms in a loop (or via `vmap` for efficiency).
* **Compute Heat Flux Components** :
* **Convective Term** : J_conv = ∑_i e_iᵗᵒᵗᵃˡ v_i, where e_iᵗᵒᵗᵃˡ = e_iᵖᵒᵗ (kinetic energy is not included here, as per the paper's decomposition).
* **Conductive Term** : J_cond = ∑_i (F_i · v_i) r_i. Here, r_i is the position of atom i, but the paper uses a more rigorous form involving pair interactions. The script approximates it by summing over all atoms, which is a common practice but may not be exact for all potentials. 
* **Total Heat Flux** : J = J_conv + J_cond.

### 3. Output

The script appends the heat flux vector for each frame to the file `J_eann.txt`.

## Important Notes

* The EANN potential in this script is  local, corresponding to **short-range corrections**.
* The `xml` file is required to set up the OpenMM context for neighbor list generation. It should define a dummy force field that does not contribute to the energy.
* For production runs, consider optimizing the force calculation loop. The current loop over atoms for gradients may be slow for large systems. Batching or `vmap` can be used to accelerate this.

## Usage Example

**text**

```
python compute_eann_J.py system.pdb traj.xyz velocities.npy eann_model.pickle polymer.xml > log.txt
```

After running, the file `J_eann.txt` will be generated, which can be directly used as input for `calc_corr.py`.
