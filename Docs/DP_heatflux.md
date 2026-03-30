# DeepMD Heat Flux Calculator (`compute_dp_J.py`)

This script computes the nonbonding heat flux vector for a given trajectory using a trained DeepMD potential. It is designed to be used with potentials that can be loaded by the DeepMD-kit.

This code also contains the knetic term of  heat flux.

## Key Input Files

* `pos.npy`: A NumPy array of atomic positions, shape `(n_frames, n_atoms, 3)`.  unit: Å
* `vel.npy`: A NumPy array of atomic velocities, shape `(n_frames, n_atoms, 3)`.  unit: m/s
* `box.npy`: A NumPy array of box vectors, shape `(n_frames, 9)`.  unit: Å
* `atype.npy`: A NumPy array of atomic types, shape `(n_atoms,)`.
* `graph.pb`: The protobuf file containing the trained DeepMD model.

## Parameter Settings

The script contains hard-coded parameters that must match your simulation:

| Parameter            | Value (Example) | Unit | Description                               |
| -------------------- | --------------- | ---- | ----------------------------------------- |
| dt                   | 0.5             | fs   | Time step used in the MD simulation       |
| Nc                   | 1               | -    | Frame stride (process every Nc-th frame)  |
| eV2kCal              | 23.06031        | -    | Conversion factor from eV to kcal/mol     |
| g_A2_per_fs2_to_kcal | 2390.1          | -    | Conversion for kinetic energy to kcal/mol |
| m_per_s_2_A_per_fs   | 1e-5            | -    | Velocity conversion factor (m/s → Å/fs) |

## Heat Flux Calculation Methodology

1. **Load Inputs** : Load positions, velocities, and atomic types.
2. **Initialize DeepMD** : Load the `graph.pb` model to evaluate per-atom energies and virials.
3. **Loop over Frames** (with stride `Nc`):
   * **Get Potential Energy** : `ae` from the DeepMD model, in eV.
   * **Calculate Kinetic Energy** : `ake = g_A2_per_fs2_to_kcal * mass * (vel²) / 2`.
   * **Total Per-Atom Energy** : `e_i = ae + ake`, converted to kcal/mol.
   * **Remove Energy Bias** : Subtract the mean energy across all processed frames for stability.
   * **Get Atomic Virial** : `av` from the DeepMD model, in eV, reshaped to `(n_atoms, 3, 3)`.
   * **Compute Convective Term** : `J_conv = e_i * vel`.
   * **Compute Conductive Term** : `J_cond = (vel.reshape(1, -1) @ av.reshape(-1, 3)).squeeze()`. This is a vectorized implementation of the per-atom stress contribution: J_cond = ∑_i S_i · v_i.
   * **Total Heat Flux** : `J = J_conv + J_cond`.
4. **Output** : For each frame, print the frame index and the three components of J to stdout.

## Unit Consistency

* **Input** : Positions in Å, velocities in m/s (converted to Å/fs internally).
* **Output** : Heat flux components in units of  **kcal/mol·Å/fs** . This is the raw quantity that will be scaled later in the autocorrelation and integration steps.

## Usage Example

**text**

```
# Run the calculation
python compt_J.py > J_dp.txt
```

## Important Notes

* This script uses the **total** per-atom energy (potential + kinetic) for the convective term. This is standard practice.
* The `scale` factor in `calc_corr.py` is used to convert the raw `J` data to the proper units for the GK formula. The raw output from this script is suitable as input for `calc_corr.py` without any additional processing.
* The DeepMD model must be compatible with the given atomic types and simulation box.
