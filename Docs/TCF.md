# Heat Flux Autocorrelation Calculator (`calc_corr.py`)

This script calculates the normalized heat flux autocorrelation function (HFACF) from a set of heat flux trajectories. It also applies the necessary scaling to convert the raw HFACF to the form required for the Green-Kubo integral.

## Command-Line Arguments

**text**

```
python calc_corr.py <dest> <corrfile> <pdbfile>
```

| Argument | Description                                                                       |
| -------- | --------------------------------------------------------------------------------- |
| dest     | Path to a directory containing the total heat flux trajectory files.              |
| corrfile | Output file name for the raw, un-scaled autocorrelation data for each trajectory. |
| pdbfile  | Topology file for reading box volume.                                             |

## Input File Format

Each file in `<dest>` is expected to contain a heat flux trajectory in the following format:

time Jx Jy Jz

Where:

* time is in  **picoseconds** .
* Jx, Jy, Jz are the three components of the heat flux vector in units of  **kcal/mol·Å/fs** .

## Parameter Settings

| Parameter   | Value         | Unit | Description                                                     |
| ----------- | ------------- | ---- | --------------------------------------------------------------- |
| T           | 298           | K    | Temperature of the simulation.                                  |
| kB          | 1.3806504e-23 | J/K  | Boltzmann constant.                                             |
| volume      | L_x L_y L_z   | Å³ | Simulation box volume (must match the simulation).              |
| corr_length | 1000          | -    | Number of points for the correlation function.                  |
| dt          | 0.5           | fs   | MD time step.                                                   |
| Ns          | 1             | -    | Number of steps between saved frames.                           |
| Nc          | 10            | -    | Stride for calculating the correlation (not fully implemented). |

**Please revise the temperature, correlation length, time settings (Nc, Ns, dt) and other parameters according to your system and needs !!!**

## Calculation Steps

1. **Data Reading** : The script reads all files in `<dest>`.
2. **Autocorrelation Calculation** :

* For each file, it calculates the autocorrelation of the Jx, Jy, and Jz components separately using `np.correlate(x, x, mode='full')`.
* The result is normalized by the number of overlapping points at each lag time.
* The three components are averaged to give the isotropic correlation: C_Jʳᵃʷ(t) = ⅓ (C_Jx + C_Jy + C_Jz).

1. **Scaling** :

* The raw HFACF is in units of (kcal/mol·Å/fs)².
* The scale factor is defined as: scale = (kcal/mol)² · fs⁻¹ · Å⁻¹ / (k_B T² V).
* After multiplication by `scale`, the HFACF has units of W²·m⁻¹·K⁻², which is the proper form for integration.

1. **Output** :

* **`corrfile`** : Writes the time and the raw (unscaled) correlation for each trajectory. This is useful for debugging.
* **`corr.txt`** : Writes the time and the scaled, averaged correlation function. This file is the direct input for `integrate.py`.

## Important Notes

* The `volume` and `T` must be accurate. Check your simulation parameters.
* The `dt`, `Ns`, and `corr_length` must be consistent with your trajectory. The time points `ts` are calculated as `ts = np.arange(0, corr_length) * Ns * Nc * dt`. Ensure `Ns * Nc * dt` equals the time step of your output trajectory.

## Usage Example

```
# After generating J_total.txt for each NVE ensemble
mkdir j_data
cp J_total.txt j_data/

# Run autocorrelation
python calc_corr.py j_data allcorr.data topology.pdb

# This will create allcorr.data (raw) and corr.txt (scaled and averaged)
```
