# GNN Heat Flux Calculator (`compute_gnn_J.py`)

This script calculates the heat flux using a Graph Neural Network (GNN) potential, specifically the Subgraph Neural Network (sGNN) model from the DMFF package. It implements the rigorous centroid method for computing the conductive heat flux, as described in the literature (Surblys et al., *Phys. Rev. E*  **2019** , *99*, 051301).

## Command-Line Arguments

**text**

```
python compute_gnn_J.py <topo> <ifn> <psr2> <velfile>
```

| Argument | Description                                                                                                           |
| -------- | --------------------------------------------------------------------------------------------------------------------- |
| topo     | PDB file containing system topology and initial coordinates.                                                          |
| ifn      | Trajectory file (e.g., DCD, XYZ) readable by MDAnalysis.                                                              |
| psr2     | Pickle file containing the trained sGNN model parameters.                                                             |
| velfile  | Numpy file with atomic velocities, shape `(n_frames, n_atoms, 3)`. **Units: m/s** (converted to Å/fs internally). |

## Parameter Settings

| Parameter          | Value     | Unit | Description                              |
| ------------------ | --------- | ---- | ---------------------------------------- |
| N_vmap             | 800       | -    | Batch size for JAX vmap parallelization. |
| kj_2_kcal          | 0.2390032 | -    | Unit conversion.                         |
| m_per_s_2_A_per_fs | 1e-5      | -    | Velocity conversion.                     |

## Heat Flux Calculation Methodology

This script is designed for sGNN potentials, where the total energy is a sum over local subgraphs: E_total = ∑_k E_kˢᵘᵇ.

### 1. Graph and Subgraph Construction

* A topological graph is built from the system's atoms and bonds.
* Subgraphs (local atomic environments) are generated from the graph.

### 2. Loop over Frames

For each frame:

* **Get Positions and Velocities** : Coordinates from the trajectory, velocities from the `.npy` file.
* **Compute Subgraph Energies** : The sGNN model (`NewMolGNNForce.compute_subgraph_energies`) returns an array of energies for each subgraph, in kJ/mol. Then transform the unit to kcal/mol.
* **Compute Subgraph Force Derivatives** :
* A custom function `fab_calc` returns the energy of a single subgraph.
* `jax.grad` is used to compute the derivative of that energy with respect to all atomic positions.
* Using jax.`vmap`, these derivatives are computed for all subgraphs in batches, yielding a tensor `fab` of shape `(n_subgraphs, n_atoms, 3)`.
* The force on an atom from a specific subgraph is the negative of this gradient.
* **Compute Conductive Heat Flux (`get_information` method)** :
* **Centroid Method** : For each subgraph, the centroid position r₀ᵏ is calculated.
* The relative positions rₐᵏ - r₀ᵏ are computed.
* The conductive contribution from that subgraph is: J_condᵏ = ∑_{a∈k} (Fₐᵏ · vₐ) (rₐᵏ - r₀ᵏ).
* The total conductive heat flux is the sum over all subgraphs.
* **Compute Convective Heat Flux** :
* The per-atom energy is obtained by equally distributing each subgraph's energy among its constituent atoms.
* The convective term is then J_conv = ∑_i e_iᵗᵒᵗᵃˡ v_i.
* **Total Heat Flux** : J = J_conv + J_cond.

### 3. Output

* `J_conduct.txt`: Contains the conductive term for each frame.
* `J_gnn.txt`: Contains the total heat flux for each frame. （You should use this file as sGNN heat flux)

## Important Notes

* The sGNN model must be compatible with the system's **atomic types and bonding topology**.
* The centroid method is critical for ensuring that the heat flux is gauge-invariant and consistent with the GK formalism.
* The script uses `vmap` for efficient batched gradient calculations. **Adjust `N_vmap` based on your GPU memory**. If you use CPU, there is no need to adjust N_vmap, at the cost of computational speed.
