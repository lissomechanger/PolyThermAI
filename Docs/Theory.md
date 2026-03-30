# Theory for PolyThermAI

## 1. Introduction

Accurate prediction of thermal conductivity (*κ*) for amorphous polymers remains a significant challenge in computational materials science. Traditional approaches face fundamental limitations: ab initio methods such as density functional theory combined with the Boltzmann transport equation (DFT-BTE) are prohibitively expensive for disordered systems, while classical force fields used in molecular dynamics (MD) simulations lack the accuracy needed for reliable thermal transport predictions. Machine learning potentials (MLPs) offer a promising alternative but face their own challenges, particularly in capturing long-range interactions and providing accurate heat flux calculations.

This theoretical framework presents a comprehensive methodology for calculating *κ* using the Green-Kubo formalism combined with machine learning (ML)-facilitated heat flux calculations. The approach, termed PolyThermAI, addresses the limitations of existing methods by separating bonding and non-bonding interactions and using localized machine learning models to approximate complex long-range potentials for heat flux calculations.

## 2. The Green-Kubo Formalism

The Green-Kubo relation provides a rigorous statistical mechanical framework for calculating transport coefficients from equilibrium molecular dynamics simulations. For thermal conductivity, the relation expresses κ as the time integral of the heat flux autocorrelation function.

For an isotropic system, the thermal conductivity κ is given by:

κ = (V / (3 k_B T²)) ∫₀^∞ ⟨J(0)·J(t)⟩ dt

where:

* V is the system volume
* k_B is the Boltzmann constant (1.3806504 × 10⁻²³ J/K)
* T is the absolute temperature in Kelvin
* J(t) is the instantaneous heat flux vector at time t
* ⟨...⟩ denotes an ensemble average, typically approximated by a time average over a long equilibrium trajectory

The factor 1/3 arises from averaging over the three spatial dimensions (x, y, z) under the assumption of isotropy. For anisotropic systems, one would compute the full thermal conductivity tensor κ_αβ = (V / (k_B T²)) ∫₀^∞ ⟨J_α(0) J_β(t)⟩ dt.

In practice, the integral is replaced by a discrete sum over simulation time steps:

κ = (V / (3 k_B T²)) ∑ *{i=0}^{N-1} (C_J(t_i) + C_J(t* {i+1}))/2 · Δt

where C_J(t) = ⟨J(0)·J(t)⟩ is the heat flux autocorrelation function (HFACF) and Δt is the time step between correlation points.

## 3. Heat Flux Decomposition

The accurate calculation of the instantaneous heat flux J(t) is the most critical step in the Green-Kubo workflow. For a system of N atoms, the heat flux can be derived from the energy conservation equation and is typically decomposed into kinetic and potential contributions.

### 3.1 General Form

A convenient and widely used form of the heat flux, implemented in popular MD codes like LAMMPS, is:

J = ∑_i e_i v_i - ∑_i S_i · v_i

where:

* e_i is the total energy (kinetic + potential) of atom i
* v_i is the velocity vector of atom i
* S_i is the per-atom stress (virial) tensor

The first term represents the convection of energy by atomic motion, while the second term accounts for energy transfer through interatomic forces.

### 3.2 Per-Atom Energy Decomposition

The per-atom total energy e_i can be further decomposed as:

e_i = e_iᵏⁱⁿ + e_iᵖᵒᵗ

where:

* e_iᵏⁱⁿ = ½ m_i |v_i|² is the kinetic energy
* e_iᵖᵒᵗ is the potential energy contribution assigned to atom i

The assignment of potential energy to individual atoms is not unique and depends on the functional form of the interatomic potential. For simple pairwise potentials, e_iᵖᵒᵗ = ½ ∑_{j≠i} U(r_ij), providing a natural decomposition. For many-body and machine learning potentials, the decomposition is often inherent in the model architecture through a sum-over-atoms formalism.

### 3.3 Per-Atom Stress Tensor

The per-atom stress tensor S_i is defined based on the virial theorem:

S_i = ½ ∑_{j≠i} r_ij ⊗ F_ij

where:

* r_ij = r_i - r_j is the relative position vector between atoms i and j
* F_ij is the force on atom i due to atom j
* ⊗ denotes the outer product (dyadic product)

For many-body potentials, the stress contribution may involve higher-order terms, but the fundamental principle remains: S_i represents the contribution of atom i to the total system stress.

## 4. Challenges for Complex Potentials

While the heat flux decomposition is straightforward for simple classical force fields with pairwise interactions, it becomes problematic for advanced potentials that are essential for accurate polymer simulations.

### 4.1 Limitations of Classical Force Fields

Classical force fields such as OPLS-AA and PCFF, despite their widespread use, suffer from fundamental limitations for thermal transport calculations:

* **Harmonic Approximations** : The harmonic bond stretching and angle bending terms in classical force fields lack the anharmonicity that is essential for phonon scattering mechanisms. This leads to overestimation of thermal conductivity, typically by 50% or more.
* **Limited Transferability** : Parameters fitted to reproduce equilibrium properties (densities, heats of vaporization) do not necessarily capture the potential energy surface accurately for non-equilibrium properties like thermal transport.
* **Missing Many-Body Effects** : Classical force fields approximate many-body interactions through pairwise additive terms, neglecting essential electronic polarization and many-body dispersion effects.

### 4.2 Complexity of Advanced Potentials

Advanced potentials that achieve higher accuracy introduce new challenges for heat flux calculation:

* **Machine Learning Potentials** : While MLPs like DeepMD, EANN, and sGNN provide per-atom energies naturally through their sum-over-atoms architecture, the derivation of atomic stress tensors can be non-trivial.
* **Long-Range Interactions** : Potentials with explicit long-range electrostatics (Ewald sums, particle mesh Ewald) and polarization (induced dipoles, charge fluctuations) require global self-consistent solutions, and their analytical heat flux expressions are often unknown.
* **Many-Body Nature** : The non-local nature of many-body interactions complicates the spatial decomposition of energy and stress required for the heat flux formula.

## 5. The PolyThermAI Approach: ML-Facilitated Heat Flux Calculation

To overcome these challenges, the PolyThermA workflow introduces a novel approach that separates the total potential energy into components based on physical interactions and uses localized machine learning models to approximate complex parts for heat flux calculation.

### 5.1 Energy Decomposition Strategy

The total potential energy of a hybrid ML potential like PhyNEO is decomposed as:

U_phyNEO = U_nb + U_sGNN_bond

where:

* U_sGNN_bond is the bonding energy described by a subgraph neural network (sGNN) model
* U_nb is the non-bonding energy, further decomposed into physically motivated components:

U_nb = U_lr_nb + U_sr_nb + U_sr,ML_nb

with:

* U_lr_nb = U_es_lr + U_pol_lr + U_disp_lr (long-range electrostatics, polarization, dispersion via PME)
* U_sr_nb (short-range repulsion and exchange from physics-based models)
* U_sr,ML_nb (short-range non-bonding correction from an embedded atom neural network, EANN)

This decomposition serves two purposes: it improves data efficiency by incorporating physical knowledge, and it enables a practical strategy for heat flux calculation.

### 5.2 Bonding Contribution: Analytical Calculation

The bonding energy U_sGNN_bond is described by a localized sGNN model that naturally provides per-subgraph energies. For such localized models, the heat flux can be calculated analytically using the centroid method developed by Surblys et al. (Phys. Rev. E 2019, 99, 051301).

For each subgraph k with energy E_kˢᵘᵇ and containing N_k atoms:

* The centroid position is defined as r₀ᵏ = (1/N_k) ∑_{a∈k} r_aᵏ
* The relative positions are r_aᵏ - r₀ᵏ
* The force on atom a from subgraph k is F_aᵏ = -∂E_kˢᵘᵇ/∂r_aᵏ

The bonding contribution to the heat flux is then:

J_bond = ∑_a e_a_bond v_a + ∑ *k ∑* {a∈k} (F_aᵏ · v_a) (r_aᵏ - r₀ᵏ)

where e_a_bond is obtained by equally distributing the subgraph energy among its constituent atoms: e_a_bond = E_kˢᵘᵇ / N_k for all a ∈ k.

### 5.3 Non-Bonding Contribution: ML Approximation

The non-bonding energy U_nb presents the main challenge because it includes long-range, non-local interactions that are difficult to express analytically for heat flux. The PolyThermAI solution is to:

1. Generate a trajectory using the full, accurate PhyNEO potential
2. Train a localized Deep Potential (DP) model to fit U_nb and the corresponding forces
3. Use the trained DP model to compute per-atom energies e_aⁿᵇ and per-atom virials S_aⁿᵇ
4. Apply the heat flux formula to these approximated atomic quantities

The trained DP model automatically provides per-atom energies and forces in a local, sum-over-atoms form. The heat flux for the non-bonding part is then:

J_nb = ∑_a e_aⁿᵇ v_a - ∑_a S_aⁿᵇ · v_a

### 5.4 Validation of the ML Approximation

The validity of this approximation is demonstrated by applying it to a system where the exact heat flux is known. Using the OPLS-AA force field (where analytical heat flux formulas exist) for trajectory generation, a DP model is trained to fit the OPLS non-bonding energy (Lennard-Jones and Coulomb). The thermal conductivity calculated using the ML-approximated heat flux agrees with the exact OPLS result within statistical error, validating the approach.

### 5.5 Total Heat Flux

The total heat flux is obtained by summing contributions from all components:

J_total = J_kinetic + J_bond + J_sr,ML_nb + J_nb

where:

* J_kinetic = ∑_a e_aᵏⁱⁿ v_a
* J_bond is computed analytically from sGNN
* J_sr,ML_nb is computed from the EANN model (which naturally provides per-atom energies)
* J_nb is computed from the DP model trained on the physics-based non-bonding components

## 6. Numerical Implementation Considerations

### 6.1 Periodic Boundary Conditions

Periodic boundary conditions (PBCs) complicate the definition of relative positions r_ij and the assignment of energy to atoms. In the centroid method, this is handled by:

* Using the subgraph centroid as a reference point
* Shifting all atomic positions in the subgraph so that they are within the primary cell relative to the centroid
* This ensures that all distances are computed correctly under PBC

### 6.2 Time Correlation Function Calculation

The heat flux autocorrelation function C_J(t) is computed as:

C_J(t) = (1/3) ⟨J_x(0)J_x(t) + J_y(0)J_y(t) + J_z(0)J_z(t)⟩

For a discrete trajectory with N_f frames at times t_i = i·Δt_corr, the correlation is estimated using:

C_J(mΔt_corr) = (1/(N_f - m)) ∑ *{i=0}^{N_f-m-1} (1/3) ∑* {α=x,y,z} J_α(iΔt_corr) J_α((i+m)Δt_corr)

This is efficiently computed using the FFT-based correlation method implemented in NumPy.

### 6.3 Integration and Convergence

The thermal conductivity is obtained by integrating the HFACF. The integral typically converges to a plateau at long times. The final value is taken as the average over the plateau region (e.g., the last 10% of the integration time). The standard error of the mean over multiple independent trajectories provides the statistical uncertainty.

### 6.4 Unit Conversions

Careful attention must be paid to unit conversions throughout the workflow:

* Positions: typically in Ångströms (Å)
* Velocities: often in m/s from MD outputs, converted to Å/fs (1 m/s = 10⁻⁵ Å/fs)
* Energies: DeepMD outputs in eV, converted to kcal/mol (1 eV = 23.06031 kcal/mol)
* Heat flux: final units are kcal/mol·Å/fs for raw output
* Thermal conductivity: final units are W/(m·K)

The scaling factor in the autocorrelation script converts the raw HFACF from (kcal/mol·Å/fs)² to the proper units for the GK integral: scale = (kCal2J)² / (fs2s·A2m·k_B·T²·V) where kCal2J converts kcal to J, fs2s converts fs to s, and A2m converts Å to m.

## 7. Conclusions

The theoretical framework presented here establishes a rigorous methodology for calculating thermal conductivity in amorphous polymers using the Green-Kubo formalism with machine learning-facilitated heat flux calculations. Key innovations include:

* Systematic decomposition of potential energy into physically distinct components
* Analytical heat flux calculation for localized bonding interactions
* ML-based approximation for complex non-bonding interactions
* Validation of the approximation against exact classical force field results

This approach enables quantitative predictions of bulk polymer thermal conductivity starting from only small cluster quantum data, opening new possibilities for computational materials design in thermal management applications.
