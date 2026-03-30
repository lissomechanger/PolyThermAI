#!/usr/bin/env python3
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *
import dmff
from dmff.api import Hamiltonian
from dmff.common import nblist
from dmff.utils import jit_condition
from dmff.sgnn.graph import TopGraph, from_pdb, MAX_VALENCE
from dmff.sgnn.gnn import MolGNNForce
from dmff.admp.spatial import pbc_shift
from functools import partial
import pickle
import gc
import MDAnalysis as mda

# Global parameter settings
ATOMIC_MASS = {'H': 1.00784, 'C': 12.0107, 'O': 15.9994}

# Unit conversion: finally use the lammps real unit: energy (Kcal/mol), distance (angstrom), time (fs).
m_per_s_2_A_per_fs = 0.00001
kj_2_kcal = 0.2390032

# JAX vmap parellel setting, please reset it according to the hardware property (e.g., GPU memory).
N_vmap = 800

# Inherit the GNN class from sgnn module in DMFF code, and calculate the array of subgraph energies instead of the total sgnn energy.
class NewMolGNNForce(MolGNNForce):
    def __init__(self, G, n_layers=(3, 2), sizes=[(40, 20, 20), (20, 10)], nn=1, sigma=162.13039087945623, mu=117.41975505778706, seed=12345):
        super().__init__(G, n_layers, sizes, nn, sigma, mu, seed)

    def compute_subgraph_energies(self, positions, box, params):
        features = self.G.calc_subgraph_features(positions, box)
        @jit_condition(static_argnums=())
        @partial(vmap, in_axes=(0, None), out_axes=(0))
        @partial(vmap, in_axes=(0, None), out_axes=(0))
        def fc0(f_in, params):
            f = f_in
            for i in range(self.n_layers[0]):
                f = jnp.tanh(params['fc0.weight'][i].dot(f) + params['fc0.bias'][i])
            return f

        @jit_condition(static_argnums=())
        @partial(vmap, in_axes=(0, None), out_axes=(0))
        def fc1(f_in, params):
            f = f_in
            for i in range(self.n_layers[1]):
                f = jnp.tanh(params['fc1.weight'][i].dot(f) + params['fc1.bias'][i])
            return f

        @jit_condition(static_argnums=())
        @partial(vmap, in_axes=(0, None), out_axes=(0))
        def fc_final(f_in, params):
            return params['fc_final.weight'].dot(f_in) + params['fc_final.bias']

        @partial(vmap, in_axes=(0, 0, None, None), out_axes=(0))
        def message_pass(f_in, nb_connect, w, nn):
            if nn == 0:
                return f_in[0]
            elif nn == 1:
                nb_connect0 = nb_connect[0:MAX_VALENCE - 1]
                nb_connect1 = nb_connect[MAX_VALENCE - 1:2 * (MAX_VALENCE - 1)]
                nb0 = jnp.sum(nb_connect0)
                nb1 = jnp.sum(nb_connect1)
                f = f_in[0] * (1 - jnp.heaviside(nb0, 0)*w - jnp.heaviside(nb1, 0)*w) + \
                    w * nb_connect0.dot(f_in[1:MAX_VALENCE, :]) / jnp.piecewise(nb0, [nb0<1e-5, nb0>=1e-5], [lambda x: jnp.array(1e-5), lambda x: x]) + \
                    w * nb_connect1.dot(f_in[MAX_VALENCE:2*MAX_VALENCE-1, :])/ jnp.piecewise(nb1, [nb1<1e-5, nb1>=1e-5], [lambda x: jnp.array(1e-5), lambda x: x])
                return f

        features = fc0(features, params)
        features = message_pass(features, self.G.nb_connect, params['w'], self.G.nn)
        features = fc1(features, params)
        energies = fc_final(features, params).reshape(-1)
        subgraph_energies = jnp.multiply(self.G.weights, energies) * self.sigma * kj_2_kcal
        return subgraph_energies

    # Loop over subgraphs and obtain the key terms in the SGNN heat flux calculations
    def get_information(self, positions, velocities, box, box_inv, forces, U_subgraphs):
        Jpot = jnp.zeros(3)
        U_atoms = jnp.zeros(positions.shape[0])
        index = 0
        for ig, g in enumerate(self.G.subgraphs): # Loop over subgraphs
            for ip in range(g.n_sym_perm): # Loop over different permutations of the same subgraph.
                canonical_order = g.canonical_orders[ip]
                parent_indices = g.map_sub2parent[canonical_order]
                valid_indices = parent_indices[parent_indices != -1] # Get valid atomic indices in the subgraph
                valid_positions  = positions[valid_indices]
                shift_positions = valid_positions - valid_positions[0] # Shift positions in the subgraph to address PBC problem
                subgraph_positions = vmap(pbc_shift, in_axes=(0, None, None))(shift_positions, box, box_inv) # PBC shifting in the local cluster
                centroid_position = jnp.mean(subgraph_positions, axis=0) # Centroid method proposed in the previous paper
                rac = subgraph_positions - centroid_position # Relative atomic coordinates

                subgraph_velocities = velocities[valid_indices]
                subgraph_forces = forces[index][valid_indices]
                fav = jnp.einsum('ij,ij->i', subgraph_forces, subgraph_velocities)
                Jpot += jnp.einsum('i,ij->j', fav, rac)

                divisor = max(len(valid_indices), 1)
                U_atoms = U_atoms.at[valid_indices].add(U_subgraphs[index] / divisor) # Equally allocate the subgraph energy onto the atoms.
                index += 1
        return Jpot, U_atoms

    # Define the grad function to derive the derivative of each subgraph energy relative to individual atoms in the corresponding subgraph.
    def fab_calc(self, pos, box, params, index):
        E_sgnn = self.compute_subgraph_energies(pos, box, params)
        return E_sgnn[index]


if __name__ == '__main__':

    topo = sys.argv[1]    # Initial PDB file of NVE simulations
    ifn  = sys.argv[2]    # NVE trajectory file
    psr2 = sys.argv[3]    # Sgnn parameter file
    velfile = sys.argv[4] # Velocity npy file, shape of (nframes, natoms, 3)

    # Directly read atomic coordinates and topology from the universe
    velocities = np.load(velfile) * m_per_s_2_A_per_fs # unit is angstrom/fs
    velocities = jnp.array(velocities)
    u = mda.Universe(topo, ifn)
    u0 = mda.Universe(topo)
    a, b, c, alpha, beta, gamma = u0.dimensions
    box = jnp.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
    box_inv = jnp.linalg.inv(box)
    atom_elems = u.atoms.types
    natoms = u.atoms.n_atoms
    bonds = jnp.array([[bond.atoms[0].index, bond.atoms[1].index] for bond in u.bonds])

    # Loop over frames
    ae_list = []
    prealloc_initialized = False

    # Load the sgnn graph
    G = TopGraph(atom_elems, bonds, positions=u.trajectory[0].positions, box=box)
    model = NewMolGNNForce(G, nn=1)
    model.load_params(psr2)

    # Calcultate the derivatives to positions and perform JIT acceleration
    grad_func = jit(grad(model.fab_calc, argnums=0))

    # Calculate conduction term and print in the txt file
    with open('J_conduct.txt', 'w') as f1:
        for fr, ts in enumerate(u.trajectory):
            pos = jnp.array(ts.positions)
            vel = velocities[fr]

            # Preallocate memory
            if not prealloc_initialized:
                Ua = model.compute_subgraph_energies(pos, box, model.params)
                nsubgraphs_init = len(Ua)

                global fab_prealloc
                fab_prealloc = jnp.zeros((nsubgraphs_init, natoms, 3), dtype=jnp.float32)
                prealloc_initialized = True

            # Load graph
            G.set_positions(pos)
            U_subgraphs = model.compute_subgraph_energies(pos, box, model.params)
            nsubgraphs = len(U_subgraphs)

            # Compute gradients
            def batched_vmap(grad_func, pos, box, params, nsubgraphs, batch_size):
                num_batches = -(-nsubgraphs // batch_size)  # Ceiling division to ensure all graphs are covered
                batch_indices = jnp.arange(nsubgraphs)
                batch_indices_list = jnp.array_split(batch_indices, num_batches)
                grad_a_batches = []
                for indices in batch_indices_list:
                    grad_a_batch = vmap(grad_func, in_axes=(None, None, None, 0))(pos, box, params, indices)
                    grad_a_batches.append(grad_a_batch)
                grad_a = jnp.concatenate(grad_a_batches, axis=0)
                return grad_a

            fab = -batched_vmap(grad_func, pos, box, model.params, nsubgraphs, N_vmap)
            fab = fab.reshape([nsubgraphs, natoms, 3])
            fab = fab_prealloc.at[:nsubgraphs].set(fab)

            J_conduct, Ua = model.get_information(pos, vel, box, box_inv, fab, U_subgraphs)
            ae_list.append(Ua)

            del fab
            jax.device_put(fab_prealloc)
            gc.collect()
            print(*J_conduct, file=f1)
            f1.flush()

    # remove energy bias to reduce noise in convection term, this is why this term is not directly computed in the cycles.
    ae_list = jnp.array(ae_list)
    ae_list = ae_list - ae_list.mean(0)

    # Merge convection and conduction terms
    with open('J_gnn.txt', 'w') as f2:
        J_conduct = np.genfromtxt('J_conduct.txt')
        J_convect = jnp.einsum('ij,ijk->ik', ae_list, velocities)
        J_total = J_convect + jnp.array(J_conduct)
        for J in J_total:
            print(*J, file=f2)
