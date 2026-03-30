#!/usr/bin/env python3
import os
import sys
import re
import numpy as np
import jax
import jax.numpy as jnp
import MDAnalysis as mda
from jax import jit, vmap, value_and_grad, grad
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *
import dmff
from dmff.api import Hamiltonian
from dmff.common import nblist
from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales
from dmff.admp.pairwise import distribute_scalar, distribute_v3
from dmff.admp.spatial import pbc_shift
from functools import partial
import jax.nn.initializers
#from dmff.eann.eann import EANNForce, get_atomic_energies
from eann import EANNForce, get_gto, get_atomic_energies
import pickle
#from jax.config import config
#config.update('jax_enable_x64', True)
#from jax_md import partition, space


class EANNForce2(EANNForce):
    def __init__(self, n_elem, elem_indices, n_gto, rc, nipsin=2, beta=0.2, sizes=(64, 64), seed=12345):
        super().__init__(n_elem, elem_indices, n_gto, rc, nipsin=2, beta=0.2, sizes=(64, 64), seed=12345)
        self.get_atomic_energy = self.generate_atomic_energy()
    def generate_atomic_energy(self):
        @jit_condition(static_argnums=())
        def get_energy2(positions, box, pairs, params):
            pairs = pairs[:,:2]
            pairs = regularize_pairs(pairs)
            buffer_scales = pair_buffer_scales(pairs)

            # get distances
            box_inv = jnp.linalg.inv(box)
            ri = distribute_v3(positions, pairs[:, 0])
            rj = distribute_v3(positions, pairs[:, 1])
            dr = rj - ri
            dr = pbc_shift(dr, box, box_inv)

            dr_norm = jnp.linalg.norm(dr, axis=1)
            buffer_scales2 = jnp.piecewise(buffer_scales, (dr_norm <= 4, dr_norm > 4),
                            (lambda x: jnp.array(1), lambda x: jnp.array(0)))
            buffer_scales = buffer_scales2 * buffer_scales

            self.rs = params['rs']
            self.inta = params['inta']

            radial_i, radial_j = get_gto(jnp.arange(len(dr_norm)), dr_norm, pairs, self.rc, self.rs, self.inta, self.elem_indices)
            radial = jnp.concatenate((radial_i,radial_j), axis=0)
            orb_coeff = params['c'][self.elem_indices,:] # (48,16)

            features = self.get_features(radial, dr, pairs, buffer_scales, orb_coeff)
            atomic_energies = get_atomic_energies(features, self.elem_indices, params)
            return atomic_energies + params['initpot'][0]

        return get_energy2


if __name__ == '__main__':


    pdb  = sys.argv[1]
    pxyz = sys.argv[2]  # NVE trajectory
    velfile = sys.argv[3]  # NVE velocities
    psr3 = sys.argv[4]  # eann model pickle
    xml  = sys.argv[5]

    m_per_s_2_A_per_fs = 0.00001
    kj_2_kcal = 0.2390032
    velocities = np.load(velfile)
    velocities = velocities * m_per_s_2_A_per_fs  ## unit is angstrom/fs

    # Get element indices
    mol = PDBFile(pdb)
    box = jnp.array(mol.topology.getPeriodicBoxVectors().value_in_unit(angstrom))
    box_inv = jnp.linalg.inv(box)
    atomtype = ['H','C','O']
    masstype = [1.00784, 12.0107, 15.999]
    n_elem = len(atomtype)
    masses, species = [], []
    # Loop over all atoms in the topology
    for atom in mol.topology.atoms():
        element = atom.element.symbol
        #mass = atom.element.mass
        number = atomtype.index(element)
        species.append(number)
        masses.append(masstype[number])
    elem_indices = jnp.array(species)
    masses = jnp.array(masses)

    def set_box(ts):
        ts.dimensions = [box[0][0], box[1][1], box[2][2], 90, 90, 90]
        return ts
    u = mda.Universe(pdb, pxyz, transformations=[set_box])

    # Read eann params pickle
    with open(psr3, 'rb') as ifile:
        params_eann = pickle.load(ifile)

    # EANN class to case
    rc = 4
    pot_eann = EANNForce(n_elem, elem_indices, n_gto=16, rc=rc)
    pot_eann2 = EANNForce2(n_elem, elem_indices, n_gto=16, rc=rc)


    def individual_energy(pos, box, pairs, params, index):          # Function for individual energies
        Ea = pot_eann2.get_atomic_energy(pos, box, pairs, params)
        k = Ea[index][0][0]
        return k

    potential_a = jit(grad(individual_energy, argnums=0))

    # Final process
    Jlist, J_convect, J_conduct = [], [], []
    ae_list, vel_part = [], []

    H = Hamiltonian(xml)
    pots = H.createPotential(mol.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=PME, ethresh=1e-4, step_pol=5)
    nbl = nblist.NeighborList(box, rc, cov_map=pots.meta["cov_map"])

    for fr, ts in enumerate(u.trajectory):
        pos = jnp.array(ts.positions)
        vel = velocities[fr]

        nbl.allocate(pos, box)
        pairs = nbl.pairs

        # E_ml = pot_eann.get_energy(pos, box, pairs, params)  for debug
        E_atoms = pot_eann2.get_atomic_energy(pos, box, pairs, params_eann) # individual potential
        # E_ml2 = jnp.sum(E_atoms)  for debug
        E_a2 = E_atoms[:,0,0]
        Ua = jnp.array(E_a2) * kj_2_kcal

        natoms = len(Ua)
        # Force
        for i in range(natoms):
            force_a = potential_a(pos, box, pairs, params_eann, i) * kj_2_kcal
            if i == 0:
                fab = force_a
            else:
                fab = jnp.vstack((fab, force_a))


        # Calculate heat flux, first item
        Ea = Ua
        vel = jnp.array(velocities[fr])
        natoms = len(Ea)
        ae_list.append(Ea)

        # Second item
        def pbc_dr(dr_ab, box):
            box_inv = jnp.linalg.inv(box)
            return pbc_shift(dr_ab, box, box_inv)

        # Compute relative distances with PBC
        ra = jnp.repeat(pos, natoms, axis=0)
        rb = jnp.tile(pos, (natoms, 1))
        dr = rb - ra
        rab = vmap(pbc_dr, in_axes=(0, None), out_axes=0)(dr, box)

        # Compute velocity products
        vb = jnp.tile(vel, (natoms, 1))
        fabv = vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(fab, vb)
        Ev2 = jnp.dot(fabv, rab)
        J_conduct.append(Ev2)

        vel_fr = velocities[fr]
        vel_part.append(vel_fr)

    # remove energy bias
    ae_list = np.array(ae_list)
    ae_list = ae_list - ae_list.mean(0)
    J_conduct = np.array(J_conduct)
    vel_part  = np.array(vel_part)

    for fi in range(J_conduct.shape[0]):
        vel = vel_part[fi]
        J_convect = np.dot(ae_list[fi], vel)
        J = J_convect + J_conduct[fi]
        Jlist.append(J)


    Jlist = np.array(Jlist)
    print(Jlist.shape)
    with open('J_eann.txt', 'a') as f1:
        for J in Jlist:
            print(*J, file=f1)





