import numpy as np
from numpy.linalg import eigh, norm, pinv
from scipy.linalg import lstsq  # performs better than numpy.linalg.lstsq
from tblite.ase import TBLite
from ase.geometry import get_distances_derivatives
from ase.calculators.mixing import MixedCalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution 

from ase import units
from ase.calculators.calculator import (
    BaseCalculator,
    CalculationFailed,
    Calculator,
    CalculatorSetupError,
    all_changes,
)
from ase.visualize import view
from ase.io import read, write
from ase.io.trajectory import Trajectory
import json
import pandas as pd
import os
from ase.calculators.harmonic import HarmonicForceField, HarmonicCalculator

from ase.md.velocitydistribution import Stationary, ZeroRotation
from ase.md.andersen import Andersen 
from ase.units import fs
from scipy.integrate import trapezoid



"""
    Theory for Anharmonic Correction via Thermodynamic Integration (TI)
    ====================================================================
    
    Thermodynamic integration (TI), i.e. λ-path integration, connects two thermodynamic 
    states via a λ-path. Here, the TI begins from a reference system '0' with known free 
    energy (Harmonic Approximation) and the Anharmonic Correction is obtained via 
    integration over the λ-path to the target system '1' (the fully interacting anharmonic 
    system). Hence, the free energy of the target system can be written as:
    
        A₁ = A₀ + ΔA₀→₁
    
    where the second term corresponds to the integral over the λ-path:
    
        ΔA₀→₁ = ∫₀¹ dλ⟨H₁ - H₀⟩_λ
    
    The term ⟨...⟩_λ represents the NVT ensemble average of the system driven by the 
    classical Hamiltonian H_λ determined by the coupling parameter λ ∈ [0, 1]:
    
        H_λ = λH₁ + (1 - λ)H₀
    
    Since the Hamiltonians differ only in their potential energy contributions V₁ and V₀, 
    the free energy change can be computed from the potentials:
    
        ΔA₀→₁ = ∫₀¹ dλ⟨V₁ - V₀⟩_λ
    
    The Cartesian coordinates x used in the common Harmonic Approximation are not insensitive 
    to overall rotations and translations that must leave the total energy invariant. This 
    limitation can be overcome by transformation of the Hessian in x to a suitable coordinate 
    system q (e.g. internal coordinates). Since the force field of that Hessian which is 
    harmonic in x is not necessarily equivalently harmonic in q, the free energy correction 
    can be rewritten to:
    
        A₁ = A₀,ₓ + ΔA₀,ₓ→₀,q + ΔA₀,q→₁
    
    The terms in this equation correspond to:
    - A₀,ₓ: the free energy from the Harmonic Approximation with the reference Hessian
    - ΔA₀,ₓ→₀,q: the free energy change due to the coordinate transformation obtained via TI
    - ΔA₀,q→₁: the free energy change from the harmonic to the fully interacting system 
                obtained via TI
    
    References
    ----------
    [1] Amsler, J. et al. for details on the implementation and theory.
    
"""

# Step 1: Get the cartesian coordinate paarmeters (ref_atoms, ref_energies and hessian_array in array). This will give you the  A₀_x
# option 1: Get these information from previous calculation (RRHO(pynta), HR, HT)
structure = read("/projects/westgroup/akinyemi.az/Pynta/project_pynta/pynta/pynta/testing_data/Ru0001_fischer_tropsch_3_17_25/Adsorbates/[H][H]/0/0.xyz")
ref_atoms = structure.copy()[-2:]
print(ref_atoms)
print(f"Length of {ref_atoms} is {len(ref_atoms)}")
print(f"Length of {structure} is {len(structure)}")
print(len(structure))
# from previous calculation
ref_energy= -123.456

hessian_x = "/projects/westgroup/akinyemi.az/Pynta/project_pynta/pynta/pynta/testing_data/Ru0001_fischer_tropsch_3_17_25/TS0/4/vib.json_vib.json"

import json
with open(hessian_x, 'r') as file:
    data = json.load(file)

hessian_matrix = data['hessian']

hessian_array = np.array(hessian_matrix)
print(hessian_array.shape)
hessian_array


# option 2: Calculate it yourself
# def build_harmonic_force_calculator(ref_atoms, ref_energy, hessian_array):
#     """
#     Create a harmonic force-field calculator for a given reference configuration.
#     """
#     # Create the harmonic force field
#     hff= HarmonicForceField(ref_atoms=ref_atoms, ref_energy=ref_energy,hessian_x=hessian_array)

#     # Copy atoms and assign calculator
#     atoms = ref_atoms.copy()
#     atoms.calc = HarmonicCalculator(hff)

#     return atoms

# atoms = build_harmonic_force_calculator(
#     ref_atoms=ref_atoms,
#     ref_energy=ref_energy,
#     hessian_x=hessian_array
# )
# print(atoms)

# Step 2:  Move through the intregation path from the cartesian coordinate to the internal coordinate
def get_distance_definitions(ref_atoms):
    """Generate all unique atom-pair distance definitions."""
    num_atoms = len(ref_atoms)
    dist_defs = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist_defs.append([i, j])
    return dist_defs


# Generate and store the distance definitions
dist_defs = get_distance_definitions(ref_atoms)
print("Distance definitions:", dist_defs)


def adsorbate_get_q_from_x(ref_atoms):
    """
    Compute internal coordinates (distances) for the adsorbate. 
    ΔA₀,ₓ→₀,q
    """
    q_vec = [ref_atoms.get_distance(i, j) for i, j in dist_defs]
    return np.asarray(q_vec)


def adsorbate_get_jacobian(ref_atoms):
    """
    Return Jacobian matrix for coordinates defined by distances.
    """
    ref_pos = ref_atoms.get_positions()

    # build distance vectors r_j - r_i
    dist_vecs = [ref_pos[j] - ref_pos[i] for i, j in dist_defs]

    # derivative dr/dx for each distance
    derivs = get_distances_derivatives(dist_vecs)

    jac = []
    for i, defin in enumerate(dist_defs):
        dqi_dxj = np.zeros(ref_pos.shape)  # shape (Natoms, 3)
        for k, deriv in enumerate(derivs[i]):
            atom_index = defin[k]
            dqi_dxj[atom_index] = deriv
        jac.append(dqi_dxj.flatten())

    return np.asarray(jac)


parameters = {
    'ref_atoms': ref_atoms,
    'ref_energy': ref_energy,
    'hessian_x': hessian_array,
    'get_q_from_x': adsorbate_get_q_from_x,
    'get_jacobian': adsorbate_get_jacobian,
    'cartesian': False
}

hff = HarmonicForceField(**parameters)
calc = HarmonicCalculator(hff)

# Step 3: calculate the free energy change due to coordinate transformation (from cartesian to internal coordinate). This will give you the Anharonic correction
def compute_anharmonic_correction(
    ref_atoms,
    ref_energy,
    hessian_array,
    adsorbate_get_q_from_x,
    adsorbate_get_jacobian,
    temperature=300,
    lambdas=(0.00, 0.25, 0.50, 0.75, 1.00),
    steps_per_lambda=50
):
    """
    Compute the anharmonic free-energy correction (dA) between Cartesian and
    internal-coordinate harmonic descriptions using thermodynamic integration.
    ΔA₀→₁ = ∫₀¹ dλ⟨V₁ - V₀⟩_λ
    """

    # Create Harmonic Force Fields (Cartesian and Internal)
    params = {
        "ref_atoms": ref_atoms,
        "ref_energy": ref_energy,
        "hessian_x": hessian_array,
        "get_q_from_x": adsorbate_get_q_from_x,
        "get_jacobian": adsorbate_get_jacobian,
        "cartesian": True,
        "variable_orientation": True
    }

    # Cartesian harmonic FF
    hff_cart = HarmonicForceField(**params)
    calc_cart = HarmonicCalculator(hff_cart)

    # Internal-coordinate harmonic FF
    params["cartesian"] = False
    hff_int = HarmonicForceField(**params)
    calc_int = HarmonicCalculator(hff_int)

    # Perform thermodynamic integration: 
    ediffs = {}
    fs = 1.0 * units.fs

    for lamb in lambdas:
        ediffs[lamb] = []

        # Mixed calculator: H(λ) = (1 - λ) H_internal  +  λ H_cartesian
        calc_mix = MixedCalculator(calc_int, calc_cart, 1 - lamb, lamb)
        e_cart = calc_cart.get_potential_energy(atoms)

        atoms = ref_atoms.copy()
        atoms.calc = calc_mix

        # Initialize velocities and remove translation/rotation
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
        Stationary(atoms)
        ZeroRotation(atoms)

        # Andersen thermostat MD
        with Andersen(atoms, 0.5 * fs, temperature_K=temperature,
                      andersen_prob=0.05, fixcm=False) as dyn:

            for _ in dyn.irun(steps_per_lambda):
                e_int, e_cart = calc_mix.get_energy_contributions(atoms)
                ediffs[lamb].append(float(e_cart) - float(e_int))

        # Mean at this lambda
        ediffs[lamb] = np.mean(ediffs[lamb])


    # Trapezoidal integration over lambda
    dA = trapezoid([ediffs[l] for l in lambdas], x=lambdas)

    return dA
# Step 4: Calculate A1: Total anharmonic free energy
def calculate_anharmonic_free_energy(A0, dA_x_to_q, dA_q_to_1=0.0):
    """
    Calculate total anharmonic free energy.
    A₁ = A₀ + ΔA₀→₁
    A₁ = A₀ + ΔA₀,x→₀,q + ΔA₀,q→₁
    """
    return A0 + dA_x_to_q + dA_q_to_1

# Run calculation
dA_x_to_q = compute_anharmonic_correction(
    ref_atoms, ref_energy, hessian_array,
    adsorbate_get_q_from_x, adsorbate_get_jacobian,
    temperature=300,
    lambdas=(0.0, 0.25, 0.5, 0.75, 1.0),
    steps_per_lambda=20  # reduce for faster testing
)

A1 = calculate_anharmonic_free_energy(ref_energy, dA_x_to_q)
print("Anharmonic free energy A1:", A1)

# Step 5: Calculate the hindered partition function

# Step 6: Calculate Thermodynamics.

