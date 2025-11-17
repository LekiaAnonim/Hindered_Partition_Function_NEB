from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS, FIRE
from ase.vibrations import Vibrations, VibrationsData
from acat.adsorption_sites import SlabAdsorptionSites
from acat.utilities import get_mic
from ase.build import add_adsorbate, fcc111
from ase.constraints import FixAtoms, Hookean
from ase.visualize import view
from ase.visualize.ngl import view_ngl
# import nglview as nv
from ase.optimize.minimahopping import MinimaHopping, MHPlot
from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.geometry import get_distances
from copy import deepcopy
import numpy as np
from ase.build import bulk, molecule
from ase.db import connect
from ase.eos import calculate_eos
import os
from scipy.spatial.transform import Rotation as R
import pandas as pd
from ase import io
from ase.mep import NEB
from ase.optimize import MDMin
import matplotlib.pyplot as plt

# Import Fairchem Calculator
# from fairchem.core.calculate import pretrained_mlip
# from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

# Import MACE Calculator
from mace.calculators import mace_mp


#!/usr/bin/env python3

vacuum = 20 # Angstrom
size = (3, 3, 3) # slab size
lattice_constant_vdW = 3.92 # Angstrom
lattice_constant = 3.97 # Angstrom
surface_type = 'fcc111'
metal = 'Pt'

# predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
# calc = FAIRChemCalculator(predictor, task_name="oc20")

calc = mace_mp(model="medium", device='cpu')

# ase_calculator = EMT()
ase_calculator = calc

def calculate_energy(slab, adsorbate, position, energy_type='total'):
    ''' Calculate energy of adsorbate on slab at given position
    
    Args:
        slab: clean slab structure
        adsorbate: optimized adsorbate molecule
        position: [x, y] position on surface
        energy_type: 'total' for PES/barriers, 'adsorption' for site comparison
        
    Returns:
        energy in eV (total or adsorption depending on energy_type)
    ''' 
    # Create a copy to avoid modifying original
    test_slab = slab.copy()
    
    # Add adsorbate at specified xy position with default height
    add_adsorbate(test_slab, adsorbate, height=2.0, position=position[:2])
    
    # Calculate total energy
    test_slab.calc = ase_calculator
    E_total = test_slab.get_potential_energy()
    
    if energy_type == 'total':
        # For PES and barrier calculations
        return E_total
    elif energy_type == 'adsorption':
        # For comparing site stability
        E_slab = slab.get_potential_energy()
        E_ads = adsorbate.get_potential_energy()
        E_adsorption = E_total - E_slab - E_ads
        return E_adsorption
    else:
        raise ValueError(f"Unknown energy_type: {energy_type}")

def PES(n_points=10):
    """
    Calculate 2D Potential Energy Surface for CH4 on Pt(111)
    
    Args:
        n_points: number of grid points in each direction
        
    Returns:
        X, Y, Z meshgrid arrays for plotting
    """
    print("Setting up PES calculation...")
    
    # Create clean slab and optimized adsorbate
    slab = fcc111(metal, size=size, a=lattice_constant, vacuum=vacuum)
    slab.calc = ase_calculator
    E_slab = slab.get_potential_energy()
    print(f"Clean slab energy: {E_slab:.3f} eV")
    
    # Optimize adsorbate
    adsorbate = molecule('CH4')
    adsorbate.calc = ase_calculator
    opt = BFGS(adsorbate, trajectory=None)
    opt.run(fmax=0.05)
    E_ads = adsorbate.get_potential_energy()
    print(f"Adsorbate energy: {E_ads:.3f} eV")
    
    # Create 2D grid of positions
    cell = slab.get_cell()
    x_max = cell[0, 0]  # ~11.91 Å (3 × 3.97 Å)
    y_max = cell[1, 1]  # ~11.91 Å (3 × 3.97 Å)
    x_range = np.linspace(0, x_max, n_points)
    y_range = np.linspace(0, y_max, n_points)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate energies at each grid point
    energies = np.zeros_like(X)
    
    for i in range(n_points):
        for j in range(n_points):
            position = [X[i, j], Y[i, j]]
            # Use 'total' energy for PES (not 'adsorption')
            energies[i, j] = calculate_energy(slab, adsorbate, position, energy_type='total')
            
            if (i * n_points + j + 1) % 10 == 0:
                print(f"  Progress: {i * n_points + j + 1}/{n_points**2}")
    
    print("\nPES calculation complete!")
    
    # Shift energies relative to minimum for better visualization
    E_min = np.min(energies)
    # energies_relative = energies - E_min
    energies_relative = energies
    
    # Plot 2D contour map
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Contour plot (show relative energies for clarity)
    contour = ax1.contourf(X, Y, energies_relative, levels=20, cmap='RdYlBu_r')
    ax1.contour(X, Y, energies_relative, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    fig.colorbar(contour, ax=ax1, label='Relative Energy (eV)')
    ax1.set_xlabel('X Position (Å)', fontsize=12)
    ax1.set_ylabel('Y Position (Å)', fontsize=12)
    ax1.set_title('2D Potential Energy Surface (Total Energy)', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    
    # 3D surface plot (use relative energies)
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, energies_relative, cmap='RdYlBu_r', alpha=0.8)
    ax2.set_xlabel('X Position (Å)')
    ax2.set_ylabel('Y Position (Å)')
    ax2.set_zlabel('Relative Energy (eV)')
    ax2.set_title('3D PES (Total Energy - E_min)', fontweight='bold')
    fig.colorbar(surf, ax=ax2, label='Energy (eV)', shrink=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return X, Y, energies


def get_unique_sites(site_list, cell, unique_composition=False,
                         unique_subsurf=False,
                         return_signatures=False,
                         return_site_indices=False,
                         about=None):
        """Function mostly copied from the ACAT software 
        Get all symmetry-inequivalent adsorption sites (one
        site for each type).

        Parameters
        ----------
        unique_composition : bool, default False
            Take site composition into consideration when
            checking uniqueness.

        unique_subsurf : bool, default False
            Take subsurface element into consideration when
            checking uniqueness.

        return_signatures : bool, default False
            Whether to return the unique signatures of the
            sites instead.

        return_site_indices: bool, default False
            Whether to return the indices of each unique
            site (in the site list).

        about: numpy.array, default None
            If specified, returns unique sites closest to
            this reference position.

        """ 
        sl = site_list[:]
        key_list = ['site', 'morphology']
        if unique_composition:
            key_list.append('composition')
            if unique_subsurf:
                key_list.append('subsurf_element')
        else:
            if unique_subsurf:
                raise ValueError('to include the subsurface element, ' +
                                 'unique_composition also need to be set to True')

        seen_tuple = []
        uni_sites = []
        if about is not None:
            sl = sorted(sl, key=lambda x: get_mic(x['position'],
                        about, cell, return_squared_distance=True))
        for i, s in enumerate(sl):
            sig = tuple(s[k] for k in key_list)
            if sig not in seen_tuple:
                seen_tuple.append(sig)
                if return_site_indices:
                    s = i
                uni_sites.append(s)

        return uni_sites


def generate_unique_placements(slab,sites):
    nslab = len(slab)
    middle = sum(slab.cell)/2.0

    unique_single_sites = get_unique_sites(sites,slab.cell,about=middle)

    unique_site_pairs = dict() # (site1site,site1morph,),(site2site,site2morph),xydist,zdist
    for unique_site in unique_single_sites:
        uni_site_fingerprint = (unique_site["site"],unique_site["morphology"])
        for site in sites:
            site_fingerprint = (site["site"],site["morphology"])
            bd,d = get_distances([unique_site["position"]], [site["position"]], cell=slab.cell, pbc=(True,True,False))
            xydist = np.linalg.norm(bd[0][0][:1])
            zdist = bd[0][0][2]

            fingerprint = (uni_site_fingerprint,site_fingerprint,round(xydist,3),round(zdist,3))

            if fingerprint in unique_site_pairs.keys():
                current_sites = unique_site_pairs[fingerprint]
                current_dist = np.linalg.norm(sum([s["position"][:1] for s in current_sites])/2-middle[:1])
                possible_dist = np.linalg.norm((unique_site["position"][:1]+site["position"][:1])/2-middle[:1])
                if possible_dist < current_dist:
                    unique_site_pairs[fingerprint] = [unique_site,site]
            else:
                unique_site_pairs[fingerprint] = [unique_site,site]

    unique_site_pairs_lists = list(unique_site_pairs.values())
    unique_site_lists = [[unique_site] for unique_site in unique_single_sites]

    single_site_bond_params_lists = []
    for unique_site_list in unique_site_lists:
        pos = deepcopy(unique_site_list[0]["position"])
        single_site_bond_params_lists.append([{"site_pos": pos,"ind": None, "k": 100.0, "deq": 0.0}])

    double_site_bond_params_lists = []
    for unique_site_pair_list in unique_site_pairs_lists:
        bond_params_list = []
        for site in unique_site_pair_list:
            pos = deepcopy(site["position"])
            bond_params_list.append({"site_pos": pos,"ind": None, "k": 100.0, "deq": 0.0})
        double_site_bond_params_lists.append(bond_params_list)

    return unique_site_lists,unique_site_pairs_lists,single_site_bond_params_lists,double_site_bond_params_lists


def site_density(slab, cas):
    '''
    Calculate the site density in molecules/m^2
    '''
    S = len(cas.get_sites())
    cell = slab.cell
    n = np.cross(cell[0],cell[1])
    A = np.linalg.norm(n)
    site_density = S/A * 10**20 #molecules/m^2
    print(f"Site density: {site_density:.2e} molecules/m^2")
    return site_density


def adsorption_sites_and_unique_placements(slab, surface_type='fcc111'):
    ads_sites = SlabAdsorptionSites(slab, surface=surface_type)
    all_sites = ads_sites.get_sites()
    print(f"Total number of sites found: {len(all_sites)}")
    site_dens = site_density(slab, ads_sites)
    unique_site_lists, unique_site_pair_lists, single_bond_params, double_bond_params = generate_unique_placements(slab, all_sites)
    print(f"\nUnique single sites: {len(unique_site_lists)}")
    print(f"Unique site pairs: {len(unique_site_pair_lists)}")
    for i, site_list in enumerate(unique_site_lists):
        site = site_list[0]
    print(f"Site {i}: {site['site']} at position {site['position']}")
    return all_sites, site_dens, unique_site_lists, unique_site_pair_lists, single_bond_params, double_bond_params


def init_molecule(mol='CH4'):
    '''
    Create a clean adsorbate molecule
    '''
    adsorbate = molecule(mol)
    print(f"Number of atoms: {len(adsorbate)}")
    return adsorbate


def opt_molecule(adsorbate):
    '''
    Optimize the adsorbate molecule
    '''
    adsorbate.calc = ase_calculator
    # Optimize
    opt = BFGS(adsorbate, trajectory=None)
    opt.run(fmax=0.05)
    return adsorbate


def rotate_adsorbate_about_axis(atoms, adsorbate_indices, rotation_center_xy, 
                                angle_degrees, axis='z'):
    atoms_rotated = atoms.copy()
    
    # Get adsorbate positions
    ads_positions = atoms_rotated.positions[adsorbate_indices].copy()
    
    # Translate so rotation axis passes through origin in x-y plane
    ads_positions[:, 0] -= rotation_center_xy[0]
    ads_positions[:, 1] -= rotation_center_xy[1]
    
    # Create rotation
    if axis == 'z':
        rot_vector = np.array([0, 0, 1])
    elif axis == 'x':
        rot_vector = np.array([1, 0, 0])
    elif axis == 'y':
        rot_vector = np.array([0, 1, 0])
    else:
        raise ValueError(f"Unknown axis: {axis}")
    
    rotation = R.from_rotvec(np.radians(angle_degrees) * rot_vector)
    ads_positions = rotation.apply(ads_positions)
    
    # Translate back
    ads_positions[:, 0] += rotation_center_xy[0]
    ads_positions[:, 1] += rotation_center_xy[1]
    
    # Update positions
    atoms_rotated.positions[adsorbate_indices] = ads_positions
    
    return atoms_rotated


# Defining reasonable Hookean thresholds

hookean_rt = {
    'O-H': 1.4,
    'C-H': 1.59,
    'C-O': 1.79,
    'C=O': 1.58,
    'C-C': 1.54,
    'C=C': 1.34,
    'C-N': 1.47,
    'C=N': 1.27
}

hookean_k = {
    'O-H': 5,
    'C-H': 7,
    'C-O': 5,
    'C=O': 10,
    'C-C': 5,
    'C=C': 10,
    'C-N': 5,
    'C=N': 10
}



from ase.data import covalent_radii, atomic_numbers

class AdsorbateConstraintManager:
    """
    Automatically detect bonds in adsorbate and apply Hookean constraints
    """
    
    def __init__(self, hookean_rt, hookean_k):
        """
        Args:
            hookean_rt: dict of equilibrium bond lengths {bond_type: distance}
            hookean_k: dict of spring constants {bond_type: k_value}
        """
        self.hookean_rt = hookean_rt
        self.hookean_k = hookean_k
        
        # Tolerances for bond detection
        self.bond_tolerance = 0.3  # Å beyond covalent radii sum
        
    def detect_bonds(self, atoms, adsorbate_indices):
        """
        Detect bonds within adsorbate based on distances
        
        Args:
            atoms: ASE Atoms object (full system)
            adsorbate_indices: list of indices belonging to adsorbate
        
        Returns:
            list of dicts: [{'atom1': i, 'atom2': j, 'bond_type': 'C-H', 
                            'distance': 1.09, 'order': 1}]
        """
        bonds = []
        ads_symbols = [atoms[i].symbol for i in adsorbate_indices]
        ads_positions = atoms.positions[adsorbate_indices]
        
        # Check all pairs
        for i, idx_i in enumerate(adsorbate_indices):
            for j, idx_j in enumerate(adsorbate_indices):
                if i >= j:  # Avoid double counting and self
                    continue
                
                symbol_i = atoms[idx_i].symbol
                symbol_j = atoms[idx_j].symbol
                
                # Calculate distance
                distance = np.linalg.norm(ads_positions[i] - ads_positions[j])
                
                # Covalent radii sum
                r_cov_sum = (covalent_radii[atomic_numbers[symbol_i]] + 
                           covalent_radii[atomic_numbers[symbol_j]])
                
                # Check if bonded (distance < covalent sum + tolerance)
                if distance < r_cov_sum + self.bond_tolerance:
                    # Classify bond type
                    bond_type, bond_order = self._classify_bond(
                        symbol_i, symbol_j, distance
                    )
                    
                    bonds.append({
                        'atom1': idx_i,
                        'atom2': idx_j,
                        'symbol1': symbol_i,
                        'symbol2': symbol_j,
                        'bond_type': bond_type,
                        'distance': distance,
                        'order': bond_order
                    })
        
        return bonds
    
    def _classify_bond(self, symbol1, symbol2, distance):
        """
        Classify bond type and order based on atoms and distance
        
        Returns:
            bond_type (str): e.g., 'C-H', 'C=O'
            bond_order (int): 1, 2, or 3
        """
        # Canonical ordering (alphabetical)
        if symbol1 > symbol2:
            symbol1, symbol2 = symbol2, symbol1
        
        bond_label = f"{symbol1}-{symbol2}"
        
        # Check if it's a double or triple bond based on distance
        # C=O is typically 1.20-1.25 Å, C-O is 1.40-1.45 Å
        # C=C is typically 1.30-1.35 Å, C-C is 1.50-1.55 Å

        # https://pdf.sciencedirectassets.com/315523/3-s2.0-C20141017292/3-s2.0-B9780124095472113708/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDgaCXVzLWVhc3QtMSJHMEUCIGGtZyUoKY0wuppjkDwmLQasLQhhcR6Cnepo0NNpqrqMAiEAy76DLLmtZVmypHnSQMkCUZ6OvyfnjMDdplSLwwgL1FEquwUI8f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDPKh%2F%2FvBsAq%2B0U5wxiqPBWjY%2Be2zdmyUvExVeA1eJPPhi%2Brp3sz4wDGJWRf8tigmuKoR5hDAnsFdWRm1AKNwTpendx7ZaV1igHUW0IO%2F2yn3%2FizFG0rdXgQqBTFj8kS3HX8cy7x5YLsknZfijjlyrWKr%2BkgBy5oj4f%2Fdc7d9WOn8k02rsTxiACIRm1Pwf5wk%2BUW6zdPUsDqsDc3gH540H06orprJ7IbYzifzmFKlphgrzIC%2FyhHUvXwrKowZo5xdrJYk30G7BOI%2FTRkuSNo64poyjBEEcOv8gCGRtU5KbR0T6GDpU5dPPW6E%2F00gd2NVxc8QSs1Q6DcPfBONwQrbV57JIvKFLo9LpljGAySh6v9bjR2B11bVyNKlPmnMnAGCHZVzgfgPlTdJHUER1yKxI%2F8RTl8Zx7%2FZYTB9U%2FpTxTA0RR5vLi1Sz%2Bvh1A0rnfw07AFouLxYusYpFohUaz9%2BeCs8E2JpVQA3e7n6IC9Vu%2BPeTYK8pp5xLNWovj9FNKRmdKo2Q0fFmhS4Rw%2F%2FklN79vdqOagXI6O%2BEiHNrSCgnF2Q32WvIv62eeb73Y%2BE1RxVZDClujUuShnaFLC7YbpU3qZbSVQLVac3uT2M1O%2FircV7xufj9ASBkwwXu1bbZMX1TePs2TOYZOhJIvcUAeasq0x6VRdkUdaDyPnIVaOJiSO4VlBySuPOPnY5ND%2BrTe3K4hMiJesFYACbLGEw0rW2Y%2BRBLxkky2mbcdNulUN8fc5KXsDfEII3zRHJXxpcjDvnDReq8Iid7sx6xRpc1vgjsg68pPAjwPyCHqOJ1FXYhENPmnSZuDCyDBiO3BdH9IB3UcWBtRiCFosEh5q%2FOF9%2F%2B1dL9n7daig1jXgvkxx4bH5eMwhpmGaX7sn9X7S7CcQw14WOyAY6sQGefSutnE%2B8Ki20c4sAUupIMDqsH2ooUPtoB6pTdlcx9TpnAmsbYSXrgsXpU6XrAPKsVrY%2BeHUn%2FYxqYieiiakA%2Fye3PzlYwB2QN9lt%2F8RIjBmEpEmpYFvLC4t52bYWwIUpXW9NWnJFSGpdOZZ4USlrF7pVyTUdeyEIbnU2B3swCpfnO4yCnhr5N45DBSrdY3b7ugXTioKPgF8%2B5s%2Fyk8Us9aGKvmY0LDTQMTevx%2BSJ1TI%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20251030T163848Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYVAK5I3G4%2F20251030%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=7ca8b1d95838e7bd64e8d9786271994caafca2ba6542ee100c4bcbf653286e35&hash=deff698932bc3b32fd5223d514bdf410131c0167dc7923ee40b7bd0d3a743571&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=B9780124095472113708&tid=spdf-60eca882-67f6-4f7d-9301-d98bf5be7a6a&sid=8f38fe1381154245933a318-6a5b8f67f93cgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=13145f5650015557505b&rr=996c58588f72e007&cc=us
        
        if bond_label == "C-O":
            if distance < 1.30:  # C=O double bond
                return "C=O", 2
            else:  # C-O single bond
                return "C-O", 1
        
        elif bond_label == "C-C":
            if distance < 1.25:  # C≡C triple bond
                return "C≡C", 3
            elif distance > 1.25 and distance < 1.40:  # C=C double bond
                return "C=C", 2
            else:  # C-C single bond
                return "C-C", 1
        
        elif bond_label == "C-N":
            if distance < 1.15:  # C≡N single bond
                return "C≡N", 3
            elif distance > 1.15 and distance < 1.30:  # C=N double bond
                return "C=N", 2
            else:  # C-N single bond
                return "C-N", 1
        
        else:
            # Default: single bond
            return bond_label, 1
    
    def get_hookean_parameters(self, bond_type):
        """
        Get rt and k for a given bond type
        
        Returns rt, k or None if not defined
        """
        # Try exact match first
        if bond_type in self.hookean_rt and bond_type in self.hookean_k:
            return self.hookean_rt[bond_type], self.hookean_k[bond_type]
        
        # Try reversed (e.g., H-C instead of C-H)
        reversed_type = '-'.join(reversed(bond_type.split('-')))
        if reversed_type in self.hookean_rt and reversed_type in self.hookean_k:
            return self.hookean_rt[reversed_type], self.hookean_k[reversed_type]
        
        # Try without bond order indicator (e.g., C-O for C=O)
        if '=' in bond_type or '≡' in bond_type:
            simple_type = bond_type.replace('=', '-').replace('≡', '-')
            if simple_type in self.hookean_rt and simple_type in self.hookean_k:
                return self.hookean_rt[simple_type], self.hookean_k[simple_type]
        
        return None, None
    
    def create_hookean_constraints(self, atoms, adsorbate_indices):
        """
        Create Hookean constraints for all bonds in adsorbate
        
        Args:
            atoms: ASE Atoms object
            adsorbate_indices: list of adsorbate atom indices
        
        Returns:
            list of Hookean constraint objects
        """
        bonds = self.detect_bonds(atoms, adsorbate_indices)
        
        print(f"\n=== Detected {len(bonds)} bonds in adsorbate ===")
        
        hookean_constraints = []
        
        for bond in bonds:
            bond_type = bond['bond_type']
            rt, k = self.get_hookean_parameters(bond_type)
            
            if rt is not None and k is not None:
                # Create Hookean constraint
                constraint = Hookean(
                    a1=bond['atom1'],
                    a2=bond['atom2'],
                    rt=rt,  # Equilibrium distance
                    k=k     # Spring constant
                )
                hookean_constraints.append(constraint)
                
                print(f"  {bond['symbol1']}-{bond['symbol2']}: "
                      f"d={bond['distance']:.3f} Å → Hookean(rt={rt:.2f}, k={k})")
            else:
                print(f"  {bond_type}: No Hookean parameters defined "
                      f"(d={bond['distance']:.3f} Å)")
        
        return hookean_constraints
    
    def apply_constraints(self, structure, adsorbate_indices, 
                         fix_slab_indices=None, fix_slab_tag=None):
        """
        Apply both slab constraints and adsorbate Hookean constraints
        
        Args:
            structure: ASE Atoms object (slab + adsorbate)
            adsorbate_indices: list of adsorbate atom indices
            fix_slab_indices: list of slab atoms to fix (optional)
            fix_slab_tag: fix atoms with tag > this value (optional)
        
        Returns:
            modified structure with constraints applied
        """
        constraints = []
        
        # 1. Fix slab atoms
        if fix_slab_indices is not None:
            slab_constraint = FixAtoms(indices=fix_slab_indices)
            constraints.append(slab_constraint)
            print(f"Fixed {len(fix_slab_indices)} slab atoms by index")
        
        elif fix_slab_tag is not None:
            fix_indices = [atom.index for atom in structure if atom.tag > fix_slab_tag]
            slab_constraint = FixAtoms(indices=fix_indices)
            constraints.append(slab_constraint)
            print(f"Fixed {len(fix_indices)} slab atoms with tag > {fix_slab_tag}")
        
        # 2. Add Hookean constraints for adsorbate bonds
        hookean_constraints = self.create_hookean_constraints(
            structure, adsorbate_indices
        )
        constraints.extend(hookean_constraints)
        
        # Apply all constraints
        structure.set_constraint(constraints)
        
        print(f"\nTotal constraints applied: {len(constraints)}")
        print(f"  - Slab: 1 (FixAtoms)")
        print(f"  - Adsorbate: {len(hookean_constraints)} (Hookean)")
        
        return structure


def create_structure(slab, opt_adsorbate, site, bond_params=None, height=None, rotation=None, rotation_center='binding', binding_atom_idx=None, hookean_rt=None, hookean_k=None,
                                       apply_hookean=True):
    """
    Args:
        slab: ASE Atoms object (surface)
        adsorbate: ASE Atoms object (molecule to adsorb)
        site_pos: [x, y, z] position of adsorption site
        height: height above site to place adsorbate (Å)
        rotation: rotation angle around z-axis (degrees)
        rotation_center: 'com', 'binding', 'site', or [x, y] coordinates
            'com': center of mass (gas-phase-like)
            'binding': closest atom to surface (surface-chemistry-like)
            'site': the adsorption site position
            [x, y]: explicit coordinates
        binding_atom_idx: index of binding atom (if rotation_center='binding')
                         If None, automatically finds lowest atom
    
    """
    # bond_params is a list with one dict: [{"site_pos": pos, "k": 100.0, "deq": 0.0}]
    params = bond_params[0]
    spring_constant = params['k']       # Spring constant from bond params
    site_position = params['site_pos']  # Site position from bond params

    structure = slab.copy()
    n_slab = len(structure)
    ads = opt_adsorbate.copy()
    com = ads.get_center_of_mass()
    target_position = np.array([site['position'][0], site['position'][1], site['position'][2] + height])
    ads.translate(target_position - com)

    if rotation_center == 'com':
        # Rotate around center of mass
        center_xy = ads.get_center_of_mass()[:2]
    elif rotation_center == 'binding':
        # Rotate around binding atom (closest to surface)
        if binding_atom_idx is None:
            # Auto-detect: lowest atom
            heights = ads.positions[:, 2]
            binding_atom_idx = np.argmin(heights)
        center_xy = ads.positions[binding_atom_idx][:2]

    elif rotation_center == 'site':
        # Rotate around the site position
        center_xy = site['position'][:2]
        
    elif isinstance(rotation_center, (list, tuple, np.ndarray)):
        # Explicit coordinates
        center_xy = np.array(rotation_center[:2])
        
    else:
        raise ValueError(f"Unknown rotation_center: {rotation_center}")

    if rotation != 0:
        ads_indices = list(range(len(ads)))
        ads = rotate_adsorbate_about_axis(
            ads, 
            ads_indices, 
            center_xy, 
            rotation
        )
    add_adsorbate(structure, ads, 
                  height=height, 
                  position=site['position'][:2])
    
    # Set constraints (fix bottom layers)
    # Combine
    adsorbate_indices = list(range(n_slab, len(structure)))
    
    # Apply constraints
    if apply_hookean and hookean_rt is not None and hookean_k is not None:
        manager = AdsorbateConstraintManager(hookean_rt, hookean_k)
        structure = manager.apply_constraints(
            structure,
            adsorbate_indices,
            fix_slab_tag=1  # Fix atoms with tag > 1
        )
    else:
        # Just fix slab without Hookean
        fix_indices = [atom.index for atom in structure if atom.tag > 1]
        structure.set_constraint(FixAtoms(indices=fix_indices))
    return structure


def clean_slab(metal='Pt', size=(3, 3, 3), lattice_constant=3.97, vacuum=20):
    '''
    Create a clean slab
    '''
    slab = fcc111(metal, size=size, a=lattice_constant, vacuum=vacuum)
    slab.set_pbc(True)
    slab.calc = ase_calculator
    E_clean = slab.get_potential_energy()
    print(f"Clean slab energy: {E_clean:.3f} eV\n")
    # create a directory for the slab file if it does not exist
    slab_dir = 'Slab'
    os.makedirs(slab_dir, exist_ok=True)
    # Write slab to file
    slab.write(f'{slab_dir}/slab_init.xyz')
    return slab


def opt_slab(metal='Pt', size=(3, 3, 3), lattice_constant=3.97, vacuum=20):
    slab = clean_slab(metal=metal, size=size, lattice_constant=lattice_constant, vacuum=vacuum)
    opt = BFGS(slab, trajectory=None)
    opt.run(fmax=0.05)
    slab.calc = ase_calculator
    slab_dir = 'Slab'
    os.makedirs(slab_dir, exist_ok=True)
    # Write slab to file
    slab.write(f'{slab_dir}/slab.xyz')
    return slab


def site_screening(slab, ads, center_xy='binding', use_all_sites=True, save_results=True, output_dir='Screening_Data'):
    """
    Perform site screening and optionally save results
    
    Args:
        save_results: If True, saves screening results to pickle file
        output_dir: Directory to save results
    """
    slab = opt_slab()
    slab.calc = ase_calculator
    all_sites, site_density, unique_site_lists, unique_site_pair_lists, single_bond_params, double_bond_params = adsorption_sites_and_unique_placements(slab)
    screening_results = []
    heights = np.arange(1.5, 3.5, 0.5)
    rotations = np.arange(0, 360, 30)
    ads = opt_molecule(ads)
    if use_all_sites:
        sites_to_screen = all_sites
    else:
        sites_to_screen = [site_list[0] for site_list in unique_site_lists]

    for i, site in enumerate(sites_to_screen): 
        bond_params = [{"site_pos": site['position'], "ind": None, "k": 100.0, "deq": 0.0}]
        for height in heights:
            for rot in rotations:
                # Create structure with adsorbate at specified height and rotation
                test_slab = create_structure(
                    slab, 
                    ads, 
                    site,
                    bond_params,
                    height, 
                    rotation=rot, 
                    rotation_center=center_xy,
                    binding_atom_idx=None
                )

                # Set calculator
                test_slab.calc = ase_calculator

                # Optimize
                log_dir = 'Screening_logs'
                os.makedirs(log_dir, exist_ok=True)
                opt = BFGS(test_slab, logfile=f'{log_dir}/site_{i}_{site["site"]}_h{height}_r{rot}.log', trajectory=None)
                try:
                    opt.run(fmax=0.05)
                    converged = True
                except Exception as e:
                    print(f"Optimization failed for site {i}, height {height}, rotation {rot}: {e}")
                    converged = False
                    continue

                slab_dir = 'Screening_Results'
                os.makedirs(slab_dir, exist_ok=True)
                filename = f"{slab_dir}/slab_{site['site']}_h{height}_r{rot}.xyz"
                test_slab.write(filename)
                clean_slab = slab.copy()
                clean_slab.calc = ase_calculator
                E_slab = clean_slab.get_potential_energy()
                # Calculate adsorption energy
                E_total = test_slab.get_potential_energy()
                E_adsorbate = ads.get_potential_energy()
                E_ads = E_total - E_slab - E_adsorbate

                # Store results
                result = {
                    'site_index': i,
                    'site_type': site['site'],
                    'site_position': site['position'],
                    'height': height,
                    'rotation': rot,
                    'adsorption_energy': E_ads,
                    'total_energy': E_total,
                    'structure': test_slab.copy(),
                    'structure_file': filename,
                    'converged': converged
                }
                screening_results.append(result)

                print(f"Site {i} ({site['site']:6s}), Height {height:.1f} Å, Rotation {rot}°: E_ads = {E_ads:7.3f} eV")
    
    # Save results if requested
    # Save results if requested
    if save_results:
        import pickle
        import json
        from datetime import datetime
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save with pickle (includes ASE atoms objects) - use temp file for atomic write
            pickle_file = f"{output_dir}/screening_results.pkl"
            pickle_temp = f"{pickle_file}.tmp"
            
            try:
                with open(pickle_temp, 'wb') as f:
                    pickle.dump(screening_results, f)
                # Atomic rename
                os.replace(pickle_temp, pickle_file)
                print(f"✓ Saved pickle file: {pickle_file}")
            except Exception as e:
                print(f"⚠ Warning: Failed to save pickle file: {e}")
                if os.path.exists(pickle_temp):
                    os.remove(pickle_temp)
            
            # Also save metadata as JSON (without atoms objects)
            try:
                def convert_to_json_serializable(obj):
                    """Convert numpy types to Python native types"""
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                         np.int16, np.int32, np.int64,
                                         np.uint8, np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_to_json_serializable(item) for item in obj]
                    else:
                        return obj
                
                metadata = []
                for r in screening_results:
                    # Exclude structure object
                    meta = {k: v for k, v in r.items() if k not in ['structure']}
                    # Convert all numpy types
                    meta = convert_to_json_serializable(meta)
                    metadata.append(meta)
                
                json_file = f"{output_dir}/screening_metadata_{timestamp}.json"
                json_temp = f"{json_file}.tmp"
                
                with open(json_temp, 'w') as f:
                    json.dump(metadata, f, indent=2)
                # Atomic rename
                os.replace(json_temp, json_file)
                print(f"✓ Saved JSON metadata: {json_file}")
            except Exception as e:
                print(f"⚠ Warning: Failed to save JSON metadata: {e}")
                if os.path.exists(json_temp):
                    os.remove(json_temp)
            
            # Save summary statistics
            try:
                summary_file = f"{output_dir}/screening_summary_{timestamp}.txt"
                with open(summary_file, 'w') as f:
                    f.write(f"Screening Results Summary\n")
                    f.write(f"{'='*70}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total configurations: {len(screening_results)}\n")
                    f.write(f"Site density: {site_density}\n")
                    f.write(f"Converged: {sum(1 for r in screening_results if r.get('converged', False))}\n")
                    f.write(f"Site types: {', '.join(sorted(set(r['site_type'] for r in screening_results)))}\n")
                    f.write(f"\nEnergy Statistics:\n")
                    energies = [r['adsorption_energy'] for r in screening_results if r.get('converged', False)]
                    if energies:
                        f.write(f"  Min: {min(energies):.6f} eV\n")
                        f.write(f"  Max: {max(energies):.6f} eV\n")
                        f.write(f"  Mean: {np.mean(energies):.6f} eV\n")
                        f.write(f"  Std: {np.std(energies):.6f} eV\n")
                print(f"✓ Saved summary: {summary_file}")
            except Exception as e:
                print(f"⚠ Warning: Failed to save summary: {e}")
            
            print(f"\n{'='*70}")
            print(f"✓ Screening results saved successfully!")
            print(f"  Location: {output_dir}/")
            print(f"  Timestamp: {timestamp}")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"⚠ ERROR: Failed to save screening results!")
            print(f"  Error: {e}")
    return screening_results


def validate_screening_files(output_dir='Screening_Data'):
    """
    Validate and check integrity of screening result files
    
    Args:
        output_dir: Directory containing screening results
    
    Returns:
        Dictionary with validation results
    """
    import glob
    import json
    import pickle
    from datetime import datetime
    
    if not os.path.exists(output_dir):
        print(f" Directory not found: {output_dir}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Validating Screening Files in: {output_dir}")
    print(f"{'='*70}\n")
    
    validation = {
        'pickle_files': [],
        'json_files': [],
        'summary_files': [],
        'corrupt_files': [],
        'valid_timestamps': []
    }
    
    # Check pickle files
    pickle_files = glob.glob(f"{output_dir}/screening_results*.pkl")
    for pf in sorted(pickle_files):
        try:
            with open(pf, 'rb') as f:
                data = pickle.load(f)
            size_mb = os.path.getsize(pf) / (1024 * 1024)
            timestamp = os.path.basename(pf).replace('screening_results', '').replace('.pkl', '')
            
            validation['pickle_files'].append({
                'file': pf,
                'timestamp': timestamp,
                'size_mb': size_mb,
                'num_results': len(data),
                'valid': True
            })
            validation['valid_timestamps'].append(timestamp)
            
            print(f"✓ VALID PICKLE: {os.path.basename(pf)}")
            print(f"  Size: {size_mb:.2f} MB, Results: {len(data)}")
            
        except Exception as e:
            validation['corrupt_files'].append({'file': pf, 'error': str(e)})
            print(f" CORRUPT PICKLE: {os.path.basename(pf)}")
            print(f"  Error: {e}")
    
    # Check JSON files
    json_files = glob.glob(f"{output_dir}/screening_metadata_*.json")
    for jf in sorted(json_files):
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            size_kb = os.path.getsize(jf) / 1024
            timestamp = os.path.basename(jf).replace('screening_metadata_', '').replace('.json', '')
            
            validation['json_files'].append({
                'file': jf,
                'timestamp': timestamp,
                'size_kb': size_kb,
                'num_results': len(data),
                'valid': True
            })
            
            print(f"✓ VALID JSON: {os.path.basename(jf)}")
            print(f"  Size: {size_kb:.2f} KB, Results: {len(data)}")
            
        except json.JSONDecodeError as e:
            validation['corrupt_files'].append({'file': jf, 'error': f'JSON decode error: {e}'})
            print(f" CORRUPT JSON: {os.path.basename(jf)}")
            print(f"  Error: JSON decode failed - file may be truncated")
            
            # Try to show last few bytes
            try:
                with open(jf, 'r') as f:
                    content = f.read()
                print(f"  File size: {len(content)} bytes")
                print(f"  Last 100 chars: ...{content[-100:]}")
            except:
                pass
    
    # Check summary files
    summary_files = glob.glob(f"{output_dir}/screening_summary_*.txt")
    for sf in sorted(summary_files):
        timestamp = os.path.basename(sf).replace('screening_summary_', '').replace('.txt', '')
        validation['summary_files'].append({
            'file': sf,
            'timestamp': timestamp
        })
        print(f"✓ SUMMARY: {os.path.basename(sf)}")
    
    # Check for orphaned files
    print(f"\n{'='*70}")
    print("Timestamp Analysis:")
    print(f"{'='*70}")
    
    pickle_ts = set(v['timestamp'] for v in validation['pickle_files'])
    json_ts = set(v['timestamp'] for v in validation['json_files'])
    summary_ts = set(v['timestamp'] for v in validation['summary_files'])
    
    complete_sets = pickle_ts & json_ts & summary_ts
    incomplete_sets = (pickle_ts | json_ts | summary_ts) - complete_sets
    
    if complete_sets:
        print(f"\n✓ Complete sets (pickle + json + summary): {len(complete_sets)}")
        for ts in sorted(complete_sets):
            print(f"  - {ts}")
    
    if incomplete_sets:
        print(f"\n⚠ Incomplete sets (missing files): {len(incomplete_sets)}")
        for ts in sorted(incomplete_sets):
            has_pickle = ts in pickle_ts
            has_json = ts in json_ts
            has_summary = ts in summary_ts
            status = f"{'P' if has_pickle else '-'}{'J' if has_json else '-'}{'S' if has_summary else '-'}"
            print(f"  - {ts} [{status}]  (P=pickle, J=json, S=summary)")
    
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  Total pickle files: {len(validation['pickle_files'])}")
    print(f"  Total JSON files: {len(validation['json_files'])}")
    print(f"  Total summary files: {len(validation['summary_files'])}")
    print(f"  Corrupt files: {len(validation['corrupt_files'])}")
    print(f"  Complete sets: {len(complete_sets)}")
    print(f"{'='*70}\n")
    
    return validation


def clean_incomplete_files(output_dir='Screening_Data', dry_run=True):
    """
    Remove incomplete or corrupt screening files
    
    Args:
        output_dir: Directory containing screening results
        dry_run: If True, only show what would be deleted (default: True)
    
    Returns:
        List of files that were (or would be) deleted
    """
    validation = validate_screening_files(output_dir)
    
    if validation is None:
        return []
    
    to_delete = []
    
    # Add corrupt files
    for cf in validation['corrupt_files']:
        to_delete.append(cf['file'])
    
    # Find incomplete sets
    pickle_ts = set(v['timestamp'] for v in validation['pickle_files'])
    json_ts = set(v['timestamp'] for v in validation['json_files'])
    summary_ts = set(v['timestamp'] for v in validation['summary_files'])
    
    complete_sets = pickle_ts & json_ts & summary_ts
    all_ts = pickle_ts | json_ts | summary_ts
    incomplete_ts = all_ts - complete_sets
    
    # Add files from incomplete sets
    for ts in incomplete_ts:
        pattern_pickle = f"{output_dir}/screening_results.pkl"
        pattern_json = f"{output_dir}/screening_metadata.json"
        pattern_summary = f"{output_dir}/screening_summary.txt"
        
        for pattern in [pattern_pickle, pattern_json, pattern_summary]:
            if os.path.exists(pattern):
                to_delete.append(pattern)
    
    # Also check for .tmp files
    import glob
    tmp_files = glob.glob(f"{output_dir}/*.tmp")
    to_delete.extend(tmp_files)
    
    if not to_delete:
        print("✓ No files to clean!")
        return []
    
    print(f"\n{'='*70}")
    if dry_run:
        print(f"DRY RUN - Files that WOULD be deleted ({len(to_delete)}):")
    else:
        print(f"DELETING {len(to_delete)} files:")
    print(f"{'='*70}\n")
    
    for f in sorted(to_delete):
        size = os.path.getsize(f) / 1024 if os.path.exists(f) else 0
        print(f"  - {os.path.basename(f)} ({size:.1f} KB)")
        
        if not dry_run:
            try:
                os.remove(f)
                print(f"    ✓ Deleted")
            except Exception as e:
                print(f"     Error: {e}")
    
    if dry_run:
        print(f"\n⚠ This was a DRY RUN - no files were deleted")
        print(f"To actually delete files, run:")
        print(f"  clean_incomplete_files('{output_dir}', dry_run=False)")
    else:
        print(f"\n✓ Cleanup complete!")
    
    print(f"{'='*70}\n")
    
    return to_delete


def recover_screening_files(output_dir='Screening_Data', timestamp=None):
    """
    Recover missing JSON and summary files from valid pickle files
    
    This is useful when screening was interrupted and JSON/summary weren't saved,
    but the pickle file is complete.
    
    Args:
        output_dir: Directory containing screening results
        timestamp: Specific timestamp to recover, or None to recover all incomplete sets
    
    Returns:
        Number of timestamps recovered
    """
    import pickle
    import json
    import glob
    from datetime import datetime
    
    print(f"\n{'='*70}")
    print(f"Recovering Screening Files in: {output_dir}")
    print(f"{'='*70}\n")
    
    # Find pickle files
    if timestamp:
        pickle_files = [f"{output_dir}/screening_results.pkl"]
    else:
        pickle_files = glob.glob(f"{output_dir}/screening_results*.pkl")
    
    if not pickle_files:
        print(" No pickle files found")
        return 0
    
    recovered_count = 0
    
    for pickle_file in sorted(pickle_files):
        ts = os.path.basename(pickle_file).replace('screening_results', '').replace('.pkl', '')
        
        print(f"Processing timestamp: {ts}")
        
        # Load pickle file
        try:
            with open(pickle_file, 'rb') as f:
                screening_results = pickle.load(f)
            print(f"   Loaded pickle: {len(screening_results)} results")
        except Exception as e:
            print(f"   Failed to load pickle: {e}")
            continue
        
        # Regenerate JSON metadata
        json_file = f"{output_dir}/screening_metadata_{ts}.json"
        if os.path.exists(json_file):
            # Check if it's corrupt
            try:
                with open(json_file, 'r') as f:
                    json.load(f)
                print(f"   JSON already exists and is valid")
                json_recovered = False
            except:
                print(f"   JSON exists but is corrupt - regenerating...")
                json_recovered = True
        else:
            print(f"   JSON missing - generating...")
            json_recovered = True
        
        if json_recovered:
            try:
                def convert_to_json_serializable(obj):
                    """Convert numpy types to Python native types"""
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                         np.int16, np.int32, np.int64,
                                         np.uint8, np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_to_json_serializable(item) for item in obj]
                    else:
                        return obj
                
                metadata = []
                for r in screening_results:
                    # Exclude structure object
                    meta = {k: v for k, v in r.items() if k not in ['structure']}
                    # Convert all numpy types
                    meta = convert_to_json_serializable(meta)
                    metadata.append(meta)
                
                json_temp = f"{json_file}.tmp"
                with open(json_temp, 'w') as f:
                    json.dump(metadata, f, indent=2)
                os.replace(json_temp, json_file)
                print(f"   Regenerated JSON: {json_file}")
            except Exception as e:
                print(f"   Failed to generate JSON: {e}")
                if os.path.exists(json_temp):
                    os.remove(json_temp)
        
        # Regenerate summary
        summary_file = f"{output_dir}/screening_summary_{ts}.txt"
        if os.path.exists(summary_file):
            print(f"   Summary already exists")
            summary_recovered = False
        else:
            print(f"   Summary missing - generating...")
            summary_recovered = True
        
        if summary_recovered:
            try:
                with open(summary_file, 'w') as f:
                    f.write(f"Screening Results Summary\n")
                    f.write(f"{'='*70}\n")
                    f.write(f"Timestamp: {ts}\n")
                    f.write(f"Recovered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total configurations: {len(screening_results)}\n")
                    
                    converged = [r for r in screening_results if r.get('converged', False)]
                    f.write(f"Converged: {len(converged)}/{len(screening_results)}\n")
                    
                    site_types = sorted(set(r['site_type'] for r in screening_results))
                    f.write(f"Site types: {', '.join(site_types)}\n")
                    
                    f.write(f"\nEnergy Statistics (converged only):\n")
                    if converged:
                        energies = [r['adsorption_energy'] for r in converged]
                        f.write(f"  Min: {min(energies):.6f} eV\n")
                        f.write(f"  Max: {max(energies):.6f} eV\n")
                        f.write(f"  Mean: {np.mean(energies):.6f} eV\n")
                        f.write(f"  Std: {np.std(energies):.6f} eV\n")
                        f.write(f"  Range: {max(energies) - min(energies):.6f} eV\n")
                        
                        # Per-site statistics
                        f.write(f"\nPer-Site Statistics:\n")
                        for site_type in site_types:
                            site_results = [r for r in converged if r['site_type'] == site_type]
                            if site_results:
                                site_energies = [r['adsorption_energy'] for r in site_results]
                                f.write(f"  {site_type}:\n")
                                f.write(f"    Count: {len(site_results)}\n")
                                f.write(f"    Min: {min(site_energies):.6f} eV\n")
                                f.write(f"    Max: {max(site_energies):.6f} eV\n")
                                f.write(f"    Mean: {np.mean(site_energies):.6f} eV\n")
                
                print(f"   Generated summary: {summary_file}")
            except Exception as e:
                print(f"   Failed to generate summary: {e}")
        
        if json_recovered or summary_recovered:
            recovered_count += 1
            print(f"   Recovery complete for {ts}\n")
        else:
            print(f"   No recovery needed for {ts}\n")
    
    print(f"{'='*70}")
    print(f" Recovery complete!")
    print(f"  Timestamps processed: {len(pickle_files)}")
    print(f"  Timestamps recovered: {recovered_count}")
    print(f"{'='*70}\n")
    
    return recovered_count


def load_screening_results(filepath=None, output_dir='Screening_Data'):
    """
    Load previously saved screening results
    
    Args:
        filepath: Full path to pickle file. If None, finds most recent file in output_dir
        output_dir: Directory where screening results are saved (default: 'Screening_Data')
    
    Returns:
        screening_results: List of screening result dictionaries
    
    Usage:
        # Load most recent results
        screening_results = load_screening_results()
        
        # Load specific file
        screening_results = load_screening_results('Screening_Data/screening_results.pkl')
    """
    import pickle
    import glob
    
    # If no filepath provided, find most recent file
    if filepath is None:
        # Look for pickle files in output_dir
        pattern = f"{output_dir}/screening_results*.pkl"
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError(f"No screening results found in {output_dir}/")
        
        # Get most recent file
        filepath = max(files, key=os.path.getctime)
        print(f"Loading most recent screening results: {filepath}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Screening results file not found: {filepath}")
    
    # Load pickle file
    with open(filepath, 'rb') as f:
        screening_results = pickle.load(f)
    
    print(f"\n{'='*70}")
    print(f" Screening results loaded successfully!")
    print(f"  File: {filepath}")
    print(f"  Total configurations: {len(screening_results)}")
    
    # Count converged results
    converged_count = sum(1 for r in screening_results if r.get('converged', False))
    print(f"  Converged: {converged_count}/{len(screening_results)}")
    
    # Show site types
    site_types = set(r['site_type'] for r in screening_results)
    print(f"  Site types: {', '.join(sorted(site_types))}")
    print(f"{'='*70}\n")
    
    return screening_results


def list_screening_files(output_dir='Screening_Data'):
    """
    List all available screening result files
    
    Args:
        output_dir: Directory where screening results are saved
    
    Returns:
        List of (filepath, timestamp) tuples
    """
    import glob
    from datetime import datetime
    
    pattern = f"{output_dir}/screening_results*.pkl"
    files = glob.glob(pattern)
    
    if not files:
        print(f"No screening results found in {output_dir}/")
        return []
    
    print(f"\n{'='*70}")
    print(f"Available Screening Results Files ({len(files)} found)")
    print(f"{'='*70}\n")
    
    file_info = []
    for i, f in enumerate(sorted(files, reverse=True), 1):
        # Get file modification time
        mtime = os.path.getmtime(f)
        timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        # Get file size
        size_mb = os.path.getsize(f) / (1024 * 1024)
        
        print(f"{i}. {os.path.basename(f)}")
        print(f"   Created: {timestamp}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Path: {f}")
        print()
        
        file_info.append((f, timestamp))
    
    return file_info


def best_site_results(screening_results):
    df = pd.DataFrame(screening_results)
    df_converged = df[df['converged'] == True].copy()
    df_sorted = df_converged.sort_values('adsorption_energy')
    print(f"All site results:\n{df_sorted}")
    site_best = df_converged.loc[df_converged.groupby('site_type')['adsorption_energy'].idxmin()]
    print(f"Best site results:\n{site_best}")
    return df_sorted, site_best


def select_neb_endpoints_translation(site_best, screening_results):
    """
    Select NEB endpoints for PURE TRANSLATION between identical site types
    
    Strategy: Same site type, same height/rotation, highest vs lowest energy
    Example: fcc → fcc (different positions), NOT fcc → bridge
    """
    best_config = site_best.iloc[0]
    
    target_site_type = best_config['site_type'] #
    target_height = best_config['height']
    target_rotation = best_config['rotation']
    best_position = np.array(best_config['site_position'][:2])
    best_energy = best_config['total_energy']

  
    # Filter by site type + geometry (no distance constraint)
    df = pd.DataFrame(screening_results)
    matches = df[
        (df['site_type'] == target_site_type) &  # ← NEW: Same site type only!
        (df['height'] == target_height) & 
        (df['rotation'] == target_rotation) & 
        (df['converged'] == True)
    ].copy()
    
    if len(matches) < 2:
        print(f"\n ERROR: Not enough matching {target_site_type} sites!")
        print(f"   Need at least 2 sites of the same type for pure translation")
        return None, best_config
    
    # Calculate distances and energy differences
    matches['distance'] = matches['site_position'].apply(
        lambda pos: np.linalg.norm(np.array(pos[:2]) - best_position)
    )
    matches['dE_meV'] = (matches['total_energy'] - best_energy) * 1000
    
    # Remove the best site itself (distance ≈ 0)
    matches = matches[matches['distance'] > 0.1]
    
    # Sort by energy (descending)
    matches_sorted = matches.sort_values('total_energy', ascending=False)
    
    endpoint_initial = matches_sorted.iloc[0]  # Highest energy fcc
    endpoint_final = best_config                # Lowest energy fcc
    
    # Ensure correct ordering
    dE = endpoint_initial['total_energy'] - endpoint_final['total_energy']
    
    if dE < 0:
        print(f"\n  WARNING: Initial energy < Final energy!")
        print(f"  Swapping endpoints...")
        endpoint_initial, endpoint_final = endpoint_final, endpoint_initial
        dE = -dE
    
    return (endpoint_initial.to_dict() if hasattr(endpoint_initial, 'to_dict') else endpoint_initial,
            endpoint_final.to_dict() if hasattr(endpoint_final, 'to_dict') else endpoint_final)


def select_neb_endpoints_rotation(site_best, screening_results, rotation_angle_diff=120, use_translation_initial=True):
    """
    Select rotation NEB endpoints from screening results
    
    Strategy: Find structures at the SAME position as translation initial/final,
    then select two with DIFFERENT rotation angles.
    
    Args:
        site_best: DataFrame with best configuration info
        screening_results: List of screening result dicts
        rotation_angle_diff: Desired rotation angle difference (default 120°)
        use_translation_initial: If True, use high-energy translation initial position
                                 If False, use low-energy best position
    
    Returns:
        endpoint_rot_initial, endpoint_rot_final: Two endpoints with different rotations
    """
    best_config = site_best.iloc[0]
    target_site_type = best_config['site_type']
    target_height = best_config['height']
    target_rotation = best_config['rotation']
    
    # Filter: same site type, same height, converged
    df = pd.DataFrame(screening_results)
    matches = df[
        (df['site_type'] == target_site_type) &
        (df['height'] == target_height) &
        (df['rotation'] == target_rotation) &
        (df['converged'] == True)
    ].copy()
    
    if use_translation_initial:
        # Use the SAME position as translation initial (highest energy)
        print("\n Using TRANSLATION INITIAL position for rotation NEB")
        best_position = np.array(best_config['site_position'][:2])
        
        # Calculate distances and find highest energy at this site type/height
        matches['distance'] = matches['site_position'].apply(
            lambda pos: np.linalg.norm(np.array(pos[:2]) - best_position)
        )
        matches = matches[matches['distance'] > 0.1]  # Exclude best site itself
        
        if len(matches) < 1:
            print(f"\n ERROR: No other sites found for translation initial!")
            return None, None
        
        # Get highest energy configuration (translation initial)
        translation_initial = matches.sort_values('total_energy', ascending=False).iloc[0]
        reference_position = np.array(translation_initial['site_position'][:2])
        
        print(f"  Translation initial energy: {translation_initial['total_energy']:.6f} eV")
        print(f"  Position: {reference_position}")
        
    else:
        # Use best (lowest energy) position
        print("\n Using BEST position for rotation NEB")
        reference_position = np.array(best_config['site_position'][:2])
    
    # Now find ALL rotations at the reference position
    candidates = df[
        (df['site_type'] == target_site_type) &
        (df['height'] == target_height) &
        (df['converged'] == True)
    ].copy()
    
    # Calculate distance to reference position
    candidates['distance'] = candidates['site_position'].apply(
        lambda pos: np.linalg.norm(np.array(pos[:2]) - reference_position)
    )
    
    # Keep only same position (within 0.1 Å)
    same_position = candidates[candidates['distance'] < 0.1].copy()
    
    if len(same_position) < 2:
        print(f"\n ERROR: Found only {len(same_position)} structure(s) at reference position")
        print(f"   Need at least 2 different rotation angles")
        print(f"   Available rotations at this site:")
        for _, row in same_position.iterrows():
            print(f"     Rotation: {row['rotation']:.1f}°, Energy: {row['total_energy']:.6f} eV")
        return None, None
    
    print(f"\nFound {len(same_position)} structures at reference position with different rotations:")
    
    # Group by rotation angle
    rotations = same_position.groupby('rotation').first().sort_index()
    print(f"\nAvailable rotation angles: {sorted(same_position['rotation'].unique())}")
    
    if len(rotations) < 2:
        print(f"\n ERROR: Only found {len(rotations)} unique rotation angle(s)")
        print(f"   Cannot create rotation NEB with identical angles")
        return None, None
    
    # Strategy: Find two rotations separated by ~rotation_angle_diff degrees
    rotation_angles = sorted(same_position['rotation'].unique())
    
    # Find best pair with angle difference closest to target
    best_pair = None
    best_diff = 0
    
    for i, rot1 in enumerate(rotation_angles):
        for rot2 in rotation_angles[i+1:]:
            angle_diff = abs(rot2 - rot1)
            # Consider periodic boundary (0° = 360°)
            angle_diff = min(angle_diff, 360 - angle_diff)
            
            if best_pair is None or abs(angle_diff - rotation_angle_diff) < abs(best_diff - rotation_angle_diff):
                best_pair = (rot1, rot2)
                best_diff = angle_diff
    
    rot1, rot2 = best_pair
    print(f"\nSelected rotation angles:")
    print(f"  Initial: {rot1:.1f}°")
    print(f"  Final:   {rot2:.1f}°")
    print(f"  Angle difference: {best_diff:.1f}° (target was {rotation_angle_diff}°)")
    
    # Get the structures at these rotation angles
    endpoint_rot_initial = same_position[same_position['rotation'] == rot1].iloc[0]
    endpoint_rot_final = same_position[same_position['rotation'] == rot2].iloc[0]
    
    # Report energies
    E1 = endpoint_rot_initial['total_energy']
    E2 = endpoint_rot_final['total_energy']
    dE = abs(E2 - E1)
    
    if dE < 1e-6:
        print(f"\n  WARNING: Endpoints have IDENTICAL energies!")
        print(f"   This will cause NEB to fail. This shouldn't happen for different rotations.")
    elif dE < 1e-4:
        print(f"\n  Note: Very small energy difference ({dE*1000:.3f} meV)")
        print(f"   Rotation barrier will be tiny (CH4 may be nearly symmetric)")
    else:
        print(f"\n GOOD: Significant energy difference found ({dE*1000:.3f} meV)")
    
    print(f"✓ PURE ROTATION confirmed (same position, different angles)")
    print(f"{'='*70}")
    
    rot_initial = endpoint_rot_initial.to_dict() if hasattr(endpoint_rot_initial, 'to_dict') else endpoint_rot_initial
    rot_final = endpoint_rot_final.to_dict() if hasattr(endpoint_rot_final, 'to_dict') else endpoint_rot_final
    
    return rot_initial, rot_final


def _verify_constraints(atoms1, atoms2):
    """Verify both structures have identical constraints"""
    if len(atoms1) != len(atoms2):
        raise ValueError("Structures have different number of atoms!")
    
    if len(atoms1.constraints) != len(atoms2.constraints):
        raise ValueError("Structures have different constraints!")
    
    if atoms1.constraints:
        fixed1 = set(atoms1.constraints[0].index)
        fixed2 = set(atoms2.constraints[0].index)
        
        if fixed1 != fixed2:
            raise ValueError("Different atoms are fixed in the two structures!")
        
        print(f"✓ Constraint verification passed: {len(fixed1)} atoms fixed")
    else:
        print("WARNING: No constraints on structures")



def prepare_neb_calculation(endpoint1, endpoint2, n_images=10, barrier_type='translation'):
    """
    Prepare and run NEB calculation
    
    Args:
        endpoint1, endpoint2: Can be either:
            - Dictionary with 'structure' and 'structure_file' keys (from screening)
            - ASE Atoms objects directly
        n_images: Number of intermediate images
        barrier_type: 'translation' or 'rotation'
    """
    # Handle both dict and Atoms input
    if isinstance(endpoint1, dict):
        # It's a screening result dict
        if 'structure' in endpoint1:
            initial = endpoint1['structure'].copy()
        else:
            initial = read(endpoint1['structure_file'])
    else:
        # It's already an Atoms object
        initial = endpoint1.copy()
    
    if isinstance(endpoint2, dict):
        # It's a screening result dict
        if 'structure' in endpoint2:
            final = endpoint2['structure'].copy()
        else:
            final = read(endpoint2['structure_file'])
    else:
        # It's already an Atoms object
        final = endpoint2.copy()
    
    # Verify constraints
    _verify_constraints(initial, final)
    
    # Ensure endpoints have calculators - SEPARATE instances for each!
    print("\nSetting up calculators for NEB...")
    
    # Create separate calculator for initial endpoint
    #predictor_initial = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
    # calc_initial = FAIRChemCalculator(predictor_initial, task_name="oc20")
    calc_initial = mace_mp(model="medium", device='cpu')
    initial.calc = calc_initial
    
    # Create separate calculator for final endpoint
    #predictor_final = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
    #calc_final = FAIRChemCalculator(predictor_final, task_name="oc20")
    calc_final = mace_mp(model="medium", device='cpu')
    final.calc = calc_final
    
    # Calculate endpoint energies (important for barrier calculation later)
    print(f"Initial structure energy: {initial.get_potential_energy():.6f} eV")
    print(f"Final structure energy: {final.get_potential_energy():.6f} eV")
    
    # Create image list
    # images = [initial]
    # for i in range(n_images):
    #     image = initial.copy()
    #     images.append(image)
    # images.append(final)

    # Correct ASE NEB image construction # Azeez
    images = [initial.copy()]
    for i in range(n_images):
        images.append(initial.copy())
    images.append(final.copy())
    
    print(f"Created {len(images)} images (including 2 endpoints)")
    
    # Setup NEB with climbing image
    neb = NEB(images, climb=True, allow_shared_calculator=False)
    neb.interpolate()
    
    # Assign calculators to all intermediate images (not endpoints)
    for i, image in enumerate(images[1:-1], 1):  # Skip first and last
        #predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
        #calc_individual = FAIRChemCalculator(predictor, task_name="oc20")
        calc_individual = mace_mp(model="medium", device='cpu')
        image.calc = calc_individual
    
    
    # Optimize
    slab_dir = 'NEB_Results'
    os.makedirs(slab_dir, exist_ok=True)
    traj_file = f"{slab_dir}/neb_{barrier_type}.traj"
    log_file = f"{slab_dir}/neb_{barrier_type}.log"
    
    print(f"\nStarting NEB optimization...")
    print(f"  Trajectory: {traj_file}")
    print(f"  Log file: {log_file}")
    
    optimizer = FIRE(neb, trajectory=traj_file, logfile=log_file)
    optimizer.run(fmax=0.05)
    
    print("\n NEB optimization completed!")
    
    # Analyze using NEBTools
    from ase.mep import NEBTools  # Fixed import path
    nebtools = NEBTools(images)
    
    print("\nCalculating barriers...")
    
    # Get barrier and reaction energy using interpolated fit (recommended)
    # Returns: (forward_barrier, delta_E)
    try:
        barrier_fwd_fit, delta_E_fit = nebtools.get_barrier(fit=True, raw=False)
        print(f"  Forward barrier (fitted): {barrier_fwd_fit:.6f} eV ({barrier_fwd_fit*1000:.3f} meV)")
        print(f"  Reaction energy (ΔE): {delta_E_fit:.6f} eV")
    except Exception as e:
        print(f" Warning: Could not fit barrier curve: {e}")
        barrier_fwd_fit = None
        delta_E_fit = None
    
    # Get raw transition state energy (absolute energy, not relative)
    try:
        E_ts_abs, _ = nebtools.get_barrier(fit=True, raw=True)
        print(f"  TS absolute energy: {E_ts_abs:.6f} eV")
    except Exception as e:
        print(f"   Warning: Could not get TS energy: {e}")
        E_ts_abs = None
    
    # Plot band structure
    nebtools.plot_bands()
    
    # Find and save saddle point
    energies = []
    for i, img in enumerate(images):
        try:
            E = img.get_potential_energy()
            energies.append(E)
        except:
            energies.append(np.nan)
    
    if not all(np.isnan(energies)):
        saddle_idx = np.nanargmax(energies)
        print(f"\nSaddle point: image {saddle_idx}/{len(images)-1}")
        
        saddle_file = f"{slab_dir}/saddle_{barrier_type}.traj"
        write(saddle_file, images[saddle_idx])
        print(f"Saddle point saved: {saddle_file}")
    else:
        saddle_file = None
        saddle_idx = None
        print("\n  Warning: Could not determine saddle point")
    
    print(f"\nTrajectory saved: {traj_file}")
    
    # Save results
    result = {
        'barrier_type': barrier_type,
        'forward_barrier_fit': barrier_fwd_fit,      # A → TS (use for partition function)
        'delta_E': delta_E_fit,                      # E_B - E_A
        'transition_state_energy': E_ts_abs,         # Absolute TS energy
        'trajectory': traj_file,
        'saddle_file': saddle_file,
        'saddle_index': saddle_idx,
        'n_images': len(images)
    }

    # Save summary to file
    save_neb_summary(result, summary_file=f'{slab_dir}/neb_summary.txt', append=True)
    
    return images, result

def check_neb_endpoints(endpoint1, endpoint2, name="endpoints"):
    """
    Validate NEB endpoints before running calculation
    
    Args:
        endpoint1: First endpoint Atoms object
        endpoint2: Second endpoint Atoms object
        name: Descriptive name for this pair
    
    Returns:
        dict with validation results
    """
    
    results = {
        'valid': True,
        'issues': [],
        'warnings': []
    }
    
    # Check 1: Same number of atoms
    if len(endpoint1) != len(endpoint2):
        results['valid'] = False
        results['issues'].append(f"Different number of atoms: {len(endpoint1)} vs {len(endpoint2)}")
        print(f" CRITICAL: Different number of atoms!")
        return results
    
    print(f"✓ Same number of atoms: {len(endpoint1)}")
    
    # Check 2: Same atomic species in same order
    symbols1 = endpoint1.get_chemical_symbols()
    symbols2 = endpoint2.get_chemical_symbols()
    if symbols1 != symbols2:
        results['valid'] = False
        results['issues'].append("Different atomic species or order")
        print(f" CRITICAL: Atomic species mismatch!")
        return results
    
    print(f"✓ Same atomic species and order")
    
    # Check 3: Check for NaN or Inf in positions
    pos1 = endpoint1.get_positions()
    pos2 = endpoint2.get_positions()
    
    if np.any(np.isnan(pos1)) or np.any(np.isinf(pos1)):
        results['valid'] = False
        results['issues'].append("Endpoint 1 has NaN or Inf positions")
        print(f" CRITICAL: Endpoint 1 has invalid positions (NaN/Inf)!")
        return results
    
    if np.any(np.isnan(pos2)) or np.any(np.isinf(pos2)):
        results['valid'] = False
        results['issues'].append("Endpoint 2 has NaN or Inf positions")
        print(f" CRITICAL: Endpoint 2 has invalid positions (NaN/Inf)!")
        return results
    
    print(f" No NaN or Inf in positions")
    
    # Check 4: Calculate RMSD between endpoints
    rmsd = np.sqrt(np.mean((pos1 - pos2)**2))
    max_displacement = np.max(np.linalg.norm(pos1 - pos2, axis=1))
    
    print(f"\nStructural Differences:")
    print(f"  RMSD: {rmsd:.4f} Å")
    print(f"  Max atom displacement: {max_displacement:.4f} Å")
    
    if rmsd < 0.01:
        results['warnings'].append(f"Very small RMSD ({rmsd:.4f} Å) - endpoints might be too similar")
        print(f"   WARNING: Very small RMSD - NEB might not be meaningful")
    elif rmsd < 0.1:
        results['warnings'].append(f"Small RMSD ({rmsd:.4f} Å) - barrier might be very small")
        print(f"   WARNING: Small RMSD - expect small barrier")
    elif rmsd > 10.0:
        results['warnings'].append(f"Large RMSD ({rmsd:.4f} Å) - may need more images")
        print(f"   WARNING: Large RMSD - consider using more NEB images")
    else:
        print(f"   RMSD looks reasonable for NEB")
    
    # Check 5: Check cell similarity
    cell1 = endpoint1.get_cell()
    cell2 = endpoint2.get_cell()
    cell_diff = np.max(np.abs(cell1 - cell2))
    
    print(f"\nCell:")
    print(f"  Cell difference: {cell_diff:.6f} Å")
    
    if cell_diff > 0.01:
        results['warnings'].append(f"Cells differ ({cell_diff:.4f} Å)")
        print(f"   WARNING: Different unit cells - this might cause issues")
    else:
        print(f"   Cells are the same")
    
    # Check 6: Energy check (if calculators present)
    try:
        E1 = endpoint1.get_potential_energy()
        print(f"\nEndpoint 1 energy: {E1:.6f} eV")
    except:
        print(f"\n Endpoint 1 has no calculated energy")
        results['warnings'].append("Endpoint 1 missing energy")
    
    try:
        E2 = endpoint2.get_potential_energy()
        print(f"Endpoint 2 energy: {E2:.6f} eV")
    except:
        print(f" Endpoint 2 has no calculated energy")
        results['warnings'].append("Endpoint 2 missing energy")
    
    try:
        dE = abs(E2 - E1)
        print(f"Energy difference: {dE:.6f} eV ({dE*1000:.3f} meV)")
        
        if dE < 0.001:  # 1 meV
            results['warnings'].append(f"Very small energy difference ({dE*1000:.3f} meV)")
            print(f"   WARNING: Very small energy difference - barrier will be tiny")
    except:
        pass
    
    # Check 7: Find which atoms moved the most
    displacements = np.linalg.norm(pos1 - pos2, axis=1)
    moving_atoms = np.where(displacements > 0.1)[0]
    
    print(f"\nMoving atoms (displacement > 0.1 Å): {len(moving_atoms)}")
    if len(moving_atoms) > 0:
        top_movers = np.argsort(displacements)[-5:][::-1]  # Top 5
        print(f"  Top movers:")
        for idx in top_movers:
            if displacements[idx] > 0.01:
                print(f"    Atom {idx} ({symbols1[idx]}): {displacements[idx]:.4f} Å")
    else:
        results['warnings'].append("No atoms moving significantly")
        print(f"   WARNING: No atoms move > 0.1 Å")
    
    # Summary
    if results['valid'] and len(results['issues']) == 0:
        print(f"✓ Endpoints are VALID for NEB")
        if len(results['warnings']) > 0:
            print(f" {len(results['warnings'])} warning(s) - see details above")
    else:
        print(f" Endpoints are INVALID for NEB")
        print(f"   Issues: {len(results['issues'])}")
    
    results['rmsd'] = rmsd
    results['max_displacement'] = max_displacement
    results['moving_atoms'] = len(moving_atoms)
    
    return results



def save_neb_summary(result, summary_file='NEB_Results/neb_summary.txt', append=True):
    """
    Save NEB results to a human-readable summary file
    
    Args:
        result: Dict returned from prepare_neb_calculation() or prepare_neb_calculation_emt()
        summary_file: Path to summary file
        append: If True, append to existing file. If False, overwrite.
    """
    import os
    from datetime import datetime
    
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    mode = 'a' if append else 'w'
    
    with open(summary_file, mode) as f:
        # Header with timestamp
        f.write("="*80 + " \n")
        f.write(f"NEB Calculation Results \n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n")
        f.write("="*80 + " \n \n")
        
        # Barrier type
        barrier_type = result.get('barrier_type', 'unknown')
        f.write(f"Barrier Type: {barrier_type.upper()} \n")
        f.write("-"*80 + "\n \n")
        
        # Main results
        f.write("Main Results: \n")
        
        if result.get('forward_barrier_fit') is not None:
            barrier_eV = result['forward_barrier_fit']
            barrier_meV = barrier_eV * 1000
            barrier_kJ_mol = barrier_eV * 96.485  # Convert eV to kJ/mol
            f.write(f"  Forward Barrier:        {barrier_eV:.6f} eV \n")
            f.write(f"                          {barrier_meV:.3f} meV \n")
            f.write(f"                          {barrier_kJ_mol:.3f} kJ/mol \n")
        else:
            f.write(f"  Forward Barrier:        N/A\n")
        
        if result.get('delta_E') is not None:
            delta_E_eV = result['delta_E']
            delta_E_meV = delta_E_eV * 1000
            f.write(f"\n  Reaction Energy (ΔE):   {delta_E_eV:.6f} eV \n")
            f.write(f"                          {delta_E_meV:.3f} meV \n")
        
        if result.get('transition_state_energy') is not None:
            f.write(f" \n  TS Absolute Energy:     {result['transition_state_energy']:.6f} eV \n")
        
        # Calculation details
        f.write(f"\n Calculation Details: \n")
        f.write(f"  Number of images:       {result.get('n_images', 'N/A')} \n")
        f.write(f"  Saddle point index:     {result.get('saddle_index', 'N/A')} \n")
        
        calculator = result.get('calculator', 'FAIRChem')
        f.write(f"  Calculator:             {calculator} \n")
        
        # Files
        f.write(f"\n Output Files: \n")
        if result.get('trajectory'):
            f.write(f"  Trajectory:             {result['trajectory']} \n")
        if result.get('saddle_file'):
            f.write(f"  Saddle point:           {result['saddle_file']} \n")
        if result.get('plot_file'):
            f.write(f"  Band plot:              {result['plot_file']} \n")
        
        # Partition function values
        f.write(f"\n For Hill's Hindered Translator/Rotor:\n")
        if barrier_type == 'translation':
            f.write(f"  W_x (translation barrier) = {result.get('forward_barrier_fit', 0):.6f} eV \n")
            f.write(f"                            = {result.get('forward_barrier_fit', 0)*1000:.3f} meV \n")
        elif barrier_type == 'rotation':
            f.write(f"  W_r (rotation barrier)    = {result.get('forward_barrier_fit', 0):.6f} eV \n")
            f.write(f"                            = {result.get('forward_barrier_fit', 0)*1000:.3f} meV \n")
        
        f.write(f"\n" + "="*80 + "\n \n")
    
    print(f"\n NEB summary appended to: {summary_file}")
    return summary_file