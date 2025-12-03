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
#!/usr/bin/env python3

vacuum = 20 # Angstrom
size = (3, 3, 3) # slab size
lattice_constant_vdW = 3.92 # Angstrom
lattice_constant = 3.97 # Angstrom
surface_type = 'fcc111'
metal = 'Pt'

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
calc = FAIRChemCalculator(predictor, task_name="oc20")

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


def init_molecule(mol):
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


def create_structure(slab, opt_adsorbate, site, bond_params=None, height=None, rotation=None, rotation_center='site', binding_atom_idx=None, hookean_rt=None, hookean_k=None,
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


def clean_slab(metal='Pt', size=(3, 3, 4), lattice_constant=3.97, vacuum=20, slab_dir = 'Slab'):
    '''
    Create a clean slab
    '''
    slab = fcc111(metal, size=size, a=lattice_constant, vacuum=vacuum)
    slab.set_pbc(True)
    slab.calc = ase_calculator
    E_clean = slab.get_potential_energy()
    print(f"Clean slab energy: {E_clean:.3f} eV\n")
    # create a directory for the slab file if it does not exist
    os.makedirs(slab_dir, exist_ok=True)
    # Write slab to file
    slab.write(f'{slab_dir}/slab_init.xyz')
    return slab


def opt_slab(metal='Pt', size=(3, 3, 4), lattice_constant=3.97, vacuum=20, slab_dir = 'Slab'):
    slab = clean_slab(metal=metal, size=size, lattice_constant=lattice_constant, vacuum=vacuum, slab_dir = slab_dir)
    opt = BFGS(slab, trajectory=None)
    opt.run(fmax=0.05)
    slab.calc = ase_calculator
    os.makedirs(slab_dir, exist_ok=True)
    # Write slab to file
    slab.write(f'{slab_dir}/slab.xyz')
    return slab


def site_screening(slab, ads, center_xy='site', use_all_sites=True,
                   save_results=True, workdir="Screening_Data", resume=True):
    """
    Multiprocessing-safe site screening function with checkpoint/resume support.
    """
    log_dir = os.path.join(workdir, "logs")
    structures_dir = os.path.join(workdir, "structures")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(structures_dir, exist_ok=True)

    # Load existing results if resuming
    results_pkl = os.path.join(workdir, "screening_results.pkl")
    if resume and os.path.exists(results_pkl):
        import pickle
        with open(results_pkl, "rb") as f:
            screening_results = pickle.load(f)
        completed = {(r["site_index"], r["height"], r["rotation"]) for r in screening_results}
        print(f"Resuming: found {len(completed)} completed configurations")
    else:
        screening_results = []
        completed = set()

    def calc():
        pred = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
        return FAIRChemCalculator(pred, task_name="oc20")

    local_calc = calc()
    slab = opt_slab(slab_dir=workdir)
    slab.calc = local_calc
    clean_slab_energy = slab.get_potential_energy()

    ads = opt_molecule(ads)
    ads.calc = local_calc
    ads_energy = ads.get_potential_energy()

    all_sites, site_density, unique_site_lists, unique_site_pair_lists, \
        single_bond_params, double_bond_params = adsorption_sites_and_unique_placements(slab)

    heights = np.arange(1.5, 3.5, 0.5)
    rotations = np.arange(0, 360, 30)
    sites_to_screen = all_sites if use_all_sites else [lst[0] for lst in unique_site_lists]

    for site_idx, site in enumerate(sites_to_screen):
        bond_params = [{
            "site_pos": site["position"],
            "ind": None,
            "k": 100.0,
            "deq": 0.0
        }]

        for height in heights:
            for rot in rotations:
                # Skip if already completed
                if (site_idx, float(height), float(rot)) in completed:
                    print(f"Skipping site {site_idx}, h={height}, r={rot} (already done)")
                    continue

                test_slab = create_structure(
                    slab, ads, site, bond_params,
                    height, rotation=rot,
                    rotation_center=center_xy,
                    binding_atom_idx=None
                )
                test_slab.calc = local_calc

                log_file = os.path.join(log_dir, f"site{site_idx}_{site['site']}_h{height}_r{rot}.log")
                opt = BFGS(test_slab, logfile=log_file)

                try:
                    opt.run(fmax=0.05)
                    converged = True
                except:
                    converged = False

                if not converged:
                    continue

                E_total = test_slab.get_potential_energy()
                E_ads = E_total - clean_slab_energy - ads_energy

                struct_file = os.path.join(structures_dir, f"{site['site']}_h{height}_r{rot}.xyz")
                test_slab.write(struct_file)

                result = {
                    "site_index": site_idx,
                    "site_type": site["site"],
                    "site_position": site["position"],
                    "height": float(height),
                    "rotation": float(rot),
                    "adsorption_energy": float(E_ads),
                    "total_energy": float(E_total),
                    "structure_file": struct_file,
                    "converged": converged
                }
                screening_results.append(result)
                print(f"Site {site_idx} ({site['site']}), h={height}, r={rot} → E_ads={E_ads:.3f} eV")

                # Incremental save after each successful calculation
                if save_results:
                    import pickle
                    with open(results_pkl, "wb") as f:
                        pickle.dump(screening_results, f)

    # Final save (JSON for human readability)
    if save_results:
        import json
        json_file = os.path.join(workdir, "screening_metadata.json")
        with open(json_file, "w") as f:
            json.dump(screening_results, f, indent=2)
        print(f"✓ Saved {len(screening_results)} screening results in {workdir}")

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
    json_files = glob.glob(f"{output_dir}/screening_metadata*.json")
    for jf in sorted(json_files):
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            size_kb = os.path.getsize(jf) / 1024
            timestamp = os.path.basename(jf).replace('screening_metadata', '').replace('.json', '')
            
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
    summary_files = glob.glob(f"{output_dir}/screening_summary*.txt")
    for sf in sorted(summary_files):
        timestamp = os.path.basename(sf).replace('screening_summary', '').replace('.txt', '')
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


def select_neb_endpoints_translation(site_best, screening_results, 
                                      min_distance=2, max_distance=4.3):
    """
    Select NEB endpoints for translation between NEIGHBORING sites of same type.
    
    Parameters:
        min_distance: Minimum separation to avoid same-site duplicates (Å)
        max_distance: Maximum separation for nearest-neighbor hop (Å)
                      Typical: ~2.8 Å for fcc-fcc on Pt(111)
    """
    
    best_config = site_best.iloc[0]
    target_site_type = best_config['site_type']
    best_position = np.array(best_config['site_position'][:2])
    
    df = pd.DataFrame(screening_results)
    
    # Filter for same site type and converged
    matches = df[
        (df['site_type'] == target_site_type) &
        (df['converged'] == True)
    ].copy()
    
    if len(matches) < 2:
        print(f"\nERROR: Not enough {target_site_type} sites!")
        return None, None
    
    # Calculate distances from best site
    matches['distance'] = matches['site_position'].apply(
        lambda pos: np.linalg.norm(np.array(pos[:2]) - best_position)
    )
    
    # Find nearest neighbor (not self, within reasonable range)
    neighbors = matches[
        (matches['distance'] > min_distance) &
        (matches['distance'] < max_distance)
    ].copy()
    
    if len(neighbors) == 0:
        print(f"\nERROR: No neighboring {target_site_type} sites found!")
        print(f"  Distance range searched: {min_distance}–{max_distance} Å")
        print(f"  Available distances: {sorted(matches['distance'].unique())}")
        return None, None
    
    # Pick the closest neighbor
    neighbors_sorted = neighbors.sort_values('distance')
    neighbor = neighbors_sorted.iloc[0]
    
    print(f"\nSelected translation pathway:")
    print(f"  Site type: {target_site_type} → {target_site_type}")
    print(f"  Distance: {neighbor['distance']:.2f} Å")
    print(f"  Energy difference: {abs(neighbor['total_energy'] - best_config['total_energy'])*1000:.1f} meV")
    
    # Use structures with same height/rotation for cleaner path
    # Or let NEB find the optimal path
    endpoint1 = best_config.to_dict() if hasattr(best_config, 'to_dict') else dict(best_config)
    endpoint2 = neighbor.to_dict() if hasattr(neighbor, 'to_dict') else dict(neighbor)
    
    return endpoint1, endpoint2


def select_neb_endpoints_rotation(site_best, screening_results, 
                                   rotation_angle_diff=120):
    """
    Select rotation NEB endpoints: same position, different rotation angles.
    Only meaningful for asymmetric adsorbates (CH3, NH2, etc.), not CO.
    """
    best_config = site_best.iloc[0]
    target_site_type = best_config['site_type']
    target_height = best_config['height']
    reference_position = np.array(best_config['site_position'][:2])
    
    df = pd.DataFrame(screening_results)
    
    # Find all rotations at the same position
    candidates = df[
        (df['site_type'] == target_site_type) &
        (df['height'] == target_height) &
        (df['converged'] == True)
    ].copy()
    
    candidates['distance'] = candidates['site_position'].apply(
        lambda pos: np.linalg.norm(np.array(pos[:2]) - reference_position)
    )
    
    # Keep only structures at the same position
    same_position = candidates[candidates['distance'] < 0.1].copy()
    
    if len(same_position) < 2:
        print(f"ERROR: Found only {len(same_position)} structure(s) at reference position")
        return None, None
    
    rotation_angles = sorted(same_position['rotation'].unique())
    print(f"Available rotations: {rotation_angles}")
    
    if len(rotation_angles) < 2:
        print("ERROR: Need at least 2 different rotation angles")
        return None, None
    
    # Find pair closest to target angle difference
    best_pair = None
    best_match = float('inf')
    
    for i, rot1 in enumerate(rotation_angles):
        for rot2 in rotation_angles[i+1:]:
            diff = min(abs(rot2 - rot1), 360 - abs(rot2 - rot1))
            if abs(diff - rotation_angle_diff) < best_match:
                best_match = abs(diff - rotation_angle_diff)
                best_pair = (rot1, rot2)
    
    rot1, rot2 = best_pair
    actual_diff = min(abs(rot2 - rot1), 360 - abs(rot2 - rot1))
    
    print(f"\nSelected: {rot1:.0f}° → {rot2:.0f}° (Δ = {actual_diff:.0f}°)")
    
    ep1 = same_position[same_position['rotation'] == rot1].iloc[0]
    ep2 = same_position[same_position['rotation'] == rot2].iloc[0]
    
    dE = abs(ep2['total_energy'] - ep1['total_energy']) * 1000
    print(f"Energy difference: {dE:.1f} meV")
    
    if dE < 1:
        print("WARNING: Very small energy difference — check if molecule has rotational symmetry")
    
    return ep1.to_dict(), ep2.to_dict()


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


def prepare_neb_calculation(endpoint1, endpoint2, n_images=10,
                             barrier_type='translation', workdir="NEB"):
    os.makedirs(workdir, exist_ok=True)

    def load_structure(ep):
        if isinstance(ep, dict):
            if 'structure' in ep:
                return ep['structure'].copy()
            return read(ep['structure_file'])
        return ep.copy()

    initial = load_structure(endpoint1)
    final = load_structure(endpoint2)
    _verify_constraints(initial, final)

    # Check endpoint difference before proceeding
    disp = np.abs(final.positions - initial.positions).max()
    print(f"Max displacement between endpoints: {disp:.3f} Å")
    if disp < 0.00001:
        raise ValueError("Endpoints too similar for meaningful NEB")

    print("\nSetting up NEB calculators...")

    def fresh_calc():
        pred = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
        return FAIRChemCalculator(pred, task_name="oc20")

    initial.calc = fresh_calc()
    final.calc = fresh_calc()

    E_init = initial.get_potential_energy()
    E_final = final.get_potential_energy()
    print(f"Initial energy: {E_init:.6f} eV")
    print(f"Final energy:   {E_final:.6f} eV")
    print(f"Energy difference: {abs(E_final - E_init)*1000:.3f} meV")

    # Create images
    images = [initial]
    for _ in range(n_images):
        img = initial.copy()
        img.calc = fresh_calc()
        images.append(img)
    final.calc = fresh_calc()
    images.append(final)

    # Stage 1: Relax without climbing image
    neb = NEB(images, climb=False, allow_shared_calculator=False)
    neb.interpolate('idpp')  # Better interpolation for surfaces

    # Verify interpolation didn't produce nan
    for i, img in enumerate(images):
        if np.any(np.isnan(img.positions)):
            raise ValueError(f"NaN positions in image {i} after interpolation")

    traj_file = os.path.join(workdir, f"neb_{barrier_type}.traj")
    log_file = os.path.join(workdir, f"neb_{barrier_type}.log")

    print(f"Running NEB (stage 1: no climb)...")
    opt = FIRE(neb, trajectory=traj_file, logfile=log_file, maxstep=0.1)
    opt.run(fmax=0.1, steps=200)

    # Stage 2: Turn on climbing image
    print("Running NEB (stage 2: climbing image)...")
    neb.climb = True
    opt.run(fmax=0.05, steps=500)

    print("\nNEB completed!")

    # Analysis (same as before)
    from ase.mep import NEBTools
    nebtools = NEBTools(images)
    
    try:
        barrier_fwd, delta_E = nebtools.get_barrier(fit=True, raw=False)
    except:
        barrier_fwd = delta_E = None

    try:
        E_ts_abs, _ = nebtools.get_barrier(fit=True, raw=True)
    except:
        E_ts_abs = None

    energies = [img.get_potential_energy() for img in images]
    saddle_idx = int(np.argmax(energies))
    saddle_file = os.path.join(workdir, f"saddle_{barrier_type}.traj")
    write(saddle_file, images[saddle_idx])

    result = {
        "barrier_type": barrier_type,
        "forward_barrier_fit": barrier_fwd,
        "delta_E": delta_E,
        "transition_state_energy": E_ts_abs,
        "trajectory": traj_file,
        "saddle_file": saddle_file,
        "saddle_index": saddle_idx,
        "n_images": len(images)
    }

    summary_file = os.path.join(workdir, "neb_summary.json")
    with open(summary_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Summary saved to {summary_file}")
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



def save_neb_summary(result, summary_file, append=False):
    """
    Multiprocessing-safe NEB summary writer.
    Always writes to an adsorbate-specific directory.
    """

    import os
    from datetime import datetime

    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    mode = 'a' if append else 'w'

    with open(summary_file, mode) as f:
        # Header
        f.write("="*80 + "\n")
        f.write(f"NEB Calculation Summary\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # Barrier type
        barrier_type = result.get('barrier_type', 'unknown')
        f.write(f"Barrier Type: {barrier_type.upper()}\n")
        f.write("-"*80 + "\n\n")

        # Main results
        barrier = result.get('forward_barrier_fit')
        deltaE = result.get('delta_E')
        Ets    = result.get('transition_state_energy')

        f.write("Main Results:\n")
        if barrier is not None:
            f.write(f"  Forward Barrier:  {barrier:.6f} eV ({barrier*1000:.3f} meV)\n")
            f.write(f"                     {barrier*96.485:.3f} kJ/mol\n")
        else:
            f.write("  Forward Barrier:  N/A\n")

        if deltaE is not None:
            f.write(f"\n  Reaction Energy ΔE: {deltaE:.6f} eV ({deltaE*1000:.3f} meV)\n")

        if Ets is not None:
            f.write(f"  TS Absolute Energy: {Ets:.6f} eV\n")

        # Calculation details
        f.write("\nCalculation Details:\n")
        f.write(f"  Number of images:   {result.get('n_images', 'N/A')}\n")
        f.write(f"  Saddle point index: {result.get('saddle_index', 'N/A')}\n")
        f.write(f"  Calculator:         {result.get('calculator', 'FAIRChem')}\n")

        # Output files
        f.write("\nOutput Files:\n")
        for key in ["trajectory", "saddle_file", "plot_file"]:
            if result.get(key):
                f.write(f"  {key}: {result[key]}\n")

        # Partition function / barrier parameter
        f.write("\nPartition Function Parameters:\n")
        if barrier_type == "translation":
            f.write(f"  W_x = {barrier:.6f} eV  ({barrier*1000:.3f} meV)\n")
        elif barrier_type == "rotation":
            f.write(f"  W_r = {barrier:.6f} eV  ({barrier*1000:.3f} meV)\n")

        f.write("\n" + "="*80 + "\n")

    print(f"✓ NEB summary saved: {summary_file}")
    return summary_file




# This function should be used for site screening when the site screening was interrupted

import os, glob, re, pickle, json
import numpy as np
from ase.io import read

def recover_and_resume_screening(slab, ads, center_xy='site', use_all_sites=True,
                                  save_results=True, workdir="Screening_Data"):
    """
    Recover completed results from log/xyz files, then continue screening.
    """
    log_dir = os.path.join(workdir, "logs")
    structures_dir = os.path.join(workdir, "structures")
    
    # Set up calculator
    pred = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
    local_calc = FAIRChemCalculator(pred, task_name="oc20")
    
    # Reference energies (needed regardless)
    slab = opt_slab(slab_dir=workdir)
    slab.calc = local_calc
    clean_slab_energy = slab.get_potential_energy()
    
    ads = opt_molecule(ads)
    ads.calc = local_calc
    ads_energy = ads.get_potential_energy()
    
    # Get site info
    all_sites, site_density, unique_site_lists, unique_site_pair_lists, \
        single_bond_params, double_bond_params = adsorption_sites_and_unique_placements(slab)
    
    # === RECOVERY PHASE ===
    pattern = re.compile(r'site(\d+)_(\w+)_h([\d.]+)_r([\d.]+)\.log')
    screening_results = []
    completed = set()
    
    for log_file in glob.glob(os.path.join(log_dir, "*.log")):
        match = pattern.search(os.path.basename(log_file))
        if not match:
            continue
        
        site_idx = int(match.group(1))
        site_type = match.group(2)
        height = float(match.group(3))
        rot = float(match.group(4))
        
        struct_file = os.path.join(structures_dir, f"{site_type}_h{height}_r{rot}.xyz")
        if not os.path.exists(struct_file):
            continue
        
        # Parse energy from log
        E_total = None
        with open(log_file, 'r') as f:
            for line in reversed(f.readlines()):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        E_total = float(parts[2])
                        break
                    except ValueError:
                        continue
        
        if E_total is None:
            continue
        
        E_ads = E_total - clean_slab_energy - ads_energy
        
        screening_results.append({
            "site_index": site_idx,
            "site_type": site_type,
            "site_position": list(all_sites[site_idx]["position"]) if site_idx < len(all_sites) else None,
            "height": height,
            "rotation": rot,
            "adsorption_energy": float(E_ads),
            "total_energy": float(E_total),
            "structure_file": struct_file,
            "converged": True
        })
        completed.add((site_idx, height, rot))
    
    print(f"Recovered {len(screening_results)} completed configurations")
    
    # === RESUME PHASE ===
    heights = np.arange(1.5, 3.5, 0.5)
    rotations = np.arange(0, 360, 30)
    sites_to_screen = all_sites if use_all_sites else [lst[0] for lst in unique_site_lists]
    
    for site_idx, site in enumerate(sites_to_screen):
        bond_params = [{
            "site_pos": site["position"],
            "ind": None,
            "k": 100.0,
            "deq": 0.0
        }]
        
        for height in heights:
            for rot in rotations:
                if (site_idx, float(height), float(rot)) in completed:
                    continue
                
                test_slab = create_structure(
                    slab, ads, site, bond_params,
                    height, rotation=rot,
                    rotation_center=center_xy,
                    binding_atom_idx=None
                )
                test_slab.calc = local_calc
                
                log_file = os.path.join(log_dir, f"site{site_idx}_{site['site']}_h{height}_r{rot}.log")
                opt = BFGS(test_slab, logfile=log_file)
                
                try:
                    opt.run(fmax=0.05)
                    converged = True
                except:
                    converged = False
                
                if not converged:
                    continue
                
                E_total = test_slab.get_potential_energy()
                E_ads = E_total - clean_slab_energy - ads_energy
                
                struct_file = os.path.join(structures_dir, f"{site['site']}_h{height}_r{rot}.xyz")
                test_slab.write(struct_file)
                
                screening_results.append({
                    "site_index": site_idx,
                    "site_type": site["site"],
                    "site_position": list(site["position"]),
                    "height": float(height),
                    "rotation": float(rot),
                    "adsorption_energy": float(E_ads),
                    "total_energy": float(E_total),
                    "structure_file": struct_file,
                    "converged": converged
                })
                print(f"Site {site_idx} ({site['site']}), h={height}, r={rot} → E_ads={E_ads:.3f} eV")
                
                # Incremental save
                if save_results:
                    with open(os.path.join(workdir, "screening_results.pkl"), "wb") as f:
                        pickle.dump(screening_results, f)
    
    # Final save
    if save_results:
        with open(os.path.join(workdir, "screening_results.pkl"), "wb") as f:
            pickle.dump(screening_results, f)
        with open(os.path.join(workdir, "screening_metadata.json"), "w") as f:
            json.dump(screening_results, f, indent=2)
        print(f"✓ Total: {len(screening_results)} results saved")
    
    return screening_results


def plot_rotation_neb(traj_file, angle_range=120, n_images=10):
    """
    Plot rotational NEB energy profile by recalculating energies.
    """
    # Read images
    images = read(traj_file, index=':')
    images = images[-(n_images+2):]  # Final iteration + endpoints
    
    # Set up calculator
    predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
    calc = FAIRChemCalculator(predictor, task_name="oc20")
    
    # Recalculate energies
    energies = []
    for i, img in enumerate(images):
        img.calc = calc
        E = img.get_potential_energy()
        energies.append(E)
        print(f"Image {i}: {E:.6f} eV")
    
    energies = np.array(energies)
    energies_rel = (energies - energies[0]) * 1000  # meV
    
    # Linearly interpolate rotation angles
    angles = np.linspace(0, angle_range, len(images))
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(angles, energies_rel, 'o-', linewidth=2, markersize=8, color='tab:orange')
    plt.xlabel('Rotation Angle (°)', fontsize=12)
    plt.ylabel('Relative Energy (meV)', fontsize=12)
    plt.title('NEB Rotation Pathway', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(traj_file.replace('.traj', '_profile.png'), dpi=300)
    plt.show()
    
    E_min = energies_rel.min()
    saddle_idx = np.argmax(energies_rel[1:-1]) + 1  # Exclude endpoints
    E_saddle = energies_rel[saddle_idx]
    true_barrier = E_saddle - E_min

    print(f"\nTrue rotational barrier: {true_barrier:.3f} meV")
    print(f"Minimum at image: {np.argmin(energies_rel)} ({angles[np.argmin(energies_rel)]:.1f}°)")
    print(f"Saddle at image: {saddle_idx} ({angles[saddle_idx]:.1f}°)")
    
    return angles, energies


def plot_translation_neb(traj_path, n_images=10):
    """
    Plot NEB energy profile by recalculating energies.
    """
    # Read final iteration images
    configs = read(traj_path, index=f'-{n_images}:')
    
    # Set up calculator
    predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
    calc = FAIRChemCalculator(predictor, task_name="oc20")
    
    # Recalculate energies
    energies = []
    for i, config in enumerate(configs):
        config.calc = calc
        E = config.get_potential_energy()
        energies.append(E)
        print(f"Image {i}: {E:.6f} eV")
    
    energies = np.array(energies)
    energies_rel = (energies - energies[0]) * 1000  # meV relative to first
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(energies_rel)), energies_rel, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Image', fontsize=12)
    plt.ylabel('Energy (meV)', fontsize=12)
    plt.title('NEB Translation Pathway', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(traj_path.replace('.traj', '_profile.png'), dpi=300)
    plt.show()
    
    # Print summary
    barrier = energies_rel.max()
    saddle_idx = np.argmax(energies_rel)
    print(f"\nBarrier: {barrier:.3f} meV")
    print(f"Saddle point at image: {saddle_idx}")
    print(f"Reaction energy: {energies_rel[-1]:.3f} meV")
    
    return energies



def create_janaf_table(thermo, T_range=(50, 1000, 50), reference_T=298.15, potentialenergy=None, SI_unit=True):
    if potentialenergy is None:
        potentialenergy = 0.0
    
    temperatures = np.arange(T_range[0], T_range[1] + T_range[2], T_range[2])

    NA = 6.02214076e23 # mol^-1
    EV = 1.60218e-19  # J (electron volt conversion)
    

    if SI_unit == True:
        EV_TO_J = NA * EV # J/mol
    else:
        EV_TO_J = 1
    
    if SI_unit == True:
        data = {
            'T (K)': temperatures,
            'S (J/mol/K)': [],
            'U (J/mol)': [],
            'A (J/mol)': [],
            'Cv (J/mol/K)': [],
            'G (J/mol)': [],
            'H (J/mol)': [],
            '-(G-H(Tref))/T (J/mol/K)': [],
            'H-H(Tref) (J/mol)': [],
        }
    else:
        data = {
            'T (K)': temperatures,
            'S (eV/K)': [],
            'U (eV)': [],
            'A (eV)': [],
            'Cv (eV/K)': [],
            'G (eV)': [],
            'H (eV)': [],
            '-(G-H(Tref))/T (eV/K)': [],
            'H-H(Tref) (eV)': [],
        }

    
    S_ref = thermo.get_entropy(reference_T, potentialenergy) * EV_TO_J
    U_ref = thermo.get_internal_energy(reference_T, potentialenergy) * EV_TO_J
    A_ref = thermo.get_helmholtz_energy(reference_T, potentialenergy) * EV_TO_J
    H_ref = U_ref  # For surface species, H ≈ U (constant area)
    G_ref = A_ref  # For surface species, G ≈ A (constant area)
    
    for T in temperatures:
        # Basic properties
        S = thermo.get_entropy(T, potentialenergy) * EV_TO_J
        U = thermo.get_internal_energy(T, potentialenergy) * EV_TO_J
        A = thermo.get_helmholtz_energy(T, potentialenergy) * EV_TO_J
        
        # For surface species: H ≈ U, G ≈ A (constant area approximation)
        H = U
        G = A
        
        # Heat capacity (numerical derivative)
        dT = 1.0  # K
        if T > T_range[0]:
            U_plus = thermo.get_internal_energy(T + dT, potentialenergy) * EV_TO_J
            U_minus = thermo.get_internal_energy(T - dT, potentialenergy) * EV_TO_J
            Cv = (U_plus - U_minus) / (2 * dT)
        else:
            U_plus = thermo.get_internal_energy(T + dT, potentialenergy) * EV_TO_J
            Cv = (U_plus - U) / dT
        
        # JANAF-style derived properties
        gibbs_function = -(G - H_ref) / T if T > 0 else 0
        enthalpy_diff = H - H_ref
        
        # Store values
        if SI_unit == True:
            data['S (J/mol/K)'].append(S)
            data['U (J/mol)'].append(U)
            data['A (J/mol)'].append(A)
            data['Cv (J/mol/K)'].append(Cv)
            data['G (J/mol)'].append(G)
            data['H (J/mol)'].append(H)
            data['-(G-H(Tref))/T (J/mol/K)'].append(gibbs_function)
            data['H-H(Tref) (J/mol)'].append(enthalpy_diff)
        else:
            data['S (eV/K)'].append(S)
            data['U (eV)'].append(U)
            data['A (eV)'].append(A)
            data['Cv (eV/K)'].append(Cv)
            data['G (eV)'].append(G)
            data['H (eV)'].append(H)
            data['-(G-H(Tref))/T (eV/K)'].append(gibbs_function)
            data['H-H(Tref) (eV)'].append(enthalpy_diff)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


def plot_thermochemistry(df, filename=None):
    """
    Create plots of thermochemical properties vs temperature.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Entropy
    axes[0, 0].plot(df['T (K)'], df['S (eV/K)'] * 1000, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Temperature (K)', fontsize=12)
    axes[0, 0].set_ylabel('Entropy (meV/K)', fontsize=12)
    axes[0, 0].set_title('Entropy vs Temperature', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Heat Capacity
    axes[0, 1].plot(df['T (K)'], df['Cv (eV/K)'] * 1000, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Temperature (K)', fontsize=12)
    axes[0, 1].set_ylabel('Heat Capacity Cv (meV/K)', fontsize=12)
    axes[0, 1].set_title('Heat Capacity vs Temperature', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Free Energies
    axes[1, 0].plot(df['T (K)'], df['G (eV)'], 'g-', linewidth=2, label='G (Gibbs)')
    axes[1, 0].plot(df['T (K)'], df['U (eV)'], 'orange', linewidth=2, label='U (Internal)')
    axes[1, 0].set_xlabel('Temperature (K)', fontsize=12)
    axes[1, 0].set_ylabel('Energy (eV)', fontsize=12)
    axes[1, 0].set_title('Thermodynamic Functions vs Temperature', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Gibbs Function
    axes[1, 1].plot(df['T (K)'], df['-(G-H(Tref))/T (eV/K)'] * 1000, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Temperature (K)', fontsize=12)
    axes[1, 1].set_ylabel('-(G-H(Tref))/T (meV/K)', fontsize=12)
    axes[1, 1].set_title('Gibbs Function vs Temperature', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {filename}")
    
    plt.show()