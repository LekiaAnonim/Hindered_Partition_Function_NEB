#!/usr/bin/env python3
"""
Automated thermochemistry calculations for surface adsorbates.
Loops through adsorbate directories and processes those with completed NEB calculations.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ase.io import read
from ase.vibrations import Vibrations
from ase.thermochemistry import HinderedThermo
from fairchem.core.calculate import pretrained_mlip
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

# Import custom modules
from model.neb import *
from model.hindered_partition_function import *

BASE_DIR = "/projects/westgroup/lekia.p/NEB/Adsorbates"
LATTICE_CONSTANT = 4.012  # Angstrom (Pt)
SITE_DENSITY = 8.79e+19  # molecules/m^2

# Molecule-specific parameters
# n: number of equivalent minima per full rotation
# symmetric_number: symmetry number for rotation
# rotor_asymmetric: True for asymmetric rotors
MOLECULE_PARAMS = {
    'CH2': {'n': 2, 'symmetric_number': 2, 'rotor_asymmetric': True},
    'CH3': {'n': 3, 'symmetric_number': 3, 'rotor_asymmetric': False},
    'CH4': {'n': 3, 'symmetric_number': 12, 'rotor_asymmetric': False},  # Td symmetry
    'CO':  {'n': 1, 'symmetric_number': 1, 'rotor_asymmetric': False, 'free_rotor': True},  # Linear, free rotation
    'CO2': {'n': 2, 'symmetric_number': 2, 'rotor_asymmetric': False, 'free_rotor': True},  # Linear
    'NH2': {'n': 2, 'symmetric_number': 2, 'rotor_asymmetric': True},
    'NH3': {'n': 3, 'symmetric_number': 3, 'rotor_asymmetric': False},
    'OH':  {'n': 1, 'symmetric_number': 1, 'rotor_asymmetric': True},
    'H2O': {'n': 2, 'symmetric_number': 2, 'rotor_asymmetric': True},
}


def get_calc():
    """Create fresh FAIRChem calculator."""
    predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
    return FAIRChemCalculator(predictor, task_name="oc20")


def check_neb_complete(ads_dir):
    """Check if NEB calculations are complete for an adsorbate."""
    trans_summary = os.path.join(ads_dir, "NEB_Translation", "neb_summary.json")
    rot_summary = os.path.join(ads_dir, "NEB_Rotation", "neb_summary.json")
    screening_pkl = os.path.join(ads_dir, "Screening_Data", "screening_results.pkl")
    
    has_translation = os.path.exists(trans_summary)
    has_rotation = os.path.exists(rot_summary)
    has_screening = os.path.exists(screening_pkl)
    
    return {
        'translation': has_translation,
        'rotation': has_rotation,
        'screening': has_screening,
        'ready': has_screening and (has_translation or has_rotation)
    }


def load_neb_with_energies(traj_path, n_images=12):
    """Load NEB trajectory and recalculate energies."""
    images = read(traj_path, index=f'-{n_images}:')
    calc = get_calc()
    
    energies = []
    for img in images:
        img.calc = calc
        energies.append(img.get_potential_energy())
    
    return images, np.array(energies)


def get_energy_from_atoms(atoms):
    """Get energy from atoms, checking multiple sources."""
    if atoms.calc is not None:
        try:
            return atoms.get_potential_energy()
        except:
            pass
    if 'energy' in atoms.info:
        return atoms.info['energy']
    raise ValueError("No energy available")


def process_adsorbate(ads_name, ads_dir, T=300):
    """
    Process a single adsorbate: load data, calculate partition functions, 
    run vibrations, and generate thermochemistry tables.
    """
    print(f"\n{'='*70}")
    print(f"Processing: {ads_name}")
    print(f"{'='*70}")
    
    # Get molecule parameters
    if ads_name not in MOLECULE_PARAMS:
        print(f"WARNING: No parameters defined for {ads_name}, using defaults")
        params = {'n': 1, 'symmetric_number': 1, 'rotor_asymmetric': True}
    else:
        params = MOLECULE_PARAMS[ads_name]
    
    n = params['n']
    symmetric_number = params['symmetric_number']
    rotor_asymmetric = params['rotor_asymmetric']
    is_free_rotor = params.get('free_rotor', False)
    
    # Create output directory
    output_dir = os.path.join(ads_dir, "Thermochemistry")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize adsorbate molecule
    ads = opt_molecule(init_molecule(ads_name))
    m = np.sum(ads.get_masses())
    moi = get_moment_of_inertia_about_binding_atom(ads=ads)
    
    print(f"Mass: {m:.4f} amu")
    print(f"Moment of inertia: {moi:.4f} amu·Å²")
    print(f"Symmetry: n={n}, σ={symmetric_number}, asymmetric={rotor_asymmetric}")
    
    # Load screening results
    screening_pkl = os.path.join(ads_dir, "Screening_Data", "screening_results.pkl")
    screening_results = load_screening_results(screening_pkl)
    
    # Get number of sites
    screening_json = os.path.join(ads_dir, "Screening_Data", "screening_metadata_.json")
    if os.path.exists(screening_json):
        with open(screening_json, 'r') as f:
            screening_data = json.load(f)
        M = max(s.get('site_index', 0) for s in screening_data) + 1
    else:
        M = 9  # Default for 3x3 surface
    
    # Load NEB barriers
    trans_summary_path = os.path.join(ads_dir, "NEB_Translation", "neb_summary.json")
    rot_summary_path = os.path.join(ads_dir, "NEB_Rotation", "neb_summary.json")
    
    W_x = 0.0
    W_r = 0.0
    
    if os.path.exists(trans_summary_path):
        with open(trans_summary_path, 'r') as f:
            neb_trans = json.load(f)
        W_x = neb_trans.get('forward_barrier_eV', 0.0)
        print(f"Translation barrier W_x: {W_x*1000:.3f} meV")
    else:
        print("WARNING: No translation NEB found, using W_x = 0")
    
    if os.path.exists(rot_summary_path) and not is_free_rotor:
        with open(rot_summary_path, 'r') as f:
            neb_rot = json.load(f)
        W_r = neb_rot.get('forward_barrier_eV', 0.0)
        print(f"Rotation barrier W_r: {W_r*1000:.3f} meV")
    else:
        if is_free_rotor:
            print(f"INFO: {ads_name} is a free rotor (linear molecule), W_r = 0")
        else:
            print("WARNING: No rotation NEB found, using W_r = 0")
    
    W_y = W_x  # Assume isotropic
    
    # =========================================================================
    # GET MINIMUM STRUCTURE - THIS WAS MISSING
    # =========================================================================
    min_config = min(screening_results, key=lambda x: x['adsorption_energy'])
    minimum_structure = read(min_config['structure_file'])
    
    print(f"\nMinimum structure: {min_config['structure_file']}")
    print(f"Adsorption energy: {min_config['adsorption_energy']:.6f} eV")
    print(f"Site type: {min_config['site_type']}")
    
    # Determine adsorbate atom indices
    n_structure_atoms = len(minimum_structure)
    n_ads_atoms = len(ads)
    n_surface_atoms = n_structure_atoms - n_ads_atoms
    vibatoms = list(range(n_surface_atoms, n_structure_atoms))
    
    print(f"Total atoms: {n_structure_atoms}, Surface: {n_surface_atoms}, Adsorbate: {n_ads_atoms}")
    print(f"Vibrating atoms: {vibatoms}")
    
    # =========================================================================
    # VIBRATIONAL ANALYSIS
    # =========================================================================
    print(f"\n--- Vibrational Analysis ---")
    
    minimum_structure.calc = get_calc()
    vib_name = os.path.join(output_dir, f'vib_{ads_name}')
    
    vib = Vibrations(
        minimum_structure,
        indices=vibatoms,
        name=vib_name,
    )
    
    # Clean old files if they exist
    vib.clean()
    
    vib.run()
    vib.summary()
    
    vib_energies = vib.get_energies()
    
    # Handle imaginary frequencies (complex numbers)
    n_imaginary = 0
    if np.any(np.iscomplex(vib_energies)):
        n_imaginary = np.sum(np.iscomplex(vib_energies))
        print(f"WARNING: {n_imaginary} imaginary frequencies detected!")
        print(f"Raw vibrational energies: {vib_energies}")
        
        # Take real part and filter positive values
        vib_energies_real = np.real(vib_energies)
        vib_energies_clean = vib_energies_real[vib_energies_real > 0]
        
        print(f"Using {len(vib_energies_clean)} real positive frequencies")
        vib_energies = vib_energies_clean
    
    # Ensure vib_energies is real numpy array
    vib_energies = np.real(vib_energies).astype(float)
    print(f"Vibrational energies (meV): {vib_energies * 1000}")
    
    # =========================================================================
    # PARTITION FUNCTIONS
    # =========================================================================
    print(f"\n--- Partition Functions at T={T} K ---")
    
    # Handle free translator (W_x ≈ 0)
    if W_x < 1e-6:
        print(f"INFO: W_x ≈ 0, using free 2D translator approximation")
        q_trans = float(M)
        f_trans = 1.0
    else:
        trans = HinderedTranslationPartitionFunction(
            m=m,
            W_x=W_x,
            W_y=W_y,
            b=LATTICE_CONSTANT,
            M=M,
            T=T
        )
        q_trans = trans.q_trans()
        f_trans = trans.f_trans()
    
    # Handle free rotor (W_r ≈ 0 or is_free_rotor flag)
    if W_r < 1e-6 or is_free_rotor:
        print(f"INFO: W_r ≈ 0 or free rotor, using free 1D rotor approximation")
        # Free rotor: q = sqrt(8π³IkT/h²) / σ
        I_SI = moi * 1.66053906660e-27 * (1e-10)**2  # kg·m²
        kT_SI = 1.380649e-23 * T  # J
        h_SI = 6.62607015e-34  # J·s
        q_rot = np.sqrt(8 * np.pi**3 * I_SI * kT_SI) / (h_SI * symmetric_number)
        f_rot = 1.0
        W_r = 0.0
    else:
        rotor = HinderedRotorPartitionFunction(
            W_r=W_r,
            n=n,
            I=moi,
            T=T,
            symmetric_number=symmetric_number,
            rotor_asymmetric=rotor_asymmetric
        )
        q_rot = rotor.q_rot()
        f_rot = rotor.f_rot()
    
    print(f"q_trans: {q_trans:.6e}")
    print(f"q_rot: {q_rot:.6e}")
    
    # =========================================================================
    # THERMOCHEMISTRY
    # =========================================================================
    print(f"\n--- Thermochemistry ---")
    
    thermo = HinderedThermo(
        vib_energies=vib_energies,
        trans_barrier_energy=max(W_x, 1e-10),
        rot_barrier_energy=max(W_r, 1e-10),
        sitedensity=SITE_DENSITY,
        rotationalminima=n,
        symmetrynumber=symmetric_number,
        mass=m,
        inertia=moi,
    )
    
    # Generate JANAF table
    janaf_table = create_janaf_table(thermo, SI_unit=False)
    
    # Save JANAF table
    janaf_csv = os.path.join(output_dir, f"{ads_name}_janaf_table.csv")
    janaf_table.to_csv(janaf_csv, index=False)
    print(f"Saved JANAF table: {janaf_csv}")
    
    # =========================================================================
    # SAVE SUMMARY (JSON serializable)
    # =========================================================================
    summary = {
        'adsorbate': ads_name,
        'mass_amu': float(m),
        'moment_of_inertia_amu_A2': float(moi),
        'W_x_eV': float(W_x),
        'W_r_eV': float(W_r),
        'n_minima': int(n),
        'symmetry_number': int(symmetric_number),
        'rotor_asymmetric': bool(rotor_asymmetric),
        'is_free_rotor': bool(is_free_rotor),
        'q_trans_300K': float(np.real(q_trans)),
        'q_rot_300K': float(np.real(q_rot)),
        'site_type': str(min_config['site_type']),
        'adsorption_energy_eV': float(min_config['adsorption_energy']),
        'vibrational_energies_eV': [float(e) for e in vib_energies],
        'n_imaginary_frequencies': int(n_imaginary),
        'num_sites': int(M),
    }
    
    summary_json = os.path.join(output_dir, f"{ads_name}_summary.json")
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_json}")
    
    # Plot thermochemistry
    plot_file = os.path.join(output_dir, f"{ads_name}_thermochemistry.png")
    plot_thermochemistry(janaf_table, filename=plot_file)
    
    return summary, janaf_table

def calculate_partition_functions_vs_T(
    m, W_x, W_r, b, M, moi, n, symmetric_number, rotor_asymmetric,
    T_range=(50, 1000, 25), is_free_rotor=False
):
    """
    Calculate hindered translator and rotor partition functions vs temperature.
    
    Parameters
    ----------
    m : float
        Adsorbate mass (amu)
    W_x : float
        Translational barrier (eV)
    W_r : float
        Rotational barrier (eV)
    b : float
        Lattice constant (Å)
    M : int
        Number of surface sites
    moi : float
        Moment of inertia (amu·Å²)
    n : int
        Number of equivalent rotational minima
    symmetric_number : int
        Symmetry number
    rotor_asymmetric : bool
        True for asymmetric rotor
    T_range : tuple
        (T_min, T_max, T_step) in Kelvin
    is_free_rotor : bool
        If True, use free rotor approximation
    
    Returns
    -------
    dict with temperatures and partition function arrays
    """
    temperatures = np.arange(T_range[0], T_range[1] + T_range[2], T_range[2])
    
    results = {
        'T': temperatures,
        'q_trans': [],
        'q_rot': [],
        'f_trans': [],
        'f_rot': [],
        'P_trans': [],
        'P_rot': [],
        'q_HO_trans': [],
        'q_HO_rot': [],
        'nu_trans': None,
        'nu_rot': None,
        'r_x': None,
        'r_r': None,
    }
    
    # Physical constants for free rotor
    I_SI = moi * 1.66053906660e-27 * (1e-10)**2  # kg·m²
    h_SI = 6.62607015e-34  # J·s
    k_SI = 1.380649e-23  # J/K
    
    for T in temperatures:
        # Translation partition function
        if W_x < 1e-6:
            # Free 2D translator
            q_trans = float(M)
            f_trans = 1.0
            P_trans = 1.0
            q_HO_trans = 1.0
        else:
            trans = HinderedTranslationPartitionFunction(
                m=m, W_x=W_x, W_y=W_x, b=b, M=M, T=T
            )
            q_trans = trans.q_trans()
            f_trans = trans.f_trans()
            P_trans = trans.P_trans()
            q_HO_trans = trans.q_HO()
            
            # Store frequency info (same for all T)
            if results['nu_trans'] is None:
                results['nu_trans'] = trans.vibrational_frequency
                results['r_x'] = trans.r_x
        
        # Rotation partition function
        if W_r < 1e-6 or is_free_rotor:
            # Free 1D rotor: q = sqrt(8π³IkT) / (h·σ)
            kT_SI = k_SI * T
            q_rot = np.sqrt(8 * np.pi**3 * I_SI * kT_SI) / (h_SI * symmetric_number)
            f_rot = 1.0
            P_rot = 1.0
            q_HO_rot = 1.0
        else:
            rotor = HinderedRotorPartitionFunction(
                W_r=W_r, n=n, I=moi, T=T,
                symmetric_number=symmetric_number,
                rotor_asymmetric=rotor_asymmetric
            )
            q_rot = rotor.q_rot()
            f_rot = rotor.f_rot()
            P_rot = rotor.P_rot()
            q_HO_rot = rotor.q_HO_r()
            
            # Store frequency info
            if results['nu_rot'] is None:
                results['nu_rot'] = rotor.vibrational_frequency
                results['r_r'] = rotor.r_r
        
        results['q_trans'].append(float(np.real(q_trans)))
        results['q_rot'].append(float(np.real(q_rot)))
        results['f_trans'].append(float(np.real(f_trans)))
        results['f_rot'].append(float(np.real(f_rot)))
        results['P_trans'].append(float(np.real(P_trans)))
        results['P_rot'].append(float(np.real(P_rot)))
        results['q_HO_trans'].append(float(np.real(q_HO_trans)))
        results['q_HO_rot'].append(float(np.real(q_HO_rot)))
    
    # Convert to numpy arrays
    for key in ['q_trans', 'q_rot', 'f_trans', 'f_rot', 'P_trans', 'P_rot', 'q_HO_trans', 'q_HO_rot']:
        results[key] = np.array(results[key])
    
    return results


def plot_partition_functions(results, ads_name, W_x, W_r, save_path=None):
    """
    Create publication-quality plots of partition functions vs temperature.
    
    Parameters
    ----------
    results : dict
        Output from calculate_partition_functions_vs_T
    ads_name : str
        Adsorbate name for title
    W_x : float
        Translational barrier (eV) for annotation
    W_r : float
        Rotational barrier (eV) for annotation
    save_path : str, optional
        Path to save figure
    """
    T = results['T']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Color scheme
    c_trans = 'tab:blue'
    c_rot = 'tab:orange'
    c_HO = 'tab:gray'
    
    # --- Plot 1: Total partition functions ---
    ax = axes[0, 0]
    ax.semilogy(T, results['q_trans'], '-', color=c_trans, linewidth=2, label='$q_{trans}$')
    ax.semilogy(T, results['q_rot'], '-', color=c_rot, linewidth=2, label='$q_{rot}$')
    ax.semilogy(T, results['q_trans'] * results['q_rot'], 'k--', linewidth=2, label='$q_{trans} \\times q_{rot}$')
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Partition Function', fontsize=12)
    ax.set_title('Total Partition Functions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(T.min(), T.max())
    
    # --- Plot 2: Interpolation functions f ---
    ax = axes[0, 1]
    ax.plot(T, results['f_trans'], '-', color=c_trans, linewidth=2, label='$f_{trans}$')
    ax.plot(T, results['f_rot'], '-', color=c_rot, linewidth=2, label='$f_{rot}$')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Free motion limit')
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Interpolation Function $f$', fontsize=12)
    ax.set_title('Hindered → Free Transition', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(T.min(), T.max())
    
    # --- Plot 3: Probability factors P ---
    ax = axes[0, 2]
    ax.plot(T, results['P_trans'], '-', color=c_trans, linewidth=2, label='$P_{trans}$')
    ax.plot(T, results['P_rot'], '-', color=c_rot, linewidth=2, label='$P_{rot}$')
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Probability Factor $P$', fontsize=12)
    ax.set_title('Probability Factors', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(T.min(), T.max())
    
    # --- Plot 4: Translation partition function breakdown ---
    ax = axes[1, 0]
    ax.semilogy(T, results['q_trans'], '-', color=c_trans, linewidth=2, label='$q_{trans}$ (hindered)')
    ax.semilogy(T, results['q_HO_trans'], '--', color=c_HO, linewidth=2, label='$q_{HO}$ (harmonic)')
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Partition Function', fontsize=12)
    ax.set_title('Translation: Hindered vs Harmonic', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(T.min(), T.max())
    
    # --- Plot 5: Rotation partition function breakdown ---
    ax = axes[1, 1]
    ax.semilogy(T, results['q_rot'], '-', color=c_rot, linewidth=2, label='$q_{rot}$ (hindered)')
    ax.semilogy(T, results['q_HO_rot'], '--', color=c_HO, linewidth=2, label='$q_{HO}$ (harmonic)')
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Partition Function', fontsize=12)
    ax.set_title('Rotation: Hindered vs Harmonic', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(T.min(), T.max())
    
    # --- Plot 6: W/kT ratio (regime indicator) ---
    ax = axes[1, 2]
    kT = 8.617333262e-5 * T  # eV
    if W_x > 1e-6:
        ax.plot(T, W_x / kT, '-', color=c_trans, linewidth=2, label='$W_x/kT$')
    if W_r > 1e-6:
        ax.plot(T, W_r / kT, '-', color=c_rot, linewidth=2, label='$W_r/kT$')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Crossover ($W/kT = 1$)')
    ax.fill_between(T, 0, 1, alpha=0.2, color='green', label='Free motion regime')
    ax.fill_between(T, 1, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 10, alpha=0.2, color='blue', label='Localized regime')
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('$W/kT$', fontsize=12)
    ax.set_title('Motion Regime Indicator', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(T.min(), T.max())
    ax.set_ylim(0, max(5, W_x / (8.617333262e-5 * T.min()) if W_x > 1e-6 else 5))
    
    # Main title with parameters
    fig.suptitle(
        f'{ads_name} Partition Functions\n'
        f'$W_x$ = {W_x*1000:.2f} meV, $W_r$ = {W_r*1000:.2f} meV',
        fontsize=16, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved partition function plot: {save_path}")
    
    plt.show()
    
    return fig


def plot_partition_functions_comparison(all_results, save_path=None):
    """
    Compare partition functions across multiple adsorbates.
    
    Parameters
    ----------
    all_results : dict
        {ads_name: results_dict} for each adsorbate
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for (ads_name, results), color in zip(all_results.items(), colors):
        T = results['T']
        
        # q_trans
        axes[0, 0].semilogy(T, results['q_trans'], '-', color=color, linewidth=2, label=ads_name)
        
        # q_rot
        axes[0, 1].semilogy(T, results['q_rot'], '-', color=color, linewidth=2, label=ads_name)
        
        # f_trans
        axes[1, 0].plot(T, results['f_trans'], '-', color=color, linewidth=2, label=ads_name)
        
        # f_rot
        axes[1, 1].plot(T, results['f_rot'], '-', color=color, linewidth=2, label=ads_name)
    
    axes[0, 0].set_xlabel('Temperature (K)', fontsize=12)
    axes[0, 0].set_ylabel('$q_{trans}$', fontsize=12)
    axes[0, 0].set_title('Translational Partition Function', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Temperature (K)', fontsize=12)
    axes[0, 1].set_ylabel('$q_{rot}$', fontsize=12)
    axes[0, 1].set_title('Rotational Partition Function', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Temperature (K)', fontsize=12)
    axes[1, 0].set_ylabel('$f_{trans}$', fontsize=12)
    axes[1, 0].set_title('Translation Interpolation Function', fontsize=14, fontweight='bold')
    axes[1, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Temperature (K)', fontsize=12)
    axes[1, 1].set_ylabel('$f_{rot}$', fontsize=12)
    axes[1, 1].set_title('Rotation Interpolation Function', fontsize=14, fontweight='bold')
    axes[1, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle('Partition Function Comparison Across Adsorbates', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot: {save_path}")
    
    plt.show()
    
    return fig


# Add these functions to your thermochemistry script
# They automatically extract parameters from the saved JSON summaries

def load_adsorbate_summary(ads_name, base_dir=BASE_DIR):
    """
    Load the thermochemistry summary JSON for an adsorbate.
    
    Parameters
    ----------
    ads_name : str
        Adsorbate name (e.g., 'NH3', 'CO')
    base_dir : str
        Base directory containing adsorbate folders
    
    Returns
    -------
    dict
        Summary dictionary with all parameters
    """
    summary_path = os.path.join(
        base_dir, ads_name, "Thermochemistry", f"{ads_name}_summary.json"
    )
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}\n"
            f"Run process_adsorbate('{ads_name}', ...) first."
        )
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def calculate_partition_functions_from_results(
    ads_name, 
    base_dir=BASE_DIR, 
    T_range=(50, 1000, 25),
    save_plot=True
):
    """
    Calculate partition functions using parameters from saved results.
    
    Automatically extracts m, W_x, W_r, M, moi, n, symmetric_number, etc.
    from the adsorbate's summary JSON file.
    
    Parameters
    ----------
    ads_name : str
        Adsorbate name (e.g., 'NH3', 'CO')
    base_dir : str
        Base directory containing adsorbate folders
    T_range : tuple
        (T_min, T_max, T_step) in Kelvin
    save_plot : bool
        If True, save plot to Thermochemistry folder
    
    Returns
    -------
    dict
        Partition function results from calculate_partition_functions_vs_T
    """
    # Load saved summary
    summary = load_adsorbate_summary(ads_name, base_dir)
    
    # Extract parameters
    m = summary['mass_amu']
    W_x = summary['W_x_eV']
    W_r = summary['W_r_eV']
    moi = summary['moment_of_inertia_amu_A2']
    n = summary['n_minima']
    symmetric_number = summary['symmetry_number']
    rotor_asymmetric = summary['rotor_asymmetric']
    is_free_rotor = summary['is_free_rotor']
    M = summary['num_sites']
    
    print(f"Loaded parameters for {ads_name}:")
    print(f"  Mass: {m:.3f} amu")
    print(f"  W_x: {W_x*1000:.3f} meV")
    print(f"  W_r: {W_r*1000:.3f} meV")
    print(f"  MOI: {moi:.3f} amu·Å²")
    print(f"  n={n}, σ={symmetric_number}, free_rotor={is_free_rotor}")
    
    # Calculate partition functions
    results = calculate_partition_functions_vs_T(
        m=m,
        W_x=W_x,
        W_r=W_r,
        b=LATTICE_CONSTANT,
        M=M,
        moi=moi,
        n=n,
        symmetric_number=symmetric_number,
        rotor_asymmetric=rotor_asymmetric,
        T_range=T_range,
        is_free_rotor=is_free_rotor
    )
    
    # Optionally save plot
    if save_plot:
        output_dir = os.path.join(base_dir, ads_name, "Thermochemistry")
        plot_path = os.path.join(output_dir, f"{ads_name}_partition_functions.png")
        plot_partition_functions(results, ads_name, W_x, W_r, save_path=plot_path)
    
    return results


def calculate_all_partition_functions(
    base_dir=BASE_DIR, 
    T_range=(50, 1000, 25),
    save_plots=True,
    save_comparison=True
):
    """
    Calculate partition functions for all processed adsorbates.
    
    Scans for adsorbates with completed thermochemistry summaries
    and generates partition function data and plots for each.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing adsorbate folders
    T_range : tuple
        (T_min, T_max, T_step) in Kelvin
    save_plots : bool
        If True, save individual plots
    save_comparison : bool
        If True, save comparison plot of all adsorbates
    
    Returns
    -------
    dict
        {ads_name: results_dict} for all processed adsorbates
    """
    all_results = {}
    
    # Find adsorbates with completed summaries
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        summary_path = os.path.join(
            item_path, "Thermochemistry", f"{item}_summary.json"
        )
        
        if os.path.exists(summary_path):
            print(f"\n{'='*50}")
            print(f"Processing: {item}")
            print(f"{'='*50}")
            
            try:
                results = calculate_partition_functions_from_results(
                    item, base_dir, T_range, save_plot=save_plots
                )
                all_results[item] = results
            except Exception as e:
                print(f"ERROR processing {item}: {e}")
                import traceback
                traceback.print_exc()
    
    # Generate comparison plot
    if save_comparison and len(all_results) > 1:
        comparison_path = os.path.join(
            base_dir, "partition_functions_comparison.png"
        )
        plot_partition_functions_comparison(all_results, save_path=comparison_path)
    
    # Save combined results as JSON
    combined_path = os.path.join(base_dir, "partition_functions_all.json")
    json_results = {}
    for ads_name, results in all_results.items():
        json_results[ads_name] = {
            'T': results['T'].tolist(),
            'q_trans': results['q_trans'].tolist(),
            'q_rot': results['q_rot'].tolist(),
            'f_trans': results['f_trans'].tolist(),
            'f_rot': results['f_rot'].tolist(),
        }
    with open(combined_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved combined results: {combined_path}")
    
    return all_results



def main():
    """Main function to process all adsorbates."""
    
    print("="*70)
    print("Automated Thermochemistry Calculations for Surface Adsorbates")
    print("="*70)
    
    # Find all adsorbate directories
    adsorbate_dirs = []
    for item in os.listdir(BASE_DIR):
        item_path = os.path.join(BASE_DIR, item)
        if os.path.isdir(item_path) and item not in ['model', 'vib', '__pycache__']:
            # Check if it looks like an adsorbate directory
            if os.path.exists(os.path.join(item_path, 'Screening_Data')):
                adsorbate_dirs.append((item, item_path))
    
    print(f"\nFound {len(adsorbate_dirs)} adsorbate directories:")
    for name, path in adsorbate_dirs:
        status = check_neb_complete(path)
        status_str = f"T:{status['translation']} R:{status['rotation']} S:{status['screening']}"
        ready_str = "✓ READY" if status['ready'] else "✗ NOT READY"
        print(f"  {name:8s} [{status_str}] {ready_str}")
    
    # Process adsorbates with complete data
    results = {}
    
    for ads_name, ads_dir in adsorbate_dirs:
        status = check_neb_complete(ads_dir)
        
        if not status['ready']:
            print(f"\nSkipping {ads_name}: incomplete data")
            continue
        
        try:
            summary, table = process_adsorbate(ads_name, ads_dir)
            results[ads_name] = {'summary': summary, 'table': table, 'status': 'success'}
        except Exception as e:
            print(f"\nERROR processing {ads_name}: {e}")
            import traceback
            traceback.print_exc()
            results[ads_name] = {'status': 'error', 'error': str(e)}
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for ads_name, result in results.items():
        if result['status'] == 'success':
            s = result['summary']
            print(f"\n{ads_name}:")
            print(f"  Site: {s['site_type']}, E_ads: {s['adsorption_energy_eV']:.4f} eV")
            print(f"  W_x: {s['W_x_eV']*1000:.3f} meV, W_r: {s['W_r_eV']*1000:.3f} meV")
            print(f"  q_trans: {s['q_trans_300K']:.3e}, q_rot: {s['q_rot_300K']:.3e}")
        else:
            print(f"\n{ads_name}: FAILED - {result['error']}")
    
    # Save master summary
    master_summary = {k: v['summary'] for k, v in results.items() if v['status'] == 'success'}
    master_file = os.path.join(BASE_DIR, "thermochemistry_summary.json")
    with open(master_file, 'w') as f:
        json.dump(master_summary, f, indent=2)
    print(f"\nMaster summary saved: {master_file}")


     # Option 1: Process single adsorbate from saved results
    # results_NH3 = calculate_partition_functions_from_results('NH3')
    
    # Option 2: Process all adsorbates with completed thermochemistry
    all_results = calculate_all_partition_functions(
        T_range=(50, 1000, 25),
        save_plots=True,
        save_comparison=True
    )
    
    # Option 3: Manual comparison of specific adsorbates
    # selected = ['NH3', 'CO', 'CH3']
    # selected_results = {}
    # for ads in selected:
    #     try:
    #         selected_results[ads] = calculate_partition_functions_from_results(
    #             ads, save_plot=False
    #         )
    #     except FileNotFoundError as e:
    #         print(f"Skipping {ads}: {e}")
    
    # if len(selected_results) > 1:
    #     plot_partition_functions_comparison(
    #         selected_results, 
    #         save_path='selected_partition_comparison.png'
    #     )
    
    return results




if __name__ == "__main__":
    results = main()