"""
Revised NEB endpoint selection and calculation functions.
Addresses all identified issues with robust error handling and validation.
"""

import os
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from ase.io import read, write
from ase.optimize import FIRE, BFGS
from ase.mep import NEB
from ase.constraints import FixAtoms
from ase.geometry import find_mic

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

MOLECULAR_SYMMETRY = {
    'NH3': {'symmetry': 'C3v', 'period': 120, 'barrier_angle': 60},
    'CH3': {'symmetry': 'C3v', 'period': 120, 'barrier_angle': 60},
    'H2O': {'symmetry': 'C2v', 'period': 180, 'barrier_angle': 90},
    'CH4': {'symmetry': 'Td',  'period': 120, 'barrier_angle': 60},
    'CO':  {'symmetry': 'Cinf', 'period': 360, 'barrier_angle': None},
    'NO':  {'symmetry': 'Cinf', 'period': 360, 'barrier_angle': None},
    'CH2': {'symmetry': 'C2v', 'period': 180, 'barrier_angle': 90},
    'OH':  {'symmetry': 'Cinf', 'period': 360, 'barrier_angle': None},
    'H':   {'symmetry': 'spherical', 'period': 360, 'barrier_angle': None},
    'O':   {'symmetry': 'spherical', 'period': 360, 'barrier_angle': None},
    'N':   {'symmetry': 'spherical', 'period': 360, 'barrier_angle': None},
}

METAL_ELEMENTS = {'Pt', 'Pd', 'Ni', 'Cu', 'Au', 'Ag', 'Rh', 'Ir', 'Fe', 'Co', 
                  'Ru', 'Os', 'Mo', 'W', 'Ti', 'V', 'Cr', 'Mn', 'Zn'}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _to_dict_safe(obj):
    """Convert to dict regardless of input type."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    else:
        return dict(obj)


def make_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                          np.int32, np.int64, np.uint8, np.uint16, 
                          np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def fresh_calc(device="cpu", model="uma-s-1"):
    """Create a fresh calculator instance."""
    pred = pretrained_mlip.get_predict_unit(model, device=device)
    return FAIRChemCalculator(pred, task_name="oc20")


def get_recommended_rotation_angle(adsorbate_formula):
    """
    Get recommended rotation angle for NEB based on molecular symmetry.
    
    Parameters
    ----------
    adsorbate_formula : str
        Chemical formula of the adsorbate (e.g., 'NH3', 'CH3')
    
    Returns
    -------
    float or None
        Recommended rotation angle in degrees, or None if rotation is meaningless
    """
    if adsorbate_formula in MOLECULAR_SYMMETRY:
        info = MOLECULAR_SYMMETRY[adsorbate_formula]
        if info['barrier_angle'] is None:
            print(f"⚠️  {adsorbate_formula} has {info['symmetry']} symmetry — rotation barrier is zero/undefined")
            return None
        print(f"ℹ️  {adsorbate_formula} ({info['symmetry']}): recommended rotation = {info['barrier_angle']}°")
        return info['barrier_angle']
    else:
        print(f"⚠️  Unknown molecule '{adsorbate_formula}', defaulting to 60°")
        return 60


def detect_adsorbate_indices(atoms, n_slab_atoms=None, z_threshold=3.0):
    """
    Detect which atoms are the adsorbate vs slab.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Full system (slab + adsorbate)
    n_slab_atoms : int, optional
        If known, number of slab atoms
    z_threshold : float
        Adsorbate must be within this distance of the top (Å)
    
    Returns
    -------
    list
        Indices of adsorbate atoms
    """
    if n_slab_atoms is not None:
        return list(range(n_slab_atoms, len(atoms)))
    
    # Strategy 1: By tag (tag=0 often means adsorbate)
    tag_zero = [a.index for a in atoms if a.tag == 0]
    if tag_zero and len(tag_zero) < len(atoms) // 2:
        return tag_zero
    
    # Strategy 2: Non-metal atoms that are high
    z = atoms.positions[:, 2]
    z_max = z.max()
    high_atoms = set(np.where(z > z_max - z_threshold)[0])
    non_metal_indices = {a.index for a in atoms if a.symbol not in METAL_ELEMENTS}
    
    adsorbate_indices = list(high_atoms & non_metal_indices)
    
    if adsorbate_indices:
        return sorted(adsorbate_indices)
    
    # Fallback: just high atoms
    return sorted(list(high_atoms))


def ensure_constraints(atoms, n_fixed_layers=2, adsorbate_indices=None):
    """
    Ensure slab atoms are fixed with FixAtoms constraint.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Structure to constrain
    n_fixed_layers : int
        Number of bottom layers to fix (if using tags)
    adsorbate_indices : list, optional
        Indices of adsorbate atoms (will not be fixed)
    
    Returns
    -------
    ase.Atoms
        Structure with constraints applied
    """
    if len(atoms.constraints) > 0:
        # Check if FixAtoms already present
        for c in atoms.constraints:
            if isinstance(c, FixAtoms):
                print(f"  ✓ Constraints already present: {len(c.index)} atoms fixed")
                return atoms
    
    print("  Applying constraints...")
    
    if adsorbate_indices is None:
        adsorbate_indices = detect_adsorbate_indices(atoms)
    
    adsorbate_set = set(adsorbate_indices)
    
    # Try by tag first
    fix_indices = [a.index for a in atoms if a.tag > n_fixed_layers and a.index not in adsorbate_set]
    
    if not fix_indices:
        # Fallback: fix by z-position (bottom 60% of slab atoms)
        slab_indices = [i for i in range(len(atoms)) if i not in adsorbate_set]
        if slab_indices:
            z_slab = atoms.positions[slab_indices, 2]
            z_cutoff = np.percentile(z_slab, 60)
            fix_indices = [i for i in slab_indices if atoms.positions[i, 2] < z_cutoff]
    
    if fix_indices:
        atoms.set_constraint(FixAtoms(indices=fix_indices))
        print(f"    Fixed {len(fix_indices)} atoms, {len(adsorbate_indices)} adsorbate atoms free")
    else:
        print("    ⚠️  Could not determine which atoms to fix!")
    
    return atoms


def verify_neb_endpoints(atoms1, atoms2, name="endpoints"):
    """
    Comprehensive validation of NEB endpoints.
    
    Returns
    -------
    dict
        Validation results with 'valid' key and diagnostics
    """
    results = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'rmsd': None,
        'max_displacement': None,
    }
    
    # Check 1: Same number of atoms
    if len(atoms1) != len(atoms2):
        results['valid'] = False
        results['issues'].append(f"Different atom counts: {len(atoms1)} vs {len(atoms2)}")
        return results
    
    # Check 2: Same species
    sym1 = atoms1.get_chemical_symbols()
    sym2 = atoms2.get_chemical_symbols()
    if sym1 != sym2:
        results['valid'] = False
        results['issues'].append("Different atomic species or order")
        return results
    
    # Check 3: NaN/Inf positions
    pos1 = atoms1.get_positions()
    pos2 = atoms2.get_positions()
    
    if np.any(np.isnan(pos1)) or np.any(np.isinf(pos1)):
        results['valid'] = False
        results['issues'].append("Endpoint 1 has NaN/Inf positions")
        return results
    
    if np.any(np.isnan(pos2)) or np.any(np.isinf(pos2)):
        results['valid'] = False
        results['issues'].append("Endpoint 2 has NaN/Inf positions")
        return results
    
    # Check 4: Structural difference
    diff = pos2 - pos1
    
    # Apply minimum image convention if periodic
    if np.any(atoms1.pbc):
        diff_mic, _ = find_mic(diff, atoms1.cell, pbc=atoms1.pbc)
        diff = diff_mic
    
    rmsd = np.sqrt(np.mean(diff**2))
    max_disp = np.max(np.linalg.norm(diff, axis=1))
    
    results['rmsd'] = float(rmsd)
    results['max_displacement'] = float(max_disp)
    
    if max_disp < 0.05:
        results['valid'] = False
        results['issues'].append(f"Endpoints too similar (max displacement: {max_disp:.4f} Å)")
    elif max_disp < 0.2:
        results['warnings'].append(f"Small displacement ({max_disp:.3f} Å) — barrier may be tiny")
    
    # Check 5: Same constraints
    c1 = atoms1.constraints
    c2 = atoms2.constraints
    
    if len(c1) != len(c2):
        results['warnings'].append(f"Different constraint counts: {len(c1)} vs {len(c2)}")
    
    return results


def load_structure_from_endpoint(endpoint):
    """
    Load ASE Atoms from endpoint (dict or Atoms).
    
    Parameters
    ----------
    endpoint : dict or ase.Atoms
        Endpoint specification
    
    Returns
    -------
    ase.Atoms
        Loaded structure
    """
    if isinstance(endpoint, dict):
        if 'structure' in endpoint and endpoint['structure'] is not None:
            return endpoint['structure'].copy()
        elif 'structure_file' in endpoint:
            return read(endpoint['structure_file'])
        else:
            raise ValueError("Endpoint dict must have 'structure' or 'structure_file'")
    else:
        return endpoint.copy()


# =============================================================================
# ENDPOINT SELECTION FUNCTIONS
# =============================================================================

def select_neb_endpoints_translation_v2(site_best, screening_results, 
                                         method='long_path',
                                         use_global_minimum=True,
                                         min_crossing_distance=2.5,
                                         height_tolerance=0.05,
                                         rotation_tolerance=1.0,
                                         verify_structures=True):
    """
    Select NEB endpoints for translation barrier calculation.
    
    Parameters
    ----------
    site_best : pd.DataFrame
        Best configurations per site type (from best_site_results)
    screening_results : list
        Full screening results
    method : str
        'cross_site': Use different site types (recommended)
        'long_path': Same site type but path crosses intermediate site
    min_crossing_distance : float
        Minimum distance for 'long_path' method to ensure crossing saddle (Å)
    height_tolerance : float
        Tolerance for height matching (Å)
    rotation_tolerance : float
        Tolerance for rotation matching (degrees)
    verify_structures : bool
        Whether to load and verify structures before returning
    
    Returns
    -------
    tuple
        (endpoint1_dict, endpoint2_dict) or (None, None) on failure
    """

    if use_global_minimum:
        df = pd.DataFrame(screening_results)
        df_conv = df[df['converged'] == True]
        best_idx = df_conv['total_energy'].idxmin()
        best_config = df_conv.loc[best_idx]
    else:
        best_config = site_best.iloc[0]

    # Input validation
    if site_best is None or len(site_best) == 0:
        print("❌ ERROR: site_best is empty!")
        return None, None
    
    if not screening_results or len(screening_results) == 0:
        print("❌ ERROR: No screening results provided!")
        return None, None
    
    best_config = site_best.iloc[0]
    target_site_type = best_config['site_type']
    target_height = best_config['height']
    target_rotation = best_config['rotation']
    best_position = np.array(best_config['site_position'][:2])
    best_energy = best_config['total_energy']
    
    print(f"\n{'='*60}")
    print(f"Translation Endpoint Selection (method='{method}')")
    print(f"{'='*60}")
    print(f"Reference: {target_site_type} at ({best_position[0]:.2f}, {best_position[1]:.2f})")
    print(f"Height: {target_height:.2f} Å, Rotation: {target_rotation:.1f}°")
    print(f"Energy: {best_energy:.6f} eV")
    
    # Prepare DataFrame
    df = pd.DataFrame(screening_results)
    df = df[df['converged'] == True].copy()
    
    if len(df) == 0:
        print("❌ ERROR: No converged structures in screening results!")
        return None, None
    
    # Calculate distances from reference
    df['distance'] = df['site_position'].apply(
        lambda pos: np.linalg.norm(np.array(pos[:2]) - best_position)
    )
    df['dE_meV'] = (df['total_energy'] - best_energy) * 1000
    
    if method == 'cross_site':
        # =================================================================
        # METHOD 1: Find path to DIFFERENT site type
        # =================================================================
        other_sites = df[df['site_type'] != target_site_type].copy()
        
        if len(other_sites) == 0:
            print("❌ ERROR: No other site types found!")
            print(f"   Available site types: {df['site_type'].unique().tolist()}")
            return None, None
        
        site_types_available = other_sites['site_type'].unique()
        print(f"\nOther site types: {list(site_types_available)}")
        
        # Find nearest of each type
        candidates = []
        for st in site_types_available:
            st_sites = other_sites[other_sites['site_type'] == st]
            nearest = st_sites.nsmallest(1, 'distance').iloc[0]
            candidates.append({
                'site_type': st,
                'distance': nearest['distance'],
                'dE_meV': nearest['dE_meV'],
                'config': nearest
            })
            print(f"  {target_site_type} → {st}: d={nearest['distance']:.2f} Å, ΔE={nearest['dE_meV']:.1f} meV")
        
        # Select candidate with highest energy (= largest barrier)
        candidates.sort(key=lambda x: x['dE_meV'], reverse=True)
        selected = candidates[0]
        
        if selected['dE_meV'] < 1.0:
            print(f"\n⚠️  WARNING: Very small energy difference ({selected['dE_meV']:.2f} meV)")
            print("    Translation barrier may be negligible")
        
        print(f"\n✓ Selected: {target_site_type} → {selected['site_type']}")
        print(f"  Distance: {selected['distance']:.2f} Å")
        print(f"  Barrier estimate: {selected['dE_meV']:.1f} meV")
        
        endpoint1 = _to_dict_safe(best_config)
        endpoint2 = _to_dict_safe(selected['config'])
    
    elif method == 'long_path':
        # =================================================================
        # METHOD 2: Same site type but longer path
        # =================================================================
        same_sites = df[
            (df['site_type'] == target_site_type) &
            (np.abs(df['height'] - target_height) < height_tolerance) &
            (np.abs(df['rotation'] - target_rotation) < rotation_tolerance)
        ].copy()
        
        print(f"\nSame-type sites with matching geometry: {len(same_sites)}")
        
        far_sites = same_sites[same_sites['distance'] > min_crossing_distance]
        
        if len(far_sites) == 0:
            print(f"❌ ERROR: No {target_site_type} sites beyond {min_crossing_distance} Å")
            available_dists = sorted(same_sites['distance'].unique())
            print(f"   Available distances: {available_dists[:10]}")
            return None, None
        
        # Pick nearest beyond minimum
        selected = far_sites.nsmallest(1, 'distance').iloc[0]
        
        print(f"\n✓ Selected: {target_site_type} → {target_site_type}")
        print(f"  Distance: {selected['distance']:.2f} Å (crosses intermediate site)")
        print(f"  ΔE: {selected['dE_meV']:.2f} meV")
        
        endpoint1 = _to_dict_safe(best_config)
        endpoint2 = _to_dict_safe(selected)
    
    else:
        print(f"❌ ERROR: Unknown method '{method}'")
        print("   Valid options: 'cross_site', 'long_path'")
        return None, None
    
    # Verify different files
    if endpoint1.get('structure_file') == endpoint2.get('structure_file'):
        print("❌ ERROR: Both endpoints point to the same file!")
        return None, None
    
    # Optional structure verification
    if verify_structures:
        print("\nVerifying structures...")
        try:
            atoms1 = load_structure_from_endpoint(endpoint1)
            atoms2 = load_structure_from_endpoint(endpoint2)
            
            validation = verify_neb_endpoints(atoms1, atoms2)
            
            if not validation['valid']:
                print(f"❌ Endpoint validation failed!")
                for issue in validation['issues']:
                    print(f"   - {issue}")
                return None, None
            
            print(f"  ✓ RMSD: {validation['rmsd']:.3f} Å")
            print(f"  ✓ Max displacement: {validation['max_displacement']:.3f} Å")
            
            for warning in validation['warnings']:
                print(f"  ⚠️  {warning}")
                
        except Exception as e:
            print(f"⚠️  Could not verify structures: {e}")
    
    print(f"{'='*60}\n")
    
    return endpoint1, endpoint2


def select_neb_endpoints_rotation_v2(site_best, screening_results, 
                                      rotation_angle_diff=60,
                                      position_tolerance=0.1,
                                      height_tolerance=0.05,
                                      auto_symmetry=True,
                                      adsorbate_formula=None,
                                      verify_structures=True):
    """
    Select NEB endpoints for rotation barrier calculation.
    
    Parameters
    ----------
    site_best : pd.DataFrame
        Best configurations per site type
    screening_results : list
        Full screening results
    rotation_angle_diff : float
        Target rotation angle difference (degrees)
        Common values: 60° (C3v), 90° (C2v), 180° (Cs)
    position_tolerance : float
        Maximum distance to consider "same position" (Å)
    height_tolerance : float
        Tolerance for height matching (Å)
    auto_symmetry : bool
        Warn if angle matches molecular symmetry operation
    adsorbate_formula : str, optional
        If provided, use to check symmetry (e.g., 'NH3')
    verify_structures : bool
        Whether to verify structures before returning
    
    Returns
    -------
    tuple
        (endpoint1_dict, endpoint2_dict) or (None, None) on failure
    """
    # Input validation
    if site_best is None or len(site_best) == 0:
        print("❌ ERROR: site_best is empty!")
        return None, None
    
    if not screening_results or len(screening_results) == 0:
        print("❌ ERROR: No screening results provided!")
        return None, None
    
    # Get recommended angle from molecular symmetry
    if adsorbate_formula is not None:
        recommended = get_recommended_rotation_angle(adsorbate_formula)
        if recommended is None:
            print("⚠️  Rotation may be meaningless for this molecule")
        elif rotation_angle_diff != recommended:
            print(f"⚠️  Note: Using {rotation_angle_diff}°, but {recommended}° is recommended for {adsorbate_formula}")
    
    best_config = site_best.iloc[0]
    target_site_type = best_config['site_type']
    target_height = best_config['height']
    reference_position = np.array(best_config['site_position'][:2])
    
    print(f"\n{'='*60}")
    print(f"Rotation Endpoint Selection")
    print(f"{'='*60}")
    print(f"Reference: {target_site_type} at ({reference_position[0]:.2f}, {reference_position[1]:.2f})")
    print(f"Height: {target_height:.2f} Å")
    print(f"Target rotation difference: {rotation_angle_diff}°")
    
    # Prepare DataFrame
    df = pd.DataFrame(screening_results)
    df = df[df['converged'] == True].copy()
    
    # Find structures at same position with same height
    df['distance'] = df['site_position'].apply(
        lambda pos: np.linalg.norm(np.array(pos[:2]) - reference_position)
    )
    
    candidates = df[
        (df['site_type'] == target_site_type) &
        (df['distance'] < position_tolerance) &
        (np.abs(df['height'] - target_height) < height_tolerance)
    ].copy()
    
    if len(candidates) < 2:
        print(f"❌ ERROR: Only {len(candidates)} rotation(s) at reference position")
        print(f"   Need at least 2 different rotations")
        return None, None
    
    rotation_angles = sorted(candidates['rotation'].unique())
    print(f"\nAvailable rotations: {rotation_angles}")
    
    # Check for symmetry issues
    if auto_symmetry:
        # C3v symmetric angles
        c3v_symmetric = [0, 120, 240, 360]
        if any(abs(rotation_angle_diff - s) < 1 for s in [120, 240]):
            print(f"\n⚠️  WARNING: {rotation_angle_diff}° is a C3v symmetry operation!")
            print("    For NH3/CH3, this gives zero barrier. Consider 60° instead.")
    
    # Build rotation-energy map
    rotation_data = []
    for _, row in candidates.iterrows():
        rotation_data.append({
            'rotation': row['rotation'],
            'energy': row['total_energy'],
            'config': row
        })
    
    rotation_data.sort(key=lambda x: x['rotation'])
    
    # Print energy landscape
    E_min = min(r['energy'] for r in rotation_data)
    print("\nRotation energy landscape:")
    for r in rotation_data:
        dE = (r['energy'] - E_min) * 1000
        bar = '█' * int(dE / 2) if dE > 0 else ''
        print(f"  {r['rotation']:6.1f}°: {dE:6.2f} meV {bar}")
    
    # Find pair closest to target angle difference
    best_pair = None
    best_match = float('inf')
    
    for i, r1 in enumerate(rotation_data):
        for r2 in rotation_data[i+1:]:
            # Handle periodic angles
            diff = min(abs(r2['rotation'] - r1['rotation']), 
                      360 - abs(r2['rotation'] - r1['rotation']))
            match_quality = abs(diff - rotation_angle_diff)
            
            if match_quality < best_match:
                best_match = match_quality
                best_pair = (r1, r2)
    
    if best_pair is None:
        print("❌ ERROR: Could not find suitable rotation pair")
        return None, None
    
    r1, r2 = best_pair
    actual_diff = min(abs(r2['rotation'] - r1['rotation']), 
                     360 - abs(r2['rotation'] - r1['rotation']))
    dE = abs(r2['energy'] - r1['energy']) * 1000
    if r1['energy'] > r2['energy']:
        r1, r2 = r2, r1  # Swap so we go uphill first
    
    print(f"\n✓ Selected: {r1['rotation']:.1f}° → {r2['rotation']:.1f}° (Δ = {actual_diff:.1f}°)")
    print(f"  Energy difference: {dE:.2f} meV")
    
    if abs(actual_diff - rotation_angle_diff) > 5:
        print(f"  ⚠️  Requested {rotation_angle_diff}°, closest available is {actual_diff}°")
    
    if dE < 0.5:
        print(f"  ⚠️  Very small ΔE — rotation barrier may be negligible")
        print(f"       or endpoints may be symmetry-equivalent")
    
    endpoint1 = _to_dict_safe(r1['config'])
    endpoint2 = _to_dict_safe(r2['config'])
    
    # Verify different files
    if endpoint1.get('structure_file') == endpoint2.get('structure_file'):
        print("❌ ERROR: Both endpoints point to the same file!")
        return None, None
    
    # Optional structure verification
    if verify_structures:
        print("\nVerifying structures...")
        try:
            atoms1 = load_structure_from_endpoint(endpoint1)
            atoms2 = load_structure_from_endpoint(endpoint2)
            
            validation = verify_neb_endpoints(atoms1, atoms2)
            
            if not validation['valid']:
                print(f"❌ Endpoint validation failed!")
                for issue in validation['issues']:
                    print(f"   - {issue}")
                return None, None
            
            print(f"  ✓ RMSD: {validation['rmsd']:.3f} Å")
            
            for warning in validation['warnings']:
                print(f"  ⚠️  {warning}")
                
        except Exception as e:
            print(f"⚠️  Could not verify structures: {e}")
    
    print(f"{'='*60}\n")
    
    return endpoint1, endpoint2


# =============================================================================
# NEB CALCULATION
# =============================================================================

def prepare_neb_calculation(endpoint1, endpoint2, 
                            n_images=10,
                            barrier_type='translation',
                            workdir="NEB",
                            fmax=0.05,
                            max_steps=500,
                            climb=True,
                            fix_layers=2,
                            device="cpu",
                            interpolation='idpp'):
    """
    Prepare and run NEB calculation with robust error handling.
    
    Parameters
    ----------
    endpoint1, endpoint2 : dict or ase.Atoms
        NEB endpoints
    n_images : int
        Number of intermediate images
    barrier_type : str
        'translation' or 'rotation' (for labeling)
    workdir : str
        Output directory
    fmax : float
        Force convergence criterion (eV/Å)
    max_steps : int
        Maximum optimization steps
    climb : bool
        Use climbing image NEB
    fix_layers : int
        Number of slab layers to fix
    device : str
        Device for calculator ('cpu' or 'cuda')
    interpolation : str
        'idpp' or 'linear'
    
    Returns
    -------
    tuple
        (images, result_dict)
    """
    os.makedirs(workdir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"NEB Calculation: {barrier_type}")
    print(f"{'='*70}")
    
    # =========================================================================
    # LOAD STRUCTURES
    # =========================================================================
    print("\n[1/6] Loading structures...")
    
    try:
        initial = load_structure_from_endpoint(endpoint1)
        final = load_structure_from_endpoint(endpoint2)
    except Exception as e:
        print(f"❌ Failed to load structures: {e}")
        return None, {'error': str(e)}
    
    print(f"  Initial: {len(initial)} atoms")
    print(f"  Final:   {len(final)} atoms")
    
    # =========================================================================
    # VALIDATE ENDPOINTS
    # =========================================================================
    print("\n[2/6] Validating endpoints...")
    
    validation = verify_neb_endpoints(initial, final)
    
    if not validation['valid']:
        print("❌ Endpoint validation failed!")
        for issue in validation['issues']:
            print(f"   - {issue}")
        return None, {'error': 'Invalid endpoints', 'issues': validation['issues']}
    
    print(f"  ✓ RMSD: {validation['rmsd']:.3f} Å")
    print(f"  ✓ Max displacement: {validation['max_displacement']:.3f} Å")
    
    # =========================================================================
    # FIX PERIODIC BOUNDARY CROSSINGS
    # =========================================================================
    print("\n[3/6] Checking periodic boundaries...")
    
    diff = final.positions - initial.positions
    
    if np.any(initial.pbc):
        diff_mic, _ = find_mic(diff, initial.cell, pbc=initial.pbc)
        max_raw = np.abs(diff).max()
        max_mic = np.abs(diff_mic).max()
        
        if max_raw > max_mic + 0.5:
            print(f"  ⚠️  PBC crossing detected (raw: {max_raw:.2f} Å, MIC: {max_mic:.2f} Å)")
            print(f"  Applying minimum image correction...")
            final.positions = initial.positions + diff_mic
        else:
            print(f"  ✓ No PBC issues")
    
    # =========================================================================
    # APPLY CONSTRAINTS
    # =========================================================================
    print("\n[4/6] Setting up constraints...")
    
    adsorbate_indices = detect_adsorbate_indices(initial)
    print(f"  Detected adsorbate: atoms {adsorbate_indices}")
    
    initial = ensure_constraints(initial, n_fixed_layers=fix_layers, 
                                  adsorbate_indices=adsorbate_indices)
    final = ensure_constraints(final, n_fixed_layers=fix_layers,
                                adsorbate_indices=adsorbate_indices)
    
    # =========================================================================
    # CREATE IMAGES AND CALCULATORS
    # =========================================================================
    print("\n[5/6] Creating NEB images...")
    
    # Assign calculators to endpoints
    initial.calc = fresh_calc(device=device)
    final.calc = fresh_calc(device=device)
    
    E_init = initial.get_potential_energy()
    E_final = final.get_potential_energy()
    
    print(f"  Initial energy: {E_init:.6f} eV")
    print(f"  Final energy:   {E_final:.6f} eV")
    print(f"  ΔE: {(E_final - E_init)*1000:.2f} meV")
    
    # Create images
    images = [initial.copy()]
    for _ in range(n_images):
        img = initial.copy()
        images.append(img)
    images.append(final.copy())
    
    # Assign calculators and constraints to ALL images
    for i, img in enumerate(images):
        img.calc = fresh_calc(device=device)
        if len(initial.constraints) > 0:
            img.set_constraint(deepcopy(initial.constraints[0]))
    
    print(f"  Created {len(images)} images (including endpoints)")
    
    # Interpolation
    neb = NEB(images, climb=False, allow_shared_calculator=False)
    
    print(f"  Interpolating with '{interpolation}' method...")
    
    try:
        if interpolation == 'idpp':
            neb.interpolate('idpp')
        else:
            neb.interpolate()
        
        # Validate interpolation
        for i, img in enumerate(images):
            if np.any(np.isnan(img.positions)) or np.any(np.isinf(img.positions)):
                raise ValueError(f"NaN/Inf in image {i}")
        
        print(f"  ✓ Interpolation successful")
        
    except Exception as e:
        print(f"  ⚠️  {interpolation.upper()} failed: {e}")
        print(f"  Falling back to linear interpolation...")
        
        # Rebuild with linear interpolation
        images = [initial.copy()]
        for _ in range(n_images):
            images.append(initial.copy())
        images.append(final.copy())
        
        for i, img in enumerate(images):
            img.calc = fresh_calc(device=device)
            if len(initial.constraints) > 0:
                img.set_constraint(deepcopy(initial.constraints[0]))
        
        # Manual linear interpolation
        pos_init = initial.get_positions()
        pos_final = final.get_positions()
        
        for i in range(1, n_images + 1):
            frac = i / (n_images + 1)
            images[i].set_positions(pos_init + frac * (pos_final - pos_init))
        
        neb = NEB(images, climb=False, allow_shared_calculator=False)
        print(f"  ✓ Linear interpolation successful")
    
    # =========================================================================
    # RUN NEB OPTIMIZATION
    # =========================================================================
    print("\n[6/6] Running NEB optimization...")
    
    traj_file = os.path.join(workdir, f"neb_{barrier_type}.traj")
    log_file = os.path.join(workdir, f"neb_{barrier_type}.log")
    
    print(f"  Trajectory: {traj_file}")
    print(f"  Log: {log_file}")
    
    optimizer = FIRE(neb, trajectory=traj_file, logfile=log_file, maxstep=0.1)
    
    # Stage 1: Relax without climbing
    print(f"\n  Stage 1: Relaxing path (no climb)...")
    try:
        optimizer.run(fmax=fmax * 2, steps=max_steps // 2)
        stage1_converged = optimizer.converged()
    except Exception as e:
        print(f"  ⚠️  Stage 1 issue: {e}")
        stage1_converged = False
    
    # Stage 2: Climbing image
    if climb:
        print(f"  Stage 2: Refining with climbing image...")
        neb.climb = True
        try:
            optimizer.run(fmax=fmax, steps=max_steps)
            stage2_converged = optimizer.converged()
        except Exception as e:
            print(f"  ⚠️  Stage 2 issue: {e}")
            stage2_converged = False
    else:
        stage2_converged = stage1_converged
    
    converged = stage1_converged or stage2_converged
    
    if converged:
        print(f"\n✓ NEB optimization converged!")
    else:
        print(f"\n⚠️  NEB did not fully converge (results may still be useful)")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\nAnalyzing results...")
    
    from ase.mep import NEBTools
    nebtools = NEBTools(images)
    
    # Get barrier
    barrier_fwd = None
    delta_E = None
    E_ts_abs = None
    
    try:
        barrier_fwd, delta_E = nebtools.get_barrier(fit=True, raw=False)
        print(f"  Forward barrier (fit): {barrier_fwd:.6f} eV ({barrier_fwd*1000:.2f} meV)")
        print(f"  Reaction energy ΔE:    {delta_E:.6f} eV ({delta_E*1000:.2f} meV)")
    except Exception as e:
        print(f"  ⚠️  Could not fit barrier: {e}")
    
    try:
        E_ts_abs, _ = nebtools.get_barrier(fit=True, raw=True)
        print(f"  TS absolute energy:    {E_ts_abs:.6f} eV")
    except:
        pass
    
    # Manual energy calculation as backup
    energies = []
    for img in images:
        try:
            energies.append(img.get_potential_energy())
        except:
            energies.append(np.nan)
    
    energies = np.array(energies)
    
    if barrier_fwd is None and not np.all(np.isnan(energies)):
        barrier_fwd = float(np.nanmax(energies) - energies[0])
        delta_E = float(energies[-1] - energies[0])
        print(f"  Manual barrier: {barrier_fwd*1000:.2f} meV")
    
    # Find saddle point
    saddle_idx = int(np.nanargmax(energies))
    saddle_file = os.path.join(workdir, f"saddle_{barrier_type}.traj")
    write(saddle_file, images[saddle_idx])
    print(f"  Saddle point: image {saddle_idx} → {saddle_file}")
    
    # Save trajectory
    print(f"\n  Full trajectory saved: {traj_file}")
    
    # Plot (with error handling)
    plot_file = None
    try:
        import matplotlib.pyplot as plt
        nebtools.plot_bands()  # This shows the plot or returns None in newer ASE
        fig = plt.gcf()  # Get current figure
        if fig is not None and fig.get_axes():
            plot_file = os.path.join(workdir, f"neb_{barrier_type}_band.png")
            fig.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Band plot saved: {plot_file}")
        else:
            print(f"  ⚠️  No band plot generated")
            plot_file = None
    except Exception as e:
        print(f"  ⚠️  Could not save band plot: {e}")
        plot_file = None
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    result = {
        'barrier_type': barrier_type,
        'forward_barrier_eV': float(barrier_fwd) if barrier_fwd else None,
        'forward_barrier_meV': float(barrier_fwd * 1000) if barrier_fwd else None,
        'delta_E_eV': float(delta_E) if delta_E else None,
        'transition_state_energy': float(E_ts_abs) if E_ts_abs else None,
        'initial_energy': float(E_init),
        'final_energy': float(E_final),
        'trajectory': traj_file,
        'saddle_file': saddle_file,
        'saddle_index': saddle_idx,
        'n_images': len(images),
        'converged': converged,
        'energies_eV': [float(e) if not np.isnan(e) else None for e in energies],
        'plot_file': plot_file,
    }
    
    result = make_json_serializable(result)
    
    summary_file = os.path.join(workdir, "neb_summary.json")
    with open(summary_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n  Summary saved: {summary_file}")
    print(f"{'='*70}\n")
    
    return images, result


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def diagnose_screening_results(screening_results, verbose=True):
    """
    Diagnose screening results to identify potential issues.
    
    Parameters
    ----------
    screening_results : list
        Screening results from site_screening
    verbose : bool
        Print detailed output
    
    Returns
    -------
    dict
        Diagnostic summary
    """
    df = pd.DataFrame(screening_results)
    
    diagnosis = {
        'total_configs': len(df),
        'converged': len(df[df['converged'] == True]),
        'site_types': {},
        'issues': [],
        'warnings': [],
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("Screening Results Diagnosis")
        print(f"{'='*60}")
        print(f"\nTotal configurations: {diagnosis['total_configs']}")
        print(f"Converged: {diagnosis['converged']}")
    
    df_conv = df[df['converged'] == True]
    
    if len(df_conv) == 0:
        diagnosis['issues'].append("No converged structures!")
        if verbose:
            print("❌ No converged structures!")
        return diagnosis
    
    # Analyze by site type
    if verbose:
        print(f"\nEnergy by site type:")
    
    for st in df_conv['site_type'].unique():
        subset = df_conv[df_conv['site_type'] == st]
        E_min = subset['total_energy'].min()
        E_max = subset['total_energy'].max()
        E_range = (E_max - E_min) * 1000
        
        diagnosis['site_types'][st] = {
            'count': len(subset),
            'E_min': float(E_min),
            'E_max': float(E_max),
            'E_range_meV': float(E_range),
        }
        
        if verbose:
            print(f"  {st:10s}: n={len(subset):3d}, "
                  f"E_min={E_min:.4f} eV, range={E_range:.1f} meV")
    
    # Check energy separation between site types
    site_min_energies = {st: info['E_min'] for st, info in diagnosis['site_types'].items()}
    
    if len(site_min_energies) >= 2:
        E_values = list(site_min_energies.values())
        max_separation = (max(E_values) - min(E_values)) * 1000
        
        if verbose:
            print(f"\nSite type separation: {max_separation:.1f} meV")
        
        if max_separation < 5:
            msg = f"Very small separation between site types ({max_separation:.1f} meV)"
            diagnosis['warnings'].append(msg)
            if verbose:
                print(f"  ⚠️  {msg}")
                print("      ML potential may not distinguish sites well")
    
    if verbose:
        print(f"{'='*60}\n")
    
    return diagnosis