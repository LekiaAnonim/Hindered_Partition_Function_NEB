"""
NEB endpoint selection and calculation functions (v3).
Fixed to ensure consistent starting structure for both translation and rotation.
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
    
    Returns
    -------
    float or None
        Recommended rotation angle in degrees, or None if rotation is meaningless
    """
    if adsorbate_formula in MOLECULAR_SYMMETRY:
        info = MOLECULAR_SYMMETRY[adsorbate_formula]
        if info['barrier_angle'] is None:
            print(f"  {adsorbate_formula} has {info['symmetry']} symmetry - rotation barrier is zero/undefined")
            return None
        print(f"  {adsorbate_formula} ({info['symmetry']}): recommended rotation = {info['barrier_angle']}deg")
        return info['barrier_angle']
    else:
        print(f"  Unknown molecule '{adsorbate_formula}', defaulting to 60deg")
        return 60


def detect_adsorbate_indices(atoms, n_slab_atoms=None, z_threshold=3.0):
    """Detect which atoms are the adsorbate vs slab."""
    if n_slab_atoms is not None:
        return list(range(n_slab_atoms, len(atoms)))
    
    tag_zero = [a.index for a in atoms if a.tag == 0]
    if tag_zero and len(tag_zero) < len(atoms) // 2:
        return tag_zero
    
    z = atoms.positions[:, 2]
    z_max = z.max()
    high_atoms = set(np.where(z > z_max - z_threshold)[0])
    non_metal_indices = {a.index for a in atoms if a.symbol not in METAL_ELEMENTS}
    
    adsorbate_indices = list(high_atoms & non_metal_indices)
    
    if adsorbate_indices:
        return sorted(adsorbate_indices)
    
    return sorted(list(high_atoms))


def ensure_constraints(atoms, n_fixed_layers=2, adsorbate_indices=None):
    """Ensure slab atoms are fixed with FixAtoms constraint."""
    if len(atoms.constraints) > 0:
        for c in atoms.constraints:
            if isinstance(c, FixAtoms):
                print(f"    Constraints already present: {len(c.index)} atoms fixed")
                return atoms
    
    print("    Applying constraints...")
    
    if adsorbate_indices is None:
        adsorbate_indices = detect_adsorbate_indices(atoms)
    
    adsorbate_set = set(adsorbate_indices)
    
    fix_indices = [a.index for a in atoms if a.tag > n_fixed_layers and a.index not in adsorbate_set]
    
    if not fix_indices:
        slab_indices = [i for i in range(len(atoms)) if i not in adsorbate_set]
        if slab_indices:
            z_slab = atoms.positions[slab_indices, 2]
            z_cutoff = np.percentile(z_slab, 60)
            fix_indices = [i for i in slab_indices if atoms.positions[i, 2] < z_cutoff]
    
    if fix_indices:
        atoms.set_constraint(FixAtoms(indices=fix_indices))
        print(f"    Fixed {len(fix_indices)} atoms, {len(adsorbate_indices)} adsorbate atoms free")
    else:
        print("    WARNING: Could not determine which atoms to fix!")
    
    return atoms


def verify_neb_endpoints(atoms1, atoms2, name="endpoints"):
    """Comprehensive validation of NEB endpoints."""
    results = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'rmsd': None,
        'max_displacement': None,
    }
    
    if len(atoms1) != len(atoms2):
        results['valid'] = False
        results['issues'].append(f"Different atom counts: {len(atoms1)} vs {len(atoms2)}")
        return results
    
    sym1 = atoms1.get_chemical_symbols()
    sym2 = atoms2.get_chemical_symbols()
    if sym1 != sym2:
        results['valid'] = False
        results['issues'].append("Different atomic species or order")
        return results
    
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
    
    diff = pos2 - pos1
    
    if np.any(atoms1.pbc):
        diff_mic, _ = find_mic(diff, atoms1.cell, pbc=atoms1.pbc)
        diff = diff_mic
    
    rmsd = np.sqrt(np.mean(diff**2))
    max_disp = np.max(np.linalg.norm(diff, axis=1))
    
    results['rmsd'] = float(rmsd)
    results['max_displacement'] = float(max_disp)
    
    if max_disp < 0.05:
        results['valid'] = False
        results['issues'].append(f"Endpoints too similar (max displacement: {max_disp:.4f} A)")
    elif max_disp < 0.2:
        results['warnings'].append(f"Small displacement ({max_disp:.3f} A) - barrier may be tiny")
    
    c1 = atoms1.constraints
    c2 = atoms2.constraints
    
    if len(c1) != len(c2):
        results['warnings'].append(f"Different constraint counts: {len(c1)} vs {len(c2)}")
    
    return results


def load_structure_from_endpoint(endpoint):
    """Load ASE Atoms from endpoint (dict or Atoms)."""
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
# GLOBAL MINIMUM SELECTION
# =============================================================================

def get_global_minimum(screening_results, site_best=None):
    """
    Get the global minimum configuration from screening results.
    
    This is the SINGLE reference structure that should be used as
    endpoint1 for BOTH translation AND rotation NEB calculations.
    
    Parameters
    ----------
    screening_results : list
        Full screening results
    site_best : pd.DataFrame, optional
        Pre-computed best per site type (will use iloc[0] if provided)
    
    Returns
    -------
    pd.Series
        Configuration of the global minimum
    """
    df = pd.DataFrame(screening_results)
    df_conv = df[df['converged'] == True]
    
    if len(df_conv) == 0:
        raise ValueError("No converged structures in screening results!")
    
    best_idx = df_conv['total_energy'].idxmin()
    return df_conv.loc[best_idx]


# =============================================================================
# ENDPOINT SELECTION FUNCTIONS
# =============================================================================

def select_neb_endpoints_translation(screening_results, 
                                     reference_config=None,
                                     method='cross_site',
                                     min_crossing_distance=2.5,
                                     height_tolerance=0.05,
                                     rotation_tolerance=1.0,
                                     verify_structures=True):
    """
    Select NEB endpoints for translation barrier calculation.
    
    Parameters
    ----------
    screening_results : list
        Full screening results
    reference_config : pd.Series, optional
        Reference configuration (global minimum). If None, computed automatically.
    method : str
        'cross_site': Path to different site type (recommended)
        'long_path': Same site type but crosses intermediate site
    min_crossing_distance : float
        Minimum distance for 'long_path' method (A)
    height_tolerance : float
        Tolerance for height matching (A)
    rotation_tolerance : float
        Tolerance for rotation matching (degrees)
    verify_structures : bool
        Whether to verify structures before returning
    
    Returns
    -------
    tuple
        (endpoint1_dict, endpoint2_dict, info_dict) or (None, None, error_dict)
    """
    info = {'method': method, 'barrier_type': 'translation'}
    
    if not screening_results or len(screening_results) == 0:
        return None, None, {'error': 'No screening results provided'}
    
    df = pd.DataFrame(screening_results)
    df_conv = df[df['converged'] == True].copy()
    
    if len(df_conv) == 0:
        return None, None, {'error': 'No converged structures'}
    
    # Get reference (global minimum)
    if reference_config is None:
        reference_config = get_global_minimum(screening_results)
    
    target_site_type = reference_config['site_type']
    target_height = reference_config['height']
    target_rotation = reference_config['rotation']
    ref_position = np.array(reference_config['site_position'][:2])
    ref_energy = reference_config['total_energy']
    
    print(f"\n{'='*60}")
    print(f"Translation Endpoint Selection (method='{method}')")
    print(f"{'='*60}")
    print(f"Reference: {target_site_type} at ({ref_position[0]:.2f}, {ref_position[1]:.2f})")
    print(f"Height: {target_height:.2f} A, Rotation: {target_rotation:.1f} deg")
    print(f"Energy: {ref_energy:.6f} eV")
    
    # Calculate distances from reference
    df_conv['distance'] = df_conv['site_position'].apply(
        lambda pos: np.linalg.norm(np.array(pos[:2]) - ref_position)
    )
    df_conv['dE_meV'] = (df_conv['total_energy'] - ref_energy) * 1000
    
    if method == 'cross_site':
        # Find path to DIFFERENT site type
        other_sites = df_conv[df_conv['site_type'] != target_site_type].copy()
        
        if len(other_sites) == 0:
            return None, None, {'error': f'No other site types found (only {target_site_type})'}
        
        site_types_available = other_sites['site_type'].unique()
        print(f"\nOther site types: {list(site_types_available)}")
        
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
            print(f"  {target_site_type} -> {st}: d={nearest['distance']:.2f} A, dE={nearest['dE_meV']:.1f} meV")
        
        # Select candidate with highest energy (largest expected barrier)
        candidates.sort(key=lambda x: x['dE_meV'], reverse=True)
        selected = candidates[0]
        
        if selected['dE_meV'] < 1.0:
            print(f"\n  WARNING: Very small energy difference ({selected['dE_meV']:.2f} meV)")
        
        print(f"\n  Selected: {target_site_type} -> {selected['site_type']}")
        print(f"  Distance: {selected['distance']:.2f} A")
        print(f"  Barrier estimate: {selected['dE_meV']:.1f} meV")
        
        endpoint2_config = selected['config']
        info['final_site_type'] = selected['site_type']
        info['distance'] = selected['distance']
        info['dE_meV'] = selected['dE_meV']
    
    elif method == 'long_path':
        # Same site type but longer path crossing intermediate site
        same_sites = df_conv[
            (df_conv['site_type'] == target_site_type) &
            (np.abs(df_conv['height'] - target_height) < height_tolerance) &
            (np.abs(df_conv['rotation'] - target_rotation) < rotation_tolerance)
        ].copy()
        
        print(f"\nSame-type sites with matching geometry: {len(same_sites)}")
        
        far_sites = same_sites[same_sites['distance'] > min_crossing_distance]
        
        if len(far_sites) == 0:
            return None, None, {'error': f'No {target_site_type} sites beyond {min_crossing_distance} A'}
        
        selected = far_sites.nsmallest(1, 'distance').iloc[0]
        
        print(f"\n  Selected: {target_site_type} -> {target_site_type}")
        print(f"  Distance: {selected['distance']:.2f} A (crosses intermediate site)")
        print(f"  dE: {selected['dE_meV']:.2f} meV")
        
        endpoint2_config = selected
        info['distance'] = selected['distance']
        info['dE_meV'] = selected['dE_meV']
    
    else:
        return None, None, {'error': f"Unknown method '{method}'"}
    
    endpoint1 = _to_dict_safe(reference_config)
    endpoint2 = _to_dict_safe(endpoint2_config)
    
    # Verify different files
    if endpoint1.get('structure_file') == endpoint2.get('structure_file'):
        return None, None, {'error': 'Both endpoints point to the same file'}
    
    # Structure verification
    if verify_structures:
        print("\nVerifying structures...")
        try:
            atoms1 = load_structure_from_endpoint(endpoint1)
            atoms2 = load_structure_from_endpoint(endpoint2)
            
            validation = verify_neb_endpoints(atoms1, atoms2)
            
            if not validation['valid']:
                return None, None, {'error': 'Endpoint validation failed', 'issues': validation['issues']}
            
            print(f"    RMSD: {validation['rmsd']:.3f} A")
            print(f"    Max displacement: {validation['max_displacement']:.3f} A")
            
            info['rmsd'] = validation['rmsd']
            info['max_displacement'] = validation['max_displacement']
            
            for warning in validation['warnings']:
                print(f"    WARNING: {warning}")
                
        except Exception as e:
            print(f"  WARNING: Could not verify structures: {e}")
    
    print(f"{'='*60}\n")
    
    return endpoint1, endpoint2, info


def select_neb_endpoints_rotation(screening_results,
                                  reference_config=None,
                                  rotation_angle_diff=None,
                                  position_tolerance=0.1,
                                  height_tolerance=0.05,
                                  adsorbate_formula=None,
                                  verify_structures=True):
    """
    Select NEB endpoints for rotation barrier calculation.
    
    IMPORTANT: Uses the SAME reference_config as translation to ensure
    both barriers are computed from the same starting structure.
    
    Parameters
    ----------
    screening_results : list
        Full screening results
    reference_config : pd.Series, optional
        Reference configuration (global minimum). If None, computed automatically.
        MUST be the same as used for translation endpoints!
    rotation_angle_diff : float, optional
        Target rotation angle difference (degrees). If None, determined from
        adsorbate_formula symmetry.
    position_tolerance : float
        Maximum distance to consider "same position" (A)
    height_tolerance : float
        Tolerance for height matching (A)
    adsorbate_formula : str, optional
        Chemical formula (e.g., 'NH3') for symmetry-based angle selection
    verify_structures : bool
        Whether to verify structures before returning
    
    Returns
    -------
    tuple
        (endpoint1_dict, endpoint2_dict, info_dict) or (None, None, error_dict)
    """
    info = {'barrier_type': 'rotation'}
    
    if not screening_results or len(screening_results) == 0:
        return None, None, {'error': 'No screening results provided'}
    
    # Get recommended angle from molecular symmetry
    if rotation_angle_diff is None:
        if adsorbate_formula is not None:
            rotation_angle_diff = get_recommended_rotation_angle(adsorbate_formula)
            if rotation_angle_diff is None:
                return None, None, {'error': f'Rotation meaningless for {adsorbate_formula}'}
        else:
            rotation_angle_diff = 60  # Default
            print(f"  No adsorbate_formula provided, using default {rotation_angle_diff} deg")
    
    info['target_angle_diff'] = rotation_angle_diff
    
    df = pd.DataFrame(screening_results)
    df_conv = df[df['converged'] == True].copy()
    
    if len(df_conv) == 0:
        return None, None, {'error': 'No converged structures'}
    
    # Get reference (global minimum) - SAME as translation!
    if reference_config is None:
        reference_config = get_global_minimum(screening_results)
    
    target_site_type = reference_config['site_type']
    target_height = reference_config['height']
    ref_rotation = reference_config['rotation']
    ref_position = np.array(reference_config['site_position'][:2])
    ref_energy = reference_config['total_energy']
    
    print(f"\n{'='*60}")
    print(f"Rotation Endpoint Selection")
    print(f"{'='*60}")
    print(f"Reference: {target_site_type} at ({ref_position[0]:.2f}, {ref_position[1]:.2f})")
    print(f"Height: {target_height:.2f} A, Rotation: {ref_rotation:.1f} deg")
    print(f"Energy: {ref_energy:.6f} eV")
    print(f"Target rotation difference: {rotation_angle_diff} deg")
    
    # Find structures at same position with same height
    df_conv['distance'] = df_conv['site_position'].apply(
        lambda pos: np.linalg.norm(np.array(pos[:2]) - ref_position)
    )
    
    candidates = df_conv[
        (df_conv['site_type'] == target_site_type) &
        (df_conv['distance'] < position_tolerance) &
        (np.abs(df_conv['height'] - target_height) < height_tolerance)
    ].copy()
    
    if len(candidates) < 2:
        return None, None, {'error': f'Only {len(candidates)} rotation(s) at reference position, need >= 2'}
    
    rotation_angles = sorted(candidates['rotation'].unique())
    print(f"\nAvailable rotations at this site: {rotation_angles}")
    
    # Print energy landscape
    print("\nRotation energy landscape:")
    for _, row in candidates.sort_values('rotation').iterrows():
        dE = (row['total_energy'] - ref_energy) * 1000
        bar = '*' * int(min(dE / 2, 20)) if dE > 0 else ''
        marker = ' <-- reference' if abs(row['rotation'] - ref_rotation) < 0.1 else ''
        print(f"  {row['rotation']:6.1f} deg: {dE:6.2f} meV {bar}{marker}")
    
    # Find best partner for the reference rotation
    best_partner = None
    best_match = float('inf')
    
    for _, row in candidates.iterrows():
        # Skip the reference itself
        if abs(row['rotation'] - ref_rotation) < 0.1:
            continue
        
        # Calculate angle difference (handle periodicity)
        diff = min(abs(row['rotation'] - ref_rotation),
                   360 - abs(row['rotation'] - ref_rotation))
        
        match_quality = abs(diff - rotation_angle_diff)
        
        if match_quality < best_match:
            best_match = match_quality
            best_partner = row
            actual_diff = diff
    
    if best_partner is None:
        return None, None, {'error': 'Could not find rotation partner'}
    
    dE = (best_partner['total_energy'] - ref_energy) * 1000
    
    print(f"\n  Selected: {ref_rotation:.1f} deg -> {best_partner['rotation']:.1f} deg (delta = {actual_diff:.1f} deg)")
    print(f"  Energy difference: {dE:.2f} meV")
    
    if abs(actual_diff - rotation_angle_diff) > 5:
        print(f"  NOTE: Requested {rotation_angle_diff} deg, closest available is {actual_diff:.1f} deg")
    
    if abs(dE) < 0.5:
        print(f"  WARNING: Very small dE - rotation barrier may be negligible")
        print(f"           or endpoints may be symmetry-equivalent")
    
    info['initial_rotation'] = ref_rotation
    info['final_rotation'] = best_partner['rotation']
    info['actual_angle_diff'] = actual_diff
    info['dE_meV'] = dE
    
    # CRITICAL: endpoint1 is the reference (same as translation!)
    endpoint1 = _to_dict_safe(reference_config)
    endpoint2 = _to_dict_safe(best_partner)
    
    # Verify different files
    if endpoint1.get('structure_file') == endpoint2.get('structure_file'):
        return None, None, {'error': 'Both endpoints point to the same file'}
    
    # Structure verification
    if verify_structures:
        print("\nVerifying structures...")
        try:
            atoms1 = load_structure_from_endpoint(endpoint1)
            atoms2 = load_structure_from_endpoint(endpoint2)
            
            validation = verify_neb_endpoints(atoms1, atoms2)
            
            if not validation['valid']:
                return None, None, {'error': 'Endpoint validation failed', 'issues': validation['issues']}
            
            print(f"    RMSD: {validation['rmsd']:.3f} A")
            
            info['rmsd'] = validation['rmsd']
            
            for warning in validation['warnings']:
                print(f"    WARNING: {warning}")
                
        except Exception as e:
            print(f"  WARNING: Could not verify structures: {e}")
    
    print(f"{'='*60}\n")
    
    return endpoint1, endpoint2, info


# =============================================================================
# UNIFIED ENDPOINT SELECTION
# =============================================================================

def select_all_neb_endpoints(screening_results,
                             adsorbate_formula=None,
                             translation_method='cross_site',
                             rotation_angle_diff=None,
                             verify_structures=True):
    """
    Select endpoints for BOTH translation and rotation NEB calculations.
    
    Ensures both calculations use the SAME starting structure (global minimum).
    
    Parameters
    ----------
    screening_results : list
        Full screening results from site_screening
    adsorbate_formula : str, optional
        Chemical formula for symmetry-based rotation angle
    translation_method : str
        'cross_site' or 'long_path'
    rotation_angle_diff : float, optional
        Override rotation angle (otherwise determined from symmetry)
    verify_structures : bool
        Whether to verify structures
    
    Returns
    -------
    dict
        {
            'reference': reference_config,
            'translation': (endpoint1, endpoint2, info),
            'rotation': (endpoint1, endpoint2, info),
        }
    """
    print("\n" + "="*70)
    print("UNIFIED NEB ENDPOINT SELECTION")
    print("="*70)
    
    # Get the SINGLE global minimum
    reference_config = get_global_minimum(screening_results)
    
    print(f"\nGlobal minimum:")
    print(f"  Site type: {reference_config['site_type']}")
    print(f"  Position: ({reference_config['site_position'][0]:.3f}, {reference_config['site_position'][1]:.3f})")
    print(f"  Height: {reference_config['height']:.3f} A")
    print(f"  Rotation: {reference_config['rotation']:.1f} deg")
    print(f"  Energy: {reference_config['total_energy']:.6f} eV")
    
    if 'structure_file' in reference_config:
        print(f"  Structure: {reference_config['structure_file']}")
    
    results = {
        'reference': _to_dict_safe(reference_config),
        'translation': None,
        'rotation': None,
    }
    
    # Select translation endpoints
    trans_ep1, trans_ep2, trans_info = select_neb_endpoints_translation(
        screening_results,
        reference_config=reference_config,
        method=translation_method,
        verify_structures=verify_structures
    )
    
    if trans_ep1 is not None:
        results['translation'] = (trans_ep1, trans_ep2, trans_info)
        print(f"  Translation endpoints: OK")
    else:
        print(f"  Translation endpoints: FAILED - {trans_info.get('error', 'unknown')}")
        results['translation'] = (None, None, trans_info)
    
    # Select rotation endpoints (using SAME reference!)
    rot_ep1, rot_ep2, rot_info = select_neb_endpoints_rotation(
        screening_results,
        reference_config=reference_config,
        rotation_angle_diff=rotation_angle_diff,
        adsorbate_formula=adsorbate_formula,
        verify_structures=verify_structures
    )
    
    if rot_ep1 is not None:
        results['rotation'] = (rot_ep1, rot_ep2, rot_info)
        print(f"  Rotation endpoints: OK")
    else:
        print(f"  Rotation endpoints: FAILED - {rot_info.get('error', 'unknown')}")
        results['rotation'] = (None, None, rot_info)
    
    # Verify consistency
    if trans_ep1 is not None and rot_ep1 is not None:
        trans_file = trans_ep1.get('structure_file')
        rot_file = rot_ep1.get('structure_file')
        
        if trans_file == rot_file:
            print(f"\n  VERIFIED: Both NEB calculations use same initial structure")
        else:
            print(f"\n  ERROR: Initial structures differ!")
            print(f"    Translation: {trans_file}")
            print(f"    Rotation: {rot_file}")
    
    print("="*70 + "\n")
    
    return results


# =============================================================================
# NEB CALCULATION
# =============================================================================

def run_neb_calculation(endpoint1, endpoint2, 
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
    Run NEB calculation with robust error handling.
    
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
        Force convergence criterion (eV/A)
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
    
    # Load structures
    print("\n[1/6] Loading structures...")
    
    try:
        initial = load_structure_from_endpoint(endpoint1)
        final = load_structure_from_endpoint(endpoint2)
    except Exception as e:
        print(f"  FAILED: {e}")
        return None, {'error': str(e)}
    
    print(f"  Initial: {len(initial)} atoms")
    print(f"  Final:   {len(final)} atoms")
    
    # Validate endpoints
    print("\n[2/6] Validating endpoints...")
    
    validation = verify_neb_endpoints(initial, final)
    
    if not validation['valid']:
        print("  FAILED!")
        for issue in validation['issues']:
            print(f"    - {issue}")
        return None, {'error': 'Invalid endpoints', 'issues': validation['issues']}
    
    print(f"    RMSD: {validation['rmsd']:.3f} A")
    print(f"    Max displacement: {validation['max_displacement']:.3f} A")
    
    # Fix periodic boundary crossings
    print("\n[3/6] Checking periodic boundaries...")
    
    diff = final.positions - initial.positions
    
    if np.any(initial.pbc):
        diff_mic, _ = find_mic(diff, initial.cell, pbc=initial.pbc)
        max_raw = np.abs(diff).max()
        max_mic = np.abs(diff_mic).max()
        
        if max_raw > max_mic + 0.5:
            print(f"    PBC crossing detected (raw: {max_raw:.2f} A, MIC: {max_mic:.2f} A)")
            print(f"    Applying minimum image correction...")
            final.positions = initial.positions + diff_mic
        else:
            print(f"    No PBC issues")
    
    # Apply constraints
    print("\n[4/6] Setting up constraints...")
    
    adsorbate_indices = detect_adsorbate_indices(initial)
    print(f"  Detected adsorbate: atoms {adsorbate_indices}")
    
    initial = ensure_constraints(initial, n_fixed_layers=fix_layers, 
                                  adsorbate_indices=adsorbate_indices)
    final = ensure_constraints(final, n_fixed_layers=fix_layers,
                                adsorbate_indices=adsorbate_indices)
    
    # Create images
    print("\n[5/6] Creating NEB images...")
    
    initial.calc = fresh_calc(device=device)
    final.calc = fresh_calc(device=device)
    
    E_init = initial.get_potential_energy()
    E_final = final.get_potential_energy()
    
    print(f"  Initial energy: {E_init:.6f} eV")
    print(f"  Final energy:   {E_final:.6f} eV")
    print(f"  dE: {(E_final - E_init)*1000:.2f} meV")
    
    images = [initial.copy()]
    for _ in range(n_images):
        img = initial.copy()
        images.append(img)
    images.append(final.copy())
    
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
        
        for i, img in enumerate(images):
            if np.any(np.isnan(img.positions)) or np.any(np.isinf(img.positions)):
                raise ValueError(f"NaN/Inf in image {i}")
        
        print(f"    Interpolation successful")
        
    except Exception as e:
        print(f"    {interpolation.upper()} failed: {e}")
        print(f"    Falling back to linear interpolation...")
        
        images = [initial.copy()]
        for _ in range(n_images):
            images.append(initial.copy())
        images.append(final.copy())
        
        for i, img in enumerate(images):
            img.calc = fresh_calc(device=device)
            if len(initial.constraints) > 0:
                img.set_constraint(deepcopy(initial.constraints[0]))
        
        pos_init = initial.get_positions()
        pos_final = final.get_positions()
        
        for i in range(1, n_images + 1):
            frac = i / (n_images + 1)
            images[i].set_positions(pos_init + frac * (pos_final - pos_init))
        
        neb = NEB(images, climb=False, allow_shared_calculator=False)
        print(f"    Linear interpolation successful")
    
    # Run NEB optimization
    print("\n[6/6] Running NEB optimization...")
    
    traj_file = os.path.join(workdir, f"neb_{barrier_type}.traj")
    log_file = os.path.join(workdir, f"neb_{barrier_type}.log")
    
    print(f"  Trajectory: {traj_file}")
    print(f"  Log: {log_file}")
    
    optimizer = FIRE(neb, trajectory=traj_file, logfile=log_file, maxstep=0.1)
    
    print(f"\n  Stage 1: Relaxing path (no climb)...")
    try:
        optimizer.run(fmax=fmax * 2, steps=max_steps // 2)
        stage1_converged = optimizer.converged()
    except Exception as e:
        print(f"    Stage 1 issue: {e}")
        stage1_converged = False
    
    if climb:
        print(f"  Stage 2: Refining with climbing image...")
        neb.climb = True
        try:
            optimizer.run(fmax=fmax, steps=max_steps)
            stage2_converged = optimizer.converged()
        except Exception as e:
            print(f"    Stage 2 issue: {e}")
            stage2_converged = False
    else:
        stage2_converged = stage1_converged
    
    converged = stage1_converged or stage2_converged
    
    if converged:
        print(f"\n  NEB optimization converged!")
    else:
        print(f"\n  NEB did not fully converge (results may still be useful)")
    
    # Analysis
    print("\nAnalyzing results...")
    
    from ase.mep import NEBTools
    nebtools = NEBTools(images)
    
    barrier_fwd = None
    delta_E = None
    E_ts_abs = None
    
    try:
        barrier_fwd, delta_E = nebtools.get_barrier(fit=True, raw=False)
        print(f"  Forward barrier (fit): {barrier_fwd:.6f} eV ({barrier_fwd*1000:.2f} meV)")
        print(f"  Reaction energy dE:    {delta_E:.6f} eV ({delta_E*1000:.2f} meV)")
    except Exception as e:
        print(f"  Could not fit barrier: {e}")
    
    try:
        E_ts_abs, _ = nebtools.get_barrier(fit=True, raw=True)
        print(f"  TS absolute energy:    {E_ts_abs:.6f} eV")
    except:
        pass
    
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
    
    saddle_idx = int(np.nanargmax(energies))
    saddle_file = os.path.join(workdir, f"saddle_{barrier_type}.traj")
    write(saddle_file, images[saddle_idx])
    print(f"  Saddle point: image {saddle_idx} -> {saddle_file}")
    
    print(f"\n  Full trajectory saved: {traj_file}")
    
    # Plot
    plot_file = None
    try:
        import matplotlib.pyplot as plt
        nebtools.plot_bands()
        fig = plt.gcf()
        if fig is not None and fig.get_axes():
            plot_file = os.path.join(workdir, f"neb_{barrier_type}_band.png")
            fig.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Band plot saved: {plot_file}")
    except Exception as e:
        print(f"  Could not save band plot: {e}")
    
    # Save results
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
    """Diagnose screening results to identify potential issues."""
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
            print("  ERROR: No converged structures!")
        return diagnosis
    
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
                print(f"    WARNING: {msg}")
    
    if verbose:
        print(f"{'='*60}\n")
    
    return diagnosis