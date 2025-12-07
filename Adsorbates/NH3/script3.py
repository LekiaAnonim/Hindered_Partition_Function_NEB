#!/usr/bin/env python3
"""
NEB calculation script for NH3 on Pt(111)
Uses unified endpoint selection to ensure consistent starting structure.
"""

import sys
import numpy as np

sys.path.insert(0, '/projects/westgroup/lekia.p/NEB/Adsorbates')

# Import from original neb.py (utilities we still need)
from moels.neb import (
    validate_screening_files,
    clean_incomplete_files,
    recover_screening_files,
    load_screening_results,
    best_site_results,
)

# Import from corrected neb3.py
from models.neb3 import (
    select_all_neb_endpoints,
    run_neb_calculation,
    diagnose_screening_results,
    get_recommended_rotation_angle,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

WORKDIR = '/projects/westgroup/lekia.p/NEB/Adsorbates/NH3'
SCREENING_DIR = f'{WORKDIR}/Screening_Data'
ADSORBATE = 'NH3'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def remap_paths(screening_results, old_prefix, new_prefix):
    """Remap file paths from cluster to local."""
    for result in screening_results:
        if 'structure_file' in result:
            result['structure_file'] = result['structure_file'].replace(
                old_prefix, new_prefix
            )
    return screening_results

# =============================================================================
# MAIN
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Step 1: Load screening results
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 1: Loading Screening Results")
    print("="*70)
    
    validation = validate_screening_files(SCREENING_DIR)
    recover_screening_files(SCREENING_DIR)
    
    screening_results = load_screening_results(f'{SCREENING_DIR}/screening_results.pkl')
    # screening_results = remap_paths(
    #     screening_results,
    #     '/projects/westgroup/lekia.p/NEB/Adsorbates/NH3',
    #     WORKDIR
    # )
    
    print(f"Loaded {len(screening_results)} configurations")
    
    # -------------------------------------------------------------------------
    # Step 2: Diagnose screening results
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 2: Diagnosing Screening Results")
    print("="*70)
    
    diagnosis = diagnose_screening_results(screening_results)
    
    if diagnosis['issues']:
        print("\nCritical issues found:")
        for issue in diagnosis['issues']:
            print(f"   - {issue}")
        print("\nFix issues before proceeding!")
        return None, None
    
    # -------------------------------------------------------------------------
    # Step 3: Select endpoints for BOTH calculations (unified)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 3: Selecting NEB Endpoints (Unified)")
    print("="*70)
    
    # This ensures both translation and rotation use the SAME starting structure
    endpoints = select_all_neb_endpoints(
        screening_results,
        adsorbate_formula=ADSORBATE,
        translation_method='long_path',
        rotation_angle_diff=None,  # Auto-detect from symmetry (60° for NH3)
        verify_structures=True
    )
    
    # Extract endpoints
    trans_ep1, trans_ep2, trans_info = endpoints['translation']
    rot_ep1, rot_ep2, rot_info = endpoints['rotation']
    
    # Print reference structure info
    ref = endpoints['reference']
    print(f"\nReference structure (used for both NEB calculations):")
    print(f"  Site: {ref['site_type']}")
    print(f"  Energy: {ref['total_energy']:.6f} eV")
    print(f"  File: {ref.get('structure_file', 'N/A')}")
    
    # -------------------------------------------------------------------------
    # Step 4: Translation barrier NEB
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 4: Translation Barrier Calculation")
    print("="*70)
    
    result_trans = None
    
    if trans_ep1 is None:
        print(f"\nFailed to select translation endpoints: {trans_info.get('error', 'unknown')}")
        print("Skipping translation NEB...")
    else:
        print(f"\nTranslation path info:")
        print(f"  Method: {trans_info.get('method', 'N/A')}")
        print(f"  Distance: {trans_info.get('distance', 'N/A'):.2f} A")
        print(f"  Estimated dE: {trans_info.get('dE_meV', 'N/A'):.1f} meV")
        
        images_trans, result_trans = run_neb_calculation(
            trans_ep1, 
            trans_ep2,
            n_images=10,
            barrier_type='translation',
            workdir=f'{WORKDIR}/NEB_Translation',
            fmax=0.05,
            climb=True,
        )
        
        if result_trans and 'error' not in result_trans:
            barrier = result_trans.get('forward_barrier_meV')
            if barrier is not None:
                print(f"\nTranslation barrier: {barrier:.2f} meV")
        else:
            error_msg = result_trans.get('error', 'Unknown') if result_trans else 'No result'
            print(f"\nTranslation NEB failed: {error_msg}")
    
    # -------------------------------------------------------------------------
    # Step 5: Rotation barrier NEB
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 5: Rotation Barrier Calculation")
    print("="*70)
    
    result_rot = None
    
    if rot_ep1 is None:
        print(f"\nFailed to select rotation endpoints: {rot_info.get('error', 'unknown')}")
        print("Skipping rotation NEB...")
    else:
        print(f"\nRotation path info:")
        print(f"  Initial rotation: {rot_info.get('initial_rotation', 'N/A'):.1f} deg")
        print(f"  Final rotation: {rot_info.get('final_rotation', 'N/A'):.1f} deg")
        print(f"  Angle difference: {rot_info.get('actual_angle_diff', 'N/A'):.1f} deg")
        print(f"  Estimated dE: {rot_info.get('dE_meV', 'N/A'):.1f} meV")
        
        images_rot, result_rot = run_neb_calculation(
            rot_ep1,
            rot_ep2,
            n_images=10,
            barrier_type='rotation',
            workdir=f'{WORKDIR}/NEB_Rotation',
            fmax=0.05,
            climb=True,
        )
        
        if result_rot and 'error' not in result_rot:
            barrier = result_rot.get('forward_barrier_meV')
            if barrier is not None:
                print(f"\nRotation barrier: {barrier:.2f} meV")
        else:
            error_msg = result_rot.get('error', 'Unknown') if result_rot else 'No result'
            print(f"\nRotation NEB failed: {error_msg}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nAdsorbate: {ADSORBATE}")
    print(f"Reference site: {ref['site_type']}")
    print(f"Reference energy: {ref['total_energy']:.6f} eV")
    
    if result_trans and 'error' not in result_trans:
        print(f"\nTranslation barrier:")
        print(f"  Forward barrier: {result_trans.get('forward_barrier_meV', 'N/A'):.2f} meV")
        print(f"  Converged: {result_trans.get('converged', False)}")
        print(f"  Saddle file: {result_trans.get('saddle_file', 'N/A')}")
    else:
        print(f"\nTranslation barrier: NOT CALCULATED")
    
    if result_rot and 'error' not in result_rot:
        print(f"\nRotation barrier:")
        print(f"  Forward barrier: {result_rot.get('forward_barrier_meV', 'N/A'):.2f} meV")
        print(f"  Converged: {result_rot.get('converged', False)}")
        print(f"  Saddle file: {result_rot.get('saddle_file', 'N/A')}")
    else:
        print(f"\nRotation barrier: NOT CALCULATED")
    
    # Verify consistency
    if result_trans and result_rot:
        E_init_trans = result_trans.get('initial_energy')
        E_init_rot = result_rot.get('initial_energy')
        
        if E_init_trans is not None and E_init_rot is not None:
            diff = abs(E_init_trans - E_init_rot) * 1000
            if diff < 0.1:
                print(f"\nConsistency check: PASSED (same initial energy)")
            else:
                print(f"\nConsistency check: FAILED!")
                print(f"  Translation initial E: {E_init_trans:.6f} eV")
                print(f"  Rotation initial E: {E_init_rot:.6f} eV")
                print(f"  Difference: {diff:.2f} meV")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70 + "\n")
    
    return result_trans, result_rot


if __name__ == '__main__':
    result_trans, result_rot = main()