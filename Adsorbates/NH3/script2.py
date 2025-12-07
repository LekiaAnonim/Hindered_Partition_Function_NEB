#!/usr/bin/env python3
"""
NEB calculation script for NH3 on Pt(111)
"""

import sys
import numpy as np

sys.path.insert(0, '/projects/westgroup/lekia.p/NEB/Adsorbates')

# Import from original neb.py (utilities we still need)
from model.neb import (
    validate_screening_files,
    clean_incomplete_files,
    recover_screening_files,
    load_screening_results,
    best_site_results,
)

# Import new/revised functions from neb2.py
from model.neb2 import (
    select_neb_endpoints_translation_v2,
    select_neb_endpoints_rotation_v2,
    prepare_neb_calculation,
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
# MAIN
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Step 1: Validate and load screening results
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 1: Loading Screening Results")
    print("="*70)
    
    validation = validate_screening_files(SCREENING_DIR)
    
    # Recover if needed
    recover_screening_files(SCREENING_DIR)
    
    # Load results
    screening_results = load_screening_results(f'{SCREENING_DIR}/screening_results.pkl')
    df_sorted, site_best = best_site_results(screening_results)
    
    # -------------------------------------------------------------------------
    # Step 2: Diagnose screening results
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 2: Diagnosing Screening Results")
    print("="*70)
    
    diagnosis = diagnose_screening_results(screening_results)
    
    if diagnosis['issues']:
        print("\n❌ Critical issues found:")
        for issue in diagnosis['issues']:
            print(f"   - {issue}")
        print("\nFix issues before proceeding!")
        return
    
    # -------------------------------------------------------------------------
    # Step 3: Translation barrier NEB
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 3: Translation Barrier Calculation")
    print("="*70)
    
    endpoint1_trans, endpoint2_trans = select_neb_endpoints_translation_v2(
        site_best, 
        screening_results,
        method='cross_site',  # Use different site types
    )
    
    if endpoint1_trans is None or endpoint2_trans is None:
        print("\n❌ Failed to select translation endpoints!")
        print("   Skipping translation NEB...")
        result_trans = None
    else:
        images_trans, result_trans = prepare_neb_calculation(
            endpoint1_trans, 
            endpoint2_trans,
            n_images=10,
            barrier_type='translation',
            workdir=f'{WORKDIR}/NEB_Translation',
            fmax=0.05,
        )
        
        if result_trans:
            print(f"\n✓ Translation barrier: {result_trans.get('forward_barrier_meV', 'N/A'):.2f} meV")
    
    # -------------------------------------------------------------------------
    # Step 4: Rotation barrier NEB
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 4: Rotation Barrier Calculation")
    print("="*70)
    
    # Get recommended rotation angle for NH3
    recommended_angle = get_recommended_rotation_angle(ADSORBATE)
    
    if recommended_angle is None:
        print(f"\n⚠️  Rotation may be meaningless for {ADSORBATE}")
        print("   Skipping rotation NEB...")
        result_rot = None
    else:
        endpoint1_rot, endpoint2_rot = select_neb_endpoints_rotation_v2(
            site_best, 
            screening_results, 
            rotation_angle_diff=recommended_angle,  # 60° for NH3
            adsorbate_formula=ADSORBATE,
        )
        
        if endpoint1_rot is None or endpoint2_rot is None:
            print("\n❌ Failed to select rotation endpoints!")
            print("   Skipping rotation NEB...")
            result_rot = None
        else:
            images_rot, result_rot = prepare_neb_calculation(
                endpoint1_rot,
                endpoint2_rot,
                n_images=10,
                barrier_type='rotation',
                workdir=f'{WORKDIR}/NEB_Rotation',
                fmax=0.05,
            )
            
            if result_rot:
                print(f"\n✓ Rotation barrier: {result_rot.get('forward_barrier_meV', 'N/A'):.2f} meV")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nAdsorbate: {ADSORBATE}")
    
    if result_trans:
        print(f"\nTranslation barrier:")
        print(f"  Forward barrier: {result_trans.get('forward_barrier_meV', 'N/A'):.2f} meV")
        print(f"  Converged: {result_trans.get('converged', False)}")
        print(f"  Saddle file: {result_trans.get('saddle_file', 'N/A')}")
    else:
        print(f"\nTranslation barrier: NOT CALCULATED")
    
    if result_rot:
        print(f"\nRotation barrier:")
        print(f"  Forward barrier: {result_rot.get('forward_barrier_meV', 'N/A'):.2f} meV")
        print(f"  Converged: {result_rot.get('converged', False)}")
        print(f"  Saddle file: {result_rot.get('saddle_file', 'N/A')}")
    else:
        print(f"\nRotation barrier: NOT CALCULATED")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70 + "\n")
    
    return result_trans, result_rot


if __name__ == '__main__':
    result_trans, result_rot = main()