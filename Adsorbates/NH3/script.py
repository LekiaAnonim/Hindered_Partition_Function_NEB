import sys
sys.path.insert(0, '/projects/westgroup/lekia.p/NEB/Adsorbates')
from model.neb import *
a = 4.012
# mol = init_molecule('NH3')
# opt_mol = opt_molecule(mol)
# slab = opt_slab()

# ads = opt_molecule(init_molecule('NH3'))
# screening_results = site_screening(slab, ads, center_xy='site', use_all_sites=True, workdir='/projects/westgroup/lekia.p/NEB/Adsorbates/NH3/Screening_Data')

# Validate all screening files
validation = validate_screening_files('/projects/westgroup/lekia.p/NEB/Adsorbates/NH3/Screening_Data')

clean_incomplete_files('/projects/westgroup/lekia.p/NEB/Adsorbates/NH3/Screening_Data', dry_run=True)

# Recover the missing JSON and summary files from your valid pickle file
recover_screening_files('/projects/westgroup/lekia.p/NEB/Adsorbates/NH3/Screening_Data')

screening_results = load_screening_results('/projects/westgroup/lekia.p/NEB/Adsorbates/NH3/Screening_Data/screening_results.pkl')

df_sorted, site_best = best_site_results(screening_results)

d_nn = a / np.sqrt(2)  # Surface nearest-neighbor distance ≈ 2.81 Å

endpoint1_trans, endpoint2_trans = select_neb_endpoints_translation(
        site_best, screening_results,
    )

images_trans, result_trans = prepare_neb_calculation(
        endpoint1_trans, endpoint2_trans,
        n_images=10,
        barrier_type='translation',
        workdir='/projects/westgroup/lekia.p/NEB/Adsorbates/NH3/NEB_Translation'
    )

print(result_trans)

endpoint1_rot, endpoint2_rot = select_neb_endpoints_rotation(
        site_best, screening_results, rotation_angle_diff=120
    )

images_rot, result_rot = prepare_neb_calculation(
        endpoint1_rot,
        endpoint2_rot,
        n_images=10,
        barrier_type='rotation',
        workdir='/projects/westgroup/lekia.p/NEB/Adsorbates/NH3/NEB_Rotation'
    )

print(result_rot)