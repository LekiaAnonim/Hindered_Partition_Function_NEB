from neb import *

# mol = init_molecule('CH4')
# opt_mol = opt_molecule(mol)
# slab = opt_slab()

# ads = opt_molecule(init_molecule('CH4'))
# screening_results = site_screening(slab, ads, center_xy='binding', use_all_sites=True)

# Validate all screening files
# validation = validate_screening_files('Screening_Data')

# clean_incomplete_files('Screening_Data', dry_run=True)

# Recover the missing JSON and summary files from your valid pickle file
# recover_screening_files('Screening_Data')

screening_results = load_screening_results('/projects/westgroup/lekia.p/NEB/Screening_Data/screening_results_20251112_123452.pkl')

df_sorted, site_best = best_site_results(screening_results)

# endpoint1_trans, endpoint2_trans = select_neb_endpoints_translation(
#         site_best, screening_results
#     )

# images_trans, result_trans = prepare_neb_calculation(
#         endpoint1_trans, endpoint2_trans,
#         n_images=10,
#         barrier_type='translation'
#     )

# print(result_trans)

endpoint1_rot, endpoint2_rot = select_neb_endpoints_rotation(
        site_best, screening_results, rotation_angle_diff=120
    )

images_rot, result_rot = prepare_neb_calculation(
        endpoint1_rot,
        endpoint2_rot,
        n_images=10,
        barrier_type='rotation'
    )

print(result_rot)