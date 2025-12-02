# Workflow for Adsorbate–Slab Energy Calculations, PES, Screening, and NEB

This document documents the step-by-step workflow implemented across `model/neb.py`, `Calculations/*/solver.py` (example: `Calculations/CH3/solver.py`), and the interactive notebook `Calculations/ipynb_and_scripts/NEB.ipynb`. It describes goals, inputs, outputs, exact steps taken by the code, verification checks, and practical notes.

## High-level overview

The codebase implements the following major components (each described in detail below):

1. Calculate energy of an adsorbate on a slab at a given lateral position.
2. Compute a 2D Potential Energy Surface (PES) across lateral coordinates.
3. Detect adsorption sites and generate symmetry-unique placements for screening.
4. Create full slab+adsorbate structures with Hookean constraints for robust relaxations.
5. Perform site screening (heights × rotations) and save results (pickles/JSON/summaries).
6. Validate, clean, and recover screening data files.
7. Select NEB endpoints (translation and rotation) and prepare/run NEB calculations.
8. Use the notebook `NEB.ipynb` for interactive visualization, inspection and plotting.

---

## 1) Energy calculation of adsorbate on slab (single placement)

Goal
- Compute either the total energy of the combined system (slab + adsorbate) or the adsorption energy referenced to isolated slab + gas-phase molecule.

Inputs / Outputs / Contract
- Inputs: `slab` (ASE Atoms), `adsorbate` (ASE Atoms), `position` (x,y), `energy_type` (`'total'` or `'adsorption'`).
- Outputs: float energy (eV). For `'adsorption'` the returned value is E_total - (E_slab + E_adsorbate).
- Success criteria: numeric return, original `slab` and `adsorbate` unchanged.

Where implemented
- `calculate_energy(slab, adsorbate, position, energy_type='total')` in `model/neb.py`.

Step-by-step process implemented in the code
1. Copy the slab to avoid side-effects: `test_slab = slab.copy()`.
2. Place adsorbate at the requested lateral position and a default height (e.g., 2.0 Å):
   - `add_adsorbate(test_slab, adsorbate, height=height, position=position[:2])`.
3. Assign the calculator to `test_slab` (`test_slab.calc = ase_calculator`) where `ase_calculator` is set at module scope (e.g., `mace_mp(model="medium", device='cpu')`).
4. Compute `E_total = test_slab.get_potential_energy()`.
5. If `energy_type == 'adsorption'` compute and subtract the isolated slab and adsorbate energies.
6. Return the computed energy.

Notes and edge-cases
- Ensure the calculator is correctly initialized. The code sets `ase_calculator` at the module top.
- Watch out for overlapping atoms (placement too close to surface); the higher-level routines either avoid such placements or apply constraints.

Quick verification
- Compute energies for clean `slab.get_potential_energy()` and `adsorbate.get_potential_energy()` separately and compare magnitudes.
- Run `calculate_energy` for one coordinate and sanity-check the adsorption energy (typical chemical ranges).

---

## 2) Potential Energy Surface (PES)

Goal
- Map energy vs. lateral position on the surface to identify minima and barriers.

Inputs / Outputs
- Inputs: `n_points` (grid resolution), prepared slab and optimized adsorbate (gas-phase). Optionally a fixed height for the adsorbate.
- Outputs: numpy arrays `X, Y, energies` and visualizations (2D contour + 3D surface).

Where implemented
- `PES(n_points=10)` in `model/neb.py`. The notebook `NEB.ipynb` can call it interactively.

Step-by-step implementation
1. Build or load slab: `slab = fcc111(metal, size=size, a=lattice_constant, vacuum=vacuum)` and set `slab.calc`.
2. Optimize the gas-phase adsorbate: `opt = BFGS(adsorbate); opt.run(fmax=0.05)` then read `adsorbate.get_potential_energy()`.
3. Define grid: get cell vectors and use `np.linspace(0, x_max, n_points)` / `np.linspace(0, y_max, n_points)` to define `X, Y` via `np.meshgrid`.
4. Loop over each grid cell and evaluate `calculate_energy(slab, adsorbate, position, energy_type='total')`. Store in `energies[i, j]`.
5. Post-process: shift energies by the minimum if desired for visualization (`energies_relative = energies - energies.min()`).
6. Plot using Matplotlib: `contourf` for 2D and `plot_surface` for 3D.

Performance considerations
- Grid size grows cost as O(n_points^2). Use small n for quick checks.
- For expensive calculators (DFT/large ML), parallelize evaluations or sample high-symmetry points only.
- Use caching if many repeated evaluations are expected.

Verification
- Visual check: expected minima near symmetry sites (top/bridge/hollow).
- Confirm arrays shapes and numeric types; test with `n_points=5` first.

---

## 3) Detect adsorption sites and generate unique placements

Goal
- Enumerate symmetry-inequivalent adsorption sites (single and pair sites) and produce bond parameter dictionaries used to place adsorbates consistently.

Where implemented
- `adsorption_sites_and_unique_placements(slab, surface_type='fcc111')` and `get_unique_sites(...)` in `model/neb.py`.
- Uses `acat.adsorption_sites.SlabAdsorptionSites` to enumerate candidate sites.

Step-by-step
1. Use ACAT `SlabAdsorptionSites` to get `all_sites`.
2. Call `get_unique_sites(all_sites, slab.cell, about=middle)` to filter symmetry duplicates.
3. In `generate_unique_placements` build pairwise site information and prepare single/double site bond parameter lists, where each bond parameter contains `site_pos`, `k` (spring constant), and `deq` (equilibrium bond distance).
4. Return lists: unique single sites, unique site pairs, and bond params for screening.

Notes
- Options include uniqueness by composition or subsurface if required.
- Pairs with too-close distances should be filtered out to avoid atomic clashes.

---

## 4) Create structure with Hookean constraints

Goal
- Place adsorbate onto the slab (COM or binding atom at site), apply rotation, then apply Hookean constraints to preserve internal bonds during relaxation.

Where implemented
- `create_structure(...)` and `AdsorbateConstraintManager` in `model/neb.py`.

Step-by-step
1. Copy slab and optimized adsorbate objects.
2. Determine the target placement position: `[site_x, site_y, site_z + height]`.
3. Translate the adsorbate so its COM or binding atom aligns with the target position.
4. If rotation requested, compute rotation about chosen center using `rotate_adsorbate_about_axis` (uses `scipy.spatial.transform.Rotation`).
5. Add adsorbate to slab with `add_adsorbate`.
6. Compute `adsorbate_indices` in the combined structure.
7. Use `AdsorbateConstraintManager` to detect bonds and create `Hookean` constraints with appropriate `k` and equilibrium distances.
8. Optionally fix slab bottom layers using `FixAtoms`.
9. Return the constrained `structure` ready for single-point energy or constrained relaxation.

Verification
- Confirm number of atoms and that constraints are present in the returned `Atoms.constraints` list.
- Check that bond lengths in the adsorbate are preserved by constraints.

---

## 5) Site screening (heights and rotations)

Goal
- For each unique adsorption site (and optionally site pairs), screen a set of heights and rotation angles; compute energies and save metadata for later NEB endpoint selection.

Where implemented
- `site_screening(...)` in `model/neb.py`.

Step-by-step
1. Ensure slab is optimized with `opt_slab()` and adsorbate optimized with `opt_molecule()`.
2. Get site lists via `adsorption_sites_and_unique_placements()`.
3. Define scanning ranges: for example `heights = np.arange(1.5, 3.5, 0.5)` and `rotations = np.arange(0, 360, 30)`.
4. For each site and each (height, rotation):
   - Build structure using `create_structure` with Hookean constraints.
   - Optionally perform a constrained relaxation or compute single-point energy via `calculate_energy`.
   - Save results (energies and metadata) into a dictionary or list.
5. Save screening output to `Screening_Data/screening_results_<timestamp>.pkl` and corresponding `screening_metadata_<timestamp>.json` and `screening_summary_<timestamp>.txt`.

Performance notes
- Screening is embarrassingly parallel across (site × height × rotation); parallelize where possible.
- For ML calculators use batched evaluations or GPU where available.

Verification
- After finishing, run `validate_screening_files()` to assert files are consistent.

---

## 6) Validate, clean, and recover screening files

Goal
- Ensure file integrity (pickle, JSON, summary files), identify corrupt or orphan files, recover missing metadata where possible, and clean incomplete artifacts.

Where implemented
- `validate_screening_files(output_dir='Screening_Data')`, `clean_incomplete_files(output_dir='Screening_Data', dry_run=True)`, and `recover_screening_files(output_dir='Screening_Data', timestamp=None)` in `model/neb.py`.

Step-by-step
1. List candidate files in `Screening_Data` using glob patterns for pickles, json, and summary files.
2. For each pickle: attempt to `pickle.load` and verify expected keys/structures.
3. For each JSON: verify parse and required keys (timestamp, summary fields).
4. For summary files: check they are present and consistent with pickle timestamps.
5. For missing JSON or summary files, run `recover_screening_files` to reconstruct them from valid pickle contents.
6. Use `clean_incomplete_files(dry_run=True)` to preview deletions; set `dry_run=False` to remove corrupted partial runs.

Verification
- Run `validate_screening_files` after recovery to ensure all necessary files now exist and are consistent.

---

## 7) Select NEB endpoints and prepare/run NEB calculations

Goal
- From screened results pick meaningful initial and final states (translation/rotation endpoints) and run a NEB calculation to find the minimum energy path and barrier.

Where implemented
- Endpoint selection: `select_neb_endpoints_translation(...)` and `select_neb_endpoints_rotation(...)` in `model/neb.py`.
- NEB setup & run: `prepare_neb_calculation(endpoint1, endpoint2, n_images=10, barrier_type='translation')`.
- Endpoint verification: `check_neb_endpoints(endpoint1, endpoint2, name="endpoints")` and internal `_verify_constraints`.

Step-by-step
1. Choose `site_best` (lowest-energy placements) from `best_site_results(screening_results)`.
2. For translation endpoints choose two placements that differ by lateral position; for rotation choose two placements differing primarily by rotation angle.
3. Reconstruct ASE `Atoms` endpoints using `create_structure` with the same bond_params used during screening (or load saved structure). Ensure both endpoints have:
   - identical number of atoms,
   - consistent ordering of atoms,
   - consistent constraints for adsorbate and fixed slab layers.
4. Build the NEB images list; set `image.calc = ase_calculator` for each.
5. Transfer constraints to each image (Hookean and fixed slab constraints).
6. Initialize `neb = NEB(images)` and call `neb.interpolate()` to create an initial band.
7. Optimize the band with an optimizer (BFGS / MDMin) until forces converge: `optimizer = BFGS(neb)`; `optimizer.run(fmax=target)`.
8. Postprocess energies along images: locate the maximum (saddle) and compute barrier heights.
9. Save trajectories and summary to `NEB_Results/neb_*.traj` and `NEB_Results/neb_summary.txt` via `save_neb_summary`.

Verification
- Plot energy vs image index and ensure smooth profile.
- Inspect the saddle geometry and force magnitudes.
- Validate that the NEB endpoints are preserved and final images are relaxed along normal modes (where applicable).

---

## 8) Notebook usage (`NEB.ipynb`)

Purpose
- Interactive exploratory runs, visualization (contour and 3D PES plots), trajectory inspection, and plotting NEB energy profiles.

Suggested notebook cells
1. Imports and quick config: import `model.neb as neb`, numpy, matplotlib.
2. Setup: `slab = neb.opt_slab(); ads = neb.opt_molecule(neb.init_molecule('CH3'))`.
3. Small PES: `neb.PES(n_points=7)`.
4. Screening: run `neb.site_screening(..., save_results=False)` for a small test set.
5. Load and inspect NEB trajectories saved in `NEB_Results/` with `ase.io.read` and visualize with `nglview` or `ase.visualize.view`.
6. Plot NEB energy profile with Matplotlib.

Notebook tips
- Use small `n_points` and small screening ranges for interactive testing.
- Use `%time` to measure expensive calls.

---

## 9) Quick "Try it" examples (project root)

Run the CH3 solver example (runs screening and two NEB computations as in `Calculations/CH3/solver.py`):

```bash
python Calculations/CH3/solver.py
```

Interactive notebook: open `Calculations/ipynb_and_scripts/NEB.ipynb` in Jupyter/VSCode and run the cells described above.

---

## 10) Edge cases and common failure modes

- Missing calculator initialization: ensure `ase_calculator` (e.g., MACE, EMT) is available and set at the top of `neb.py`.
- Overlapping atoms from bad heights/rotations: tune height or skip placements flagged as overlapping.
- Inconsistent atom ordering between endpoints: NEB requires identical atom indices; ensure endpoints are constructed in the same way.
- Corrupt intermediate files: use `validate_screening_files` and `recover_screening_files`.

---

## 11) Suggested small improvements (proactive)

- Add minimal unit tests for `calculate_energy` and `create_structure`.
- Add parallel execution (joblib) to `site_screening` for production runs.
- Add a `README.md` with short “how to run” examples and expected runtimes.

---

## 12) Completion & verification notes

- This workflow file mirrors the logic implemented in `model/neb.py` and the example runner `Calculations/CH3/solver.py` and is suitable for copy/paste into the project root.
- After long runs, run `python -c "from model import neb; print('neb import ok')"` as a quick smoke test that the module imports without syntax errors.


---

If you'd like, I can now:
- add a README with the short commands and environment notes,
- create simple unit tests and run them,
- or update the `NEB.ipynb` with ready-to-run cells. Which would you prefer next?
