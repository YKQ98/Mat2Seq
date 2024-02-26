# verify that crystallm is not periodic invariant
from tqdm import tqdm
import pickle as pk
from pymatgen.io.cif import Structure
from mycif import CifWriter
import re
import numpy as np

with open("mp20_orig_train.pkl", 'rb') as f:
    train_data = pk.load(f)

error_cnt = 0

def correct_zero(structure):
    # find the original point of a unit cell, deterministically

    # Step 1: Find the indexes with the smallest atomic number
    min_atomic_number = min([site.specie.Z for site in structure])
    min_atomic_number_indexes = [i for i, site in enumerate(structure) if site.specie.Z == min_atomic_number]

    # Step 2 & 3: Calculate the neighborhood and density for these indexes using radius 1 Å, 2 Å, and 3 Å
    density_results = []
    for index in min_atomic_number_indexes:
        densities = {}
        for radius in [1, 2, 3]:  # Radii in Angstroms
            neighbors = structure.get_neighbors(structure[index], r=radius)
            density = sum([neighbor.specie.Z for neighbor in neighbors])
            densities[radius] = density
        density_results.append((index, densities))

    # Step 4: Sort these indexes by density, in decreasing order
    sorted_by_density = sorted(density_results, key=lambda x: (x[1][1], x[1][2], x[1][3]), reverse=True)

    assert sorted_by_density
    zero_idx = sorted_by_density[0][0]
    zero_pos = structure.frac_coords[zero_idx]
    pos = (structure.frac_coords - np.array(zero_pos)) % 1.
    # correct the structure by periodic shift
    structure = Structure(species=structure.species, coords=pos, lattice = structure.lattice)
    return structure

pattern_general = r"\b(\w+)\s+\1\d+\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"

# for i in tqdm(range(len(train_data))):
for i in tqdm(range(len(train_data))):
    try:
        tmp_data = train_data[i][1]
        structure = Structure.from_str(tmp_data, fmt="cif")
        pos = (structure.frac_coords - np.array([0.1, 0.2, 0.3])) % 1.
        new_struc = Structure(species=structure.species, coords=pos, lattice = structure.lattice)
        structure = correct_zero(structure)
        new_struc = correct_zero(new_struc)
        cif_old = CifWriter(struct=structure, symprec=0.1).__str__()
        cif_new = CifWriter(struct=new_struc, symprec=0.1).__str__()
        match_old = re.search(pattern_general, cif_old)
        extracted_values_old = np.array(match_old.groups()[1:],dtype=float)
        match_new = re.search(pattern_general, cif_new)
        extracted_values_new = np.array(match_new.groups()[1:],dtype=float)
        if abs(extracted_values_old - extracted_values_new).sum() > 1e-4:
            error_cnt += 1
            print(extracted_values_old, extracted_values_new)
            print(cif_old, cif_new)
            cif_old = CifWriter(struct=structure, symprec=0.1, print=True).__str__()
            cif_new = CifWriter(struct=new_struc, symprec=0.1, print=True).__str__()
    except:
        continue

print("total error cnt", error_cnt, "total number of samples in train", len(train_data))

