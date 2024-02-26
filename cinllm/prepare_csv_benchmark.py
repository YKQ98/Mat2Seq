import io
import pandas as pd
import tarfile
import argparse
from pymatgen.io.cif import Structure
from mycif import CifWriter
import re
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


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


def process_cif_files(input_csv, output_tar_gz):
    df = pd.read_csv(input_csv)
    with tarfile.open(output_tar_gz, "w:gz") as tar:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="preparing CIF files..."):
            id = row["material_id"]
            struct = Structure.from_str(row["cif"], fmt="cif")
            struct = correct_zero(struct)
            cif_content = CifWriter(struct=struct, symprec=0.1).__str__()
            
            cif_file = tarfile.TarInfo(name=f"{id}.cif")
            cif_bytes = cif_content.encode("utf-8")
            cif_file.size = len(cif_bytes)
            tar.addfile(cif_file, io.BytesIO(cif_bytes))


"""
This script is meant to be used to prepare the CDVAE benchmark 
.csv files (https://github.com/txie-93/cdvae/tree/main/data and 
https://github.com/jiaor17/DiffCSP/tree/main/data) for the 
CrystaLLM pre-processing step.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare benchmark CIF files and save to a tar.gz file.")
    parser.add_argument("input_csv", help="Path to the .csv containing the benchmark CIF files.")
    parser.add_argument("output_tar_gz", help="Path to the output tar.gz file")
    args = parser.parse_args()

    process_cif_files(args.input_csv, args.output_tar_gz)

    print(f"prepared CIF files have been saved to {args.output_tar_gz}")
