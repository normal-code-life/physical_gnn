"""Export original and predicted displacements for visual comparison in PyVista."""

import os

import numpy as np
import pyvista as pv

# Configure the case and output folder used by this manual visualization script.
folder_path = "check_data"
index = 3
os.makedirs(folder_path, exist_ok=True)
label_folders = "/home/lshi/Work/GNN_20250902/CT_original/"
mesh = pv.read(f"{label_folders}/ct_case_{index:04d}/05_mesh_ref/mesh-complete.mesh.vtu")
results_ori = np.genfromtxt(f"../../data/passive_biv/record_results/ct_case_{index:04d}.csv", delimiter=",")[:, :3]
results_pre = np.genfromtxt(f"output_{index:04d}_100.csv", delimiter=",")[1:]
# results_stress = np.genfromtxt(f'Ginsburg/Result_Records/ct_case_{index_str}.csv', delimiter=',')[:, 3:]
case_paths = f"{folder_path}/case_{index:04d}"
os.makedirs(case_paths, exist_ok=True)
mesh.point_data["displacement_ori"] = results_ori
mesh.save(f"{case_paths}/results_orig_{index:04d}.vtu")
mesh.point_data["displacement_pre"] = results_pre
mesh.save(f"{case_paths}/results_pred_{index:04d}.vtu")
print("Finished\n")
