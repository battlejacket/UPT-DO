# conda create --name open3d python=3.9
# pip install open3d
# pip install meshio
# pip install torch
# pip install tempfile
import os
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import meshio
import numpy as np
# import open3d as o3d
import torch
from tqdm import tqdm

from csv_rw import csv_to_dict

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="e.g. /data/shapenet_car/training_data")
    parser.add_argument("--dst", type=str, required=True, help="e.g. /data/shapenet_car/preprocessed")
    parser.add_argument("--saveNorm", type=bool, required=False, help="Set to False to avoid overwriting the normalization parameters")
    return vars(parser.parse_args())


def sdf(mesh, resolution):
    None
    # quads = mesh.cells_dict["quad"]

    # idx = np.flatnonzero(quads[:, -1] == 0)
    # out0 = np.empty((quads.shape[0], 2, 3), dtype=quads.dtype)

    # out0[:, 0, 1:] = quads[:, 1:-1]
    # out0[:, 1, 1:] = quads[:, 2:]

    # out0[..., 0] = quads[:, 0, None]

    # out0.shape = (-1, 3)

    # mask = np.ones(out0.shape[0], dtype=bool)
    # mask[idx * 2 + 1] = 0
    # quad_to_tri = out0[mask]

    # cells = [("triangle", quad_to_tri)]

    # new_mesh = meshio.Mesh(mesh.points, cells)

    # with tempfile.NamedTemporaryFile(delete=True, suffix=".ply") as tf:
    #     new_mesh.write(tf, file_format="ply")
    #     open3d_mesh = o3d.io.read_triangle_mesh(tf.name)
    # open3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(open3d_mesh)
    # scene = o3d.t.geometry.RaycastingScene()
    # _ = scene.add_triangles(open3d_mesh)

    # domain_min = torch.tensor([-2.0, -1.0, -4.5])
    # domain_max = torch.tensor([2.0, 4.5, 6.0])
    # tx = np.linspace(domain_min[0], domain_max[0], resolution)
    # ty = np.linspace(domain_min[1], domain_max[1], resolution)
    # tz = np.linspace(domain_min[2], domain_max[2], resolution)
    # grid = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(np.float32)
    # return torch.from_numpy(scene.compute_signed_distance(grid).numpy()).float()


def main(src, dst, save_normalization_param=True):
    src = Path(src).expanduser()
    assert src.exists(), f"'{src.as_posix()}' doesnt exist"
    # assert src.name == "training_data"
    dst = Path(dst).expanduser()
    if not dst.exists():
        os.mkdir(dst)
    # assert not dst.exists(), f"'{dst.as_posix()}' exist"
    print(f"src: {src.as_posix()}")
    print(f"dst: {dst.as_posix()}")
    
    
    # find all uris for samples
    uris = []
    for root, dirs, files in os.walk(src):
        for file in files:
            filePath = Path(os.path.join(root, file))
            uris.append(filePath)
    uris.sort()
    print(f"found {len(uris)} samples")
        
    # define csv parameters
    csvInvarNames = ["Points:0", "Points:1"]
    dictInvarNames = ["x", "y"]
    csvOutvarNames = ["Velocity:0", "Velocity:1", "Pressure"]
    dictOutvarNames = ["u", "v", "p"]
    scales = {"p": (0,1), "u": (0,1), "v": (0,1), "x": (0,1), "y": (-0.5,1)} #(translation, scale)
    skiprows = 0
    csvVarNames = csvInvarNames + csvOutvarNames
    dictVarNames = dictInvarNames + dictOutvarNames

    mapping = {}
    for csvVarName, dictVarName in zip(csvVarNames, dictVarNames):
        mapping[csvVarName] = dictVarName
    
    # Initialize variables for min/max coordinates and mean/std calculations
    min_coords = torch.tensor([float('inf'), float('inf')])
    max_coords = torch.tensor([-float('inf'), -float('inf')])
    sum_vars = torch.tensor([0.0, 0.0, 0.0])  # For u, v, p
    sum_sq_vars = torch.tensor([0.0, 0.0, 0.0])  # For u^2, v^2, p^2
    total_samples = 0

        
    for uri in tqdm(uris):
        reluri = uri.relative_to(src).with_suffix('')
        out = dst / reluri
        out.mkdir(exist_ok=True, parents=True)
        
        #read and process csv
        csvData = csv_to_dict(uri, mapping=mapping, delimiter=",", skiprows=skiprows)
        
        for key in dictVarNames:
            csvData[key] += scales[key][0]
            csvData[key] /= scales[key][1]
        NpMesh=np.concat([csvData["x"], csvData["y"]], axis=1)
        mesh_points = torch.from_numpy(NpMesh).float()
        min_coords = torch.min(min_coords, mesh_points.min(dim=0).values)
        max_coords = torch.max(max_coords, mesh_points.max(dim=0).values)
        torch.save(mesh_points, out / "mesh_points.th")        
        

        for i, outVar in enumerate(dictOutvarNames):  # u, v, p
            data = torch.tensor(csvData[outVar]).float()
            torch.save(data, out / f"{outVar}.th")
            sum_vars[i] += data.sum()
            sum_sq_vars[i] += (data ** 2).sum()
            total_samples += data.numel()

    if save_normalization_param:
        # Calculate mean and std
        mean_vars = sum_vars / total_samples
        std_vars = torch.sqrt((sum_sq_vars / total_samples) - (mean_vars ** 2))

        # Save normalization parameters
        torch.save({"min_coords": min_coords, "max_coords": max_coords}, dst / "coords_norm.th")
        torch.save({"mean": mean_vars, "std": std_vars}, dst / "vars_norm.th")

    print("fin")


if __name__ == "__main__":
    # main(**parse_args())
    main('./data/ffs/CSV/', './data/ffs/preprocessed/', True)
