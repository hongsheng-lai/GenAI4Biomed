# # Overview
# This script inference the whole dataset and save inference time and peak gpu memory for each samples. In addition, It will save all the top5 poses for each pair.

tankbind_src_folder_path = "../tankbind/"
import sys
sys.path.insert(0, tankbind_src_folder_path)

# # input preparation.

import os
import numpy as np
import pandas as pd
from feature_utils import split_protein_and_ligand
import rdkit.Chem as Chem
from feature_utils import generate_sdf_from_smiles_using_rdkit

# Process all ligand files in the dataset
data_path = "/ocean/projects/cis250160p/wli27/P-L"
save_rdkit_path = "./PL_results_save_sdf/PL_rdkit/"
os.system(f"mkdir -p {save_rdkit_path}")

N_REPEAT = 5  # Generate this many poses per binding site for ranking, ensure enough poses to rank and evaluate top 5 success rate.

# Store file paths for native ligand lookup later
ligand_file_paths = {}

# Search through all subdirectories for ligand files
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith("_ligand.sdf"):
            ligandFile = os.path.join(root, file)
            pdb_id = file.replace("_ligand.sdf", "")
            ligand_file_paths[pdb_id] = ligandFile
            smiles = Chem.MolToSmiles(Chem.MolFromMolFile(ligandFile))
            rdkitMolFile = f"{save_rdkit_path}/{file.replace('_ligand.sdf', '_mol_from_rdkit.sdf')}"
            shift_dis = 0  
            generate_sdf_from_smiles_using_rdkit(smiles, rdkitMolFile, shift_dis=shift_dis)

# # get protein feature


from feature_utils import get_protein_feature

from Bio.PDB import PDBParser
protein_dict = {}
protein_file_paths = {}  # Store protein file paths for p2rank
# Search through all subdirectories for protein files
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith("_protein.pdb"):
            proteinFile = os.path.join(root, file)
            pdb = file.replace("_protein.pdb", "")
            protein_file_paths[pdb] = proteinFile
            parser = PDBParser(QUIET=True)
            s = parser.get_structure("x", proteinFile)
            res_list = list(s.get_residues())
            protein_dict[pdb] = get_protein_feature(res_list)


# # get compound feature

from feature_utils import extract_torchdrug_feature_from_mol

compound_dict = {}
for file in os.listdir(save_rdkit_path):
    if file.endswith("_mol_from_rdkit.sdf"):
        ligandFile = f"{save_rdkit_path}/{file}"
        pdb = file.replace("_mol_from_rdkit.sdf", "")
        mol = Chem.MolFromMolFile(ligandFile)
        compound_dict[pdb] = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)   


# # p2rank

save_protein_results_path = "./PL_results_save_sdf/protein_list/"
os.system(f"mkdir -p {save_protein_results_path}")
pdb_list = list(protein_dict.keys())
# Copy protein files to save_protein_results_path for p2rank
for pdb in pdb_list:
    if pdb in protein_file_paths:
        proteinFile = protein_file_paths[pdb]
        os.system(f"cp {proteinFile} {save_protein_results_path}/{pdb}_protein.pdb")

ds = f"{save_protein_results_path}/protein_list.ds"
with open(ds, "w") as out:
    for pdb in pdb_list:
        # Write path relative to the .ds file location (they're in the same directory)
        # Use ./filename.pdb format as p2rank expects paths relative to .ds file location
        out.write(f"./{pdb}_protein.pdb\n")


p2rank = "bash /ocean/projects/cis250160p/wli27/p2rank_2.3/prank"
cmd = f"{p2rank} predict {ds} -o {save_protein_results_path}/p2rank -threads 1"
os.system(cmd)



info = []
for pdb in pdb_list:
    # self-docking
    if pdb in compound_dict:
        compound_name = pdb  # Use the same PDB ID for the compound
        # use protein center as the block center.
        com = ",".join([str(a.round(3)) for a in protein_dict[pdb][0].mean(axis=0).numpy()])
        info.append([pdb, compound_name, "protein_center", com])
        
        p2rankFile = f"{save_protein_results_path}/p2rank/{pdb}_protein.pdb_predictions.csv"
        if os.path.exists(p2rankFile):
            pocket = pd.read_csv(p2rankFile)
            pocket.columns = pocket.columns.str.strip()
            pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values
            for ith_pocket, com in enumerate(pocket_coms):
                com = ",".join([str(a.round(3)) for a in com])
                info.append([pdb, compound_name, f"pocket_{ith_pocket+1}", com])
        else:
            print(f"Warning: p2rank file not found: {p2rankFile}")
    else:
        logging.warning(f"  Warning: No compound found for protein {pdb}, skipping...")
info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'pocket_name', 'pocket_com'])
info


# # construct dataset



import torch
torch.set_num_threads(1)



from data import TankBind_prediction



dataset_path = f"./PL_results_save_sdf/dataset/"
os.system(f"rm -r {dataset_path}")
os.system(f"mkdir -p {dataset_path}")
dataset = TankBind_prediction(dataset_path, data=info, protein_dict=protein_dict, compound_dict=compound_dict)



import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm   
from model import get_model
import time
# from utils import *



batch_size = 1 # for inference, we only need to process one sample at a time to get the peak gpu memory and inference time.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(level=logging.INFO)

# GPU Memory monitoring
if device == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    logging.info(f"Initial GPU memory: {initial_memory:.2f} GB")

model = get_model(0, logging, device)
# self-dock model
modelFile = "../saved_models/self_dock.pt"

model.load_state_dict(torch.load(modelFile, map_location=device))
_ = model.eval()

if device == 'cuda':
    model_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    logging.info(f"GPU memory after loading model: {model_memory:.2f} GB")

data_loader = DataLoader(dataset, batch_size=batch_size, follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=8)
affinity_pred_list = []
y_pred_list = []

# Timing inference
inference_start_time = time.time()
batch_times = []
inference_efficiency_list = []

for sample_idx, data in enumerate(tqdm(data_loader, desc="Running inference")):
    batch_start = time.time()
    data = data.to(device)
    if device == 'cuda':
        batch_memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
    
    y_pred, affinity_pred = model(data)
    
    peak_memory = 0
    if device == 'cuda':
        batch_memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    affinity_pred_list.append(affinity_pred.detach().cpu())
    for i in range(data.y_batch.max() + 1):
        y_pred_list.append((y_pred[data['y_batch'] == i]).detach().cpu())
    
    batch_time = time.time() - batch_start
    batch_times.append(batch_time)
    sample_info = dataset.data.iloc[sample_idx]
    inference_efficiency_list.append([sample_info['protein_name'], batch_time, peak_memory])


inference_end_time = time.time()
total_inference_time = inference_end_time - inference_start_time

affinity_pred_list = torch.cat(affinity_pred_list)
inference_efficiency_list = pd.DataFrame(inference_efficiency_list, columns=['protein_name', 'inference_time', 'peak_memory'])
inference_efficiency_list.to_csv(f"./PL_results_save_sdf/inference_efficiency.csv", index=False)

# Print statistics
num_samples = len(dataset)
num_batches = len(batch_times)
avg_time_per_batch = np.mean(batch_times)
avg_time_per_sample = total_inference_time / num_samples

logging.info(f"\n{'='*50}")
logging.info(f"INFERENCE STATISTICS")
logging.info(f"{'='*50}")
logging.info(f"Total samples processed: {num_samples}")
logging.info(f"Batch size: {batch_size}")
logging.info(f"Number of batches: {num_batches}")
logging.info(f"Total inference time: {total_inference_time:.2f} seconds ({total_inference_time/60:.2f} minutes)")
logging.info(f"Average time per batch: {avg_time_per_batch:.3f} seconds")
logging.info(f"Average time per sample: {avg_time_per_sample:.3f} seconds")
if device == 'cuda':
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    logging.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")
    logging.info(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
logging.info(f"{'='*50}\n")



info = dataset.data
info['affinity'] = affinity_pred_list



#info.to_csv(f"./PL_results_save_sdf/info_with_predicted_affinity.csv")

# Select top 5 predictions by affinity for each protein-compound pair
logging.info(f"\n{'='*50}")
logging.info(f"SELECTING TOP-5 PREDICTIONS")
logging.info(f"{'='*50}")

top5_list = []
for (protein_name, compound_name), group in info.groupby(['protein_name', 'compound_name']):
    # Select top 5 binding sites by affinity for processing
    # Note: This is just to determine which binding sites to generate poses for
    # The final top-5 selection will be from ALL poses generated, ranked globally
    top5 = group.nlargest(5, 'affinity').reset_index()
    top5_list.append(top5)
    logging.info(f"{protein_name}-{compound_name}: Selected top {len(top5)} binding sites for pose generation")
    logging.info(f"  (Will generate poses for these sites, then rank ALL poses globally)")
    for idx, row in top5.iterrows():
        logging.info(f"  {row['pocket_name']}: Affinity = {row['affinity']:.3f}")

chosen = pd.concat(top5_list, ignore_index=True)
logging.info(f"{'='*50}\n")


# # from predicted interaction distance map to sdf



from generation_utils import get_LAS_distance_constraint_mask, get_info_pred_distance, write_with_new_coords, compute_coordinate_RMSD


result_folder = f'./PL_results_save_sdf/result/'
os.system(f'mkdir -p {result_folder}')

# Process top 5 predictions for each protein-compound pair
rmsd_results = []
top5_success_list = []
top5_pose_affinities = []

logging.info(f"\n{'='*50}")
logging.info(f"GENERATING POSES AND CALCULATING RMSD")
logging.info(f"{'='*50}")

for protein_compound_pair, group in chosen.groupby(['protein_name', 'compound_name']):
    protein_name, compound_name = protein_compound_pair
    logging.info(f"\nProcessing {protein_name}-{compound_name}:")
    # Generate ALL poses from ALL binding sites
    all_poses = []  # Store all poses: (pocket_name, affinity, loss, coords, mol)
    
    # Generate multiple poses for each binding site
    for i, line in group.iterrows():
        idx = line['index']
        pocket_name = line['pocket_name']
        affinity = line['affinity']
        coords = dataset[idx].coords.to(device)
        protein_nodes_xyz = dataset[idx].node_xyz.to(device)
        n_compound = coords.shape[0]
        n_protein = protein_nodes_xyz.shape[0]
        y_pred = y_pred_list[idx].reshape(n_protein, n_compound).to(device)
        y = dataset[idx].dis_map.reshape(n_protein, n_compound).to(device)
        compound_pair_dis_constraint = torch.cdist(coords, coords)
        rdkitMolFile = f"{save_rdkit_path}/{compound_name}_mol_from_rdkit.sdf"
        mol = Chem.MolFromMolFile(rdkitMolFile)
        
        # Verify: Check if predicted coordinates have hydrogens or not
        # dataset[idx].coords comes from compound_dict which was created from RDKit molecule
        # RDKit molecule should have no hydrogens 
        n_atoms_in_coords = coords.shape[0]
        n_atoms_in_rdkit_mol = mol.GetNumAtoms() if mol is not None else None
        if mol is not None:
            if n_atoms_in_coords != n_atoms_in_rdkit_mol:
                logging.warning(f"  WARNING: Coordinate count ({n_atoms_in_coords}) != RDKit molecule atoms ({n_atoms_in_rdkit_mol})")
            else:
                logging.info(f"  Verified: Predicted coordinates have {n_atoms_in_coords} atoms (matches RDKit molecule)")
        
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool().to(device)
        
        # Generate N_REPEAT poses for this binding site
        logging.info(f"  Generating {N_REPEAT} poses for {pocket_name}...")
        info = get_info_pred_distance(coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint, 
                                      LAS_distance_constraint_mask=LAS_distance_constraint_mask,
                                      n_repeat=N_REPEAT, show_progress=False)
        
        # Store ALL poses from this binding site (for global ranking later)
        for _, pose_info in info.iterrows():
            pose_coords = pose_info['coords'].astype(np.double)
            pose_loss = pose_info['loss']
            all_poses.append({
                'pocket_name': pocket_name,
                'affinity': affinity,
                'loss': pose_loss,
                'coords': pose_coords,
                'mol': mol
            })
        
        logging.info(f"  {pocket_name}: Generated {len(info)} poses, affinity = {affinity:.3f}")
    
    # Rank ALL poses globally by affinity (higher affinity = better binding prediction)
    # Higher affinity means the model predicts stronger binding at this site
    # lower loss = better pose optimization
    # Note: All poses from the same binding site have the same affinity,
    # so poses from higher-affinity sites will rank higher
    all_poses_sorted = sorted(all_poses, key=lambda x: (-x['affinity'], x['loss']))
    
    logging.info(f"  Total poses generated: {len(all_poses_sorted)}")
    logging.info(f"  Ranking all poses globally and selecting top 5 for RMSD calculation...")
    
    # Select top 5 poses from ALL poses (global ranking)
    top5_poses = all_poses_sorted[:5]
    rmsd_values_top5 = []
    
    
    # Save top 5 poses prediction (5 SDF file per protein-compound pair)
    for i, pose in enumerate(top5_poses):
        toFile = f'{result_folder}/{compound_name}_{i+1}_tankbind.sdf'
        write_with_new_coords(pose['mol'], pose['coords'], toFile)
        top5_pose_affinities.append({
            'protein_name': protein_name,
            'compound_name': compound_name,
            'pose_rank': i + 1,
            'pocket_name': pose['pocket_name'],
            'affinity': pose['affinity'],
            'loss': pose['loss']
        })
    

# Save CSV for affinity of saved top-5 poses
top5_affinity_df = pd.DataFrame(top5_pose_affinities)
top5_affinity_df.to_csv(f"./PL_results_save_sdf/top5_pose_affinities.csv", index=False)
logging.info(f"{'='*50}\n")

