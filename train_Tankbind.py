# # Overview
# Training evaluation script for the whole dataset
# works on all protein-ligand pairs with one epoch training

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
save_rdkit_path = "./PL_train_full/PL_rdkit/"
os.system(f"mkdir -p {save_rdkit_path}")

# Store file paths for lookup later
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

save_protein_results_path = "./PL_train_full/protein_list/"
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
    if pdb in compound_dict:
        compound_name = pdb        # use protein center as the block center.
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
info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'pocket_name', 'pocket_com'])
info


# # construct dataset

import torch
torch.set_num_threads(1)


import torch
from tqdm import tqdm
from new_dataset import TankBindDataSet_custom  # training dataset


dataset_path = f"./PL_train_full/dataset/"
os.system(f"rm -r {dataset_path}")
os.system(f"mkdir -p {dataset_path}")
dataset = TankBindDataSet_custom(dataset_path, data=info, protein_dict=protein_dict, compound_dict=compound_dict)



import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm   
from model import get_model
import time
# from utils import *


batch_size = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(level=logging.INFO)

# GPU Memory monitoring
if device == 'cuda':
    torch.cuda.empty_cache()  # Clear cache before starting
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    logging.info(f"Initial GPU memory: {initial_memory:.2f} GB")

model = get_model(0, logging, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if device == 'cuda':
    model_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    logging.info(f"GPU memory after loading model: {model_memory:.2f} GB")

data_loader = DataLoader(dataset, batch_size=batch_size, follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=8)
affinity_pred_list = []
y_pred_list = []

# Timing training
training_start_time = time.time()
batch_times = []
batch_memories = []  # Track memory per batch


criterion = torch.nn.MSELoss()  

timestamp = "train_run"

num_samples = len(dataset)
num_batches = max(len(data_loader), 1)

for epoch in range(1):
    model.train()


    for data in tqdm(data_loader, desc="Training"):
        batch_start = time.time()
        
        # Track memory before batch
        if device == 'cuda':
            batch_memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
        
        data = data.to(device)
        optimizer.zero_grad()
        y_pred, _ = model(data)
        dis_map = data.dis_map

        contact_loss = criterion(y_pred, dis_map)

        # print(contact_loss.item(), affinity_loss.item())
        loss = contact_loss 
        loss.backward()
        optimizer.step()
        
        # Track memory after batch
        if device == 'cuda':
            batch_memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
            batch_peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            batch_memories.append({
                'before': batch_memory_before,
                'after': batch_memory_after,
                'peak': batch_peak_memory
            })
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

    model.eval()




logging.info(f"\n{'='*50}")
logging.info(f"TRAINING STATISTICS")
logging.info(f"{'='*50}")
logging.info(f"Total samples processed: {num_samples}")
logging.info(f"Batch size: {batch_size}")
logging.info(f"Number of batches: {num_batches}")
total_training_time = time.time() - training_start_time
if len(batch_times) > 0:
    avg_time_per_batch = np.mean(batch_times)
else:
    avg_time_per_batch = total_training_time / num_batches
avg_time_per_sample = total_training_time / num_samples
logging.info(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
logging.info(f"Average time per batch: {avg_time_per_batch:.3f} seconds")
logging.info(f"Average time per sample: {avg_time_per_sample:.3f} seconds")
if len(batch_times) > 0:
    logging.info(f"Std time per batch: {np.std(batch_times):.3f} seconds")
    logging.info(f"Min batch time: {np.min(batch_times):.3f} seconds")
    logging.info(f"Max batch time: {np.max(batch_times):.3f} seconds")

if device == 'cuda':
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved_memory = torch.cuda.memory_reserved() / 1024**3  # GB
    
    logging.info(f"\nGPU Memory Statistics:")
    logging.info(f"  Initial GPU memory: {initial_memory:.2f} GB")
    logging.info(f"  Model memory: {model_memory:.2f} GB")
    logging.info(f"  Peak GPU memory usage: {peak_memory:.2f} GB")
    logging.info(f"  Current GPU memory usage: {current_memory:.2f} GB")
    logging.info(f"  Reserved GPU memory: {reserved_memory:.2f} GB")
    logging.info(f"  Memory increase (peak - initial): {peak_memory - initial_memory:.2f} GB")
    
    if len(batch_memories) > 0:
        avg_batch_memory = np.mean([m['after'] for m in batch_memories])
        max_batch_memory = np.max([m['peak'] for m in batch_memories])
        logging.info(f"  Average batch memory: {avg_batch_memory:.2f} GB")
        logging.info(f"  Max batch peak memory: {max_batch_memory:.2f} GB")

logging.info(f"{'='*50}\n")


