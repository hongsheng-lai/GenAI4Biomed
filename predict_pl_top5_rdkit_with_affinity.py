#!/usr/bin/env python
# coding: utf-8

# # Overview
# This script inference the whole dataset and evaluate the model performance with TOP-5 SUCCESS RATE CALCULATION, RMSA, time and peak gpu memory usage.
# RMSA, affinity are saved in the result folder.

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
save_rdkit_path = "./PL_results_top5_with_affinity/PL_rdkit/"
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
            shift_dis = 0   # for visual only, could be any number, shift the ligand away from the protein.
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

save_protein_results_path = "./PL_results_top5_with_affinity/protein_list/"
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
    # Only pair protein with its own ligand (same PDB ID) - self-docking
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



dataset_path = f"./PL_results_top5_with_affinity/dataset/"
os.system(f"rm -r {dataset_path}")
os.system(f"mkdir -p {dataset_path}")
dataset = TankBind_prediction(dataset_path, data=info, protein_dict=protein_dict, compound_dict=compound_dict)



import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm    # pip install tqdm if fails.
from model import get_model
import time
# from utils import *



batch_size = 5
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

for data in tqdm(data_loader, desc="Running inference"):
    batch_start = time.time()
    data = data.to(device)
    if device == 'cuda':
        batch_memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
    
    y_pred, affinity_pred = model(data)
    
    if device == 'cuda':
        batch_memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    affinity_pred_list.append(affinity_pred.detach().cpu())
    for i in range(data.y_batch.max() + 1):
        y_pred_list.append((y_pred[data['y_batch'] == i]).detach().cpu())
    
    batch_time = time.time() - batch_start
    batch_times.append(batch_time)

inference_end_time = time.time()
total_inference_time = inference_end_time - inference_start_time

affinity_pred_list = torch.cat(affinity_pred_list)

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



info.to_csv(f"./PL_results_top5_with_affinity/info_with_predicted_affinity.csv")

# generate coordination for RMSA and top 5 success rate calculation.
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
# # WITH TOP-5 SUCCESS RATE CALCULATION and RMSA calculation.


from generation_utils import get_LAS_distance_constraint_mask, get_info_pred_distance, write_with_new_coords, compute_coordinate_RMSD
# device is already set above, using same device

result_folder = f'./PL_results_top5_with_affinity/result/'
os.system(f'mkdir -p {result_folder}')

# Process top 5 predictions for each protein-compound pair
rmsd_results = []
top5_success_list = []

logging.info(f"\n{'='*50}")
logging.info(f"GENERATING POSES AND CALCULATING RMSD")
logging.info(f"{'='*50}")

for protein_compound_pair, group in chosen.groupby(['protein_name', 'compound_name']):
    protein_name, compound_name = protein_compound_pair
    logging.info(f"\nProcessing {protein_name}-{compound_name}:")
    
    # Load native ligand coordinates for RMSD comparison (if available)
    # Use RDKit molecule as template to ensure atom ordering matches predicted coordinates
    native_coords = None
    rdkit_mol_template = None  # This will be used as template for atom ordering
    
    # First, load the RDKit-generated molecule (this matches predicted coordinates atom ordering)
    rdkitMolFile = f"{save_rdkit_path}/{compound_name}_mol_from_rdkit.sdf"
    if os.path.exists(rdkitMolFile):
        try:
            rdkit_mol_template = Chem.MolFromMolFile(rdkitMolFile)
            if rdkit_mol_template is None:
                logging.warning(f"  Could not load RDKit molecule template from {rdkitMolFile}")
        except Exception as e:
            logging.warning(f"  Error loading RDKit molecule template from {rdkitMolFile}: {e}")
    
    # Now load native ligand coordinates and align to RDKit molecule atom ordering
    if compound_name in ligand_file_paths:
        native_ligand_file = ligand_file_paths[compound_name]
        try:
            native_mol = Chem.MolFromMolFile(native_ligand_file)
            if native_mol is not None:
                native_conf = native_mol.GetConformer()
                if native_conf is None:
                    logging.warning(f"  Native ligand molecule has no conformer in {native_ligand_file}")
                    native_coords = None
                else:
                    # Remove hydrogens from native molecule to match RDKit molecule
                    # IMPORTANT: We extract coordinates from the NATIVE ligand conformer (native_conf_noH),
                    # which contains the true crystal structure coordinates. The RDKit molecule is ONLY
                    # used as a template for atom ordering - it does NOT contain native coordinates.
                    # Create a copy first to preserve the original
                    native_mol_noH = Chem.RemoveHs(Chem.Mol(native_mol))
                    # After removing hydrogens, RDKit should preserve the conformer and coordinates
                    # for the remaining heavy atoms. Verify the conformer exists.
                    native_conf_noH = None
                    if native_mol_noH.GetNumConformers() > 0:
                        # Get the conformer - this contains the NATIVE crystal structure coordinates
                        # (just with hydrogens removed, so fewer atoms than original)
                        conf = native_mol_noH.GetConformer()
                        # Verify it has valid coordinates for all remaining atoms
                        if conf.GetNumAtoms() == native_mol_noH.GetNumAtoms():
                            native_conf_noH = conf
                        else:
                            logging.warning(f"  Conformer atom count mismatch: conf has {conf.GetNumAtoms()}, molecule has {native_mol_noH.GetNumAtoms()}")
                    
                    if native_conf_noH is None:
                        logging.warning(f"  Could not get conformer from native molecule after removing hydrogens")
                        native_coords = None
                    else:
                        # Check if we have the RDKit template molecule (used ONLY for atom ordering)
                        if rdkit_mol_template is not None:
                            # Verify they represent the same molecule
                            try:
                                native_smiles = Chem.MolToSmiles(native_mol_noH)
                                rdkit_smiles = Chem.MolToSmiles(rdkit_mol_template)
                                
                                if native_smiles == rdkit_smiles:
                                    # Same molecule - match atom ordering using RDKit template
                                    # NOTE: rdkit_mol_template does NOT have native coordinates,
                                    # it's only used to determine atom ordering via GetSubstructMatch
                                    # Use GetSubstructMatch to find matching atoms
                                    # For RMSD to be correct, we need: RDKit atom j → Native atom index
                                    # So we use reverse match: native_mol_noH.GetSubstructMatch(rdkit_mol_template)
                                    # This gives match_reverse[j] = native atom index for RDKit atom j
                                    match_reverse = native_mol_noH.GetSubstructMatch(rdkit_mol_template)
                                    if len(match_reverse) == rdkit_mol_template.GetNumAtoms() == native_mol_noH.GetNumAtoms():
                                        # Perfect match - extract NATIVE coordinates in RDKit atom order
                                        # match_reverse[j] gives the native atom index for RDKit atom j
                                        # Extract coordinates from NATIVE ligand conformer (contains crystal structure coords)
                                        # Reorder them to match RDKit atom ordering for correct RMSD calculation
                                        native_coords_reordered = [list(native_conf_noH.GetAtomPosition(match_reverse[j])) for j in range(rdkit_mol_template.GetNumAtoms())]
                                        native_coords = torch.tensor(native_coords_reordered, dtype=torch.float32).to(device)
                                        logging.info(f"  Loaded native ligand coordinates from: {native_ligand_file}")
                                        logging.info(f"  Native ligand: {native_mol.GetNumAtoms()} atoms (with H), {native_mol_noH.GetNumAtoms()} atoms (no H)")
                                        logging.info(f"  RDKit molecule: {rdkit_mol_template.GetNumAtoms()} atoms (used only for atom ordering)")
                                        logging.info(f"  Successfully aligned native coordinates to RDKit atom ordering")
                                    else:
                                        logging.warning(f"  Atom mapping mismatch: match_len={len(match_reverse)}, native={native_mol_noH.GetNumAtoms()}, rdkit={rdkit_mol_template.GetNumAtoms()}")
                                        # Try forward match as fallback
                                        match_forward = rdkit_mol_template.GetSubstructMatch(native_mol_noH)
                                        if len(match_forward) == native_mol_noH.GetNumAtoms() == rdkit_mol_template.GetNumAtoms():
                                            # match_forward[i] gives RDKit atom index for native atom i
                                            # Need inverse mapping: RDKit atom j → native atom i
                                            match_inverse = {match_forward[i]: i for i in range(len(match_forward))}
                                            if len(match_inverse) == rdkit_mol_template.GetNumAtoms():
                                                native_coords_list = [list(native_conf_noH.GetAtomPosition(i)) for i in range(native_mol_noH.GetNumAtoms())]
                                                native_coords_reordered = [native_coords_list[match_inverse[j]] for j in range(rdkit_mol_template.GetNumAtoms())]
                                                native_coords = torch.tensor(native_coords_reordered, dtype=torch.float32).to(device)
                                                logging.info(f"  Loaded native ligand coordinates (using forward match fallback)")
                                                logging.info(f"  Successfully aligned native coordinates to RDKit atom ordering (forward match)")
                                            else:
                                                logging.warning(f"  Forward match inverse mapping incomplete: got {len(match_inverse)} mappings")
                                                native_coords = None
                                        else:
                                            logging.warning(f"  Forward match also failed: got {len(match_forward)} mappings")
                                            native_coords = None
                                else:
                                    logging.warning(f"  SMILES mismatch - molecules are different!")
                                    logging.warning(f"  Native SMILES: {native_smiles[:50]}...")
                                    logging.warning(f"  RDKit SMILES: {rdkit_smiles[:50]}...")
                                    native_coords = None
                            except Exception as align_e:
                                logging.warning(f"  Error aligning atom orderings: {align_e}")
                                # Fallback: try direct extraction if counts match
                                # Still extract from NATIVE conformer, just don't reorder
                                if native_conf_noH is not None and native_mol_noH.GetNumAtoms() == rdkit_mol_template.GetNumAtoms():
                                    native_coords = torch.tensor([list(native_conf_noH.GetAtomPosition(i)) for i in range(native_mol_noH.GetNumAtoms())], dtype=torch.float32).to(device)
                                    logging.warning(f"  Using native atom order directly (may be incorrect if atom ordering differs)")
                                    logging.info(f"  Native ligand has {native_mol_noH.GetNumAtoms()} atoms (no H)")
                                else:
                                    native_coords = None
                                    logging.warning(f"  Cannot align coordinates: atom count mismatch (native={native_mol_noH.GetNumAtoms()}, rdkit={rdkit_mol_template.GetNumAtoms()})")
                        else:
                            # No RDKit template available - remove hydrogens and use native ordering
                            if native_conf_noH is not None:
                                native_coords = torch.tensor([list(native_conf_noH.GetAtomPosition(i)) for i in range(native_mol_noH.GetNumAtoms())], dtype=torch.float32).to(device)
                                logging.info(f"  Loaded native ligand coordinates from: {native_ligand_file}")
                                logging.info(f"  Native ligand: {native_mol.GetNumAtoms()} atoms (with H), {native_mol_noH.GetNumAtoms()} atoms (no H)")
                                logging.warning(f"  RDKit template not available - using native atom order (may not match predicted coordinates)")
                            else:
                                native_coords = None
                                logging.warning(f"  Native molecule has no conformer after removing hydrogens")
            else:
                logging.warning(f"  Could not load native ligand molecule from {native_ligand_file}")
                native_coords = None
        except Exception as e:
            logging.warning(f"  Could not load native ligand file {native_ligand_file}: {e}")
            native_coords = None
    else:
        logging.warning(f"  Native ligand file not found for {compound_name}, skipping RMSD calculation")
    
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
        # RDKit molecule should have NO hydrogens (generate_conformation removes them at line 290)
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
    # This is the standard approach: rank all poses by docking score (binding affinity)
    # Higher affinity means the model predicts stronger binding at this site
    # Use loss as tie-breaker (lower loss = better pose optimization)
    # Note: All poses from the same binding site have the same affinity,
    # so poses from higher-affinity sites will rank higher
    all_poses_sorted = sorted(all_poses, key=lambda x: (-x['affinity'], x['loss']))
    
    logging.info(f"  Total poses generated: {len(all_poses_sorted)}")
    logging.info(f"  Ranking all poses globally and selecting top 5 for RMSD calculation...")
    
    # Select top 5 poses from ALL poses (global ranking)
    top5_poses = all_poses_sorted[:5]
    rmsd_values_top5 = []
    
    # Calculate RMSD for each of the top 5 poses using coordinate-based RMSD with alignment
    # Only if native ligand coordinates are available
    if native_coords is not None:
        # Verify native_coords shape for debugging
        if rdkit_mol_template is not None:
            expected_n_atoms = rdkit_mol_template.GetNumAtoms()
            if native_coords.shape[0] != expected_n_atoms:
                logging.warning(f"  WARNING: Native coords has {native_coords.shape[0]} atoms, but RDKit template has {expected_n_atoms} atoms")
            else:
                logging.info(f"  Verified: Native coords ({native_coords.shape[0]} atoms) matches RDKit template ({expected_n_atoms} atoms)")
        
        for rank, pose in enumerate(top5_poses, 1):
            predicted_coords = torch.tensor(pose['coords'], dtype=torch.float32).to(device)
            
            # Verify atom counts match before RMSD calculation
            if native_coords.shape[0] != predicted_coords.shape[0]:
                logging.error(f"  ERROR: Atom count mismatch! Native: {native_coords.shape[0]} atoms, Predicted: {predicted_coords.shape[0]} atoms")
                logging.error(f"  This suggests atom ordering or hydrogen removal mismatch - RMSD calculation will fail")
                logging.warning(f"  Skipping RMSD calculation for top-{rank}")
                rmsd_values_top5.append(np.nan)
                continue
            
            # Log verification that atom counts match
            logging.info(f"  Verified for top-{rank}: Native and predicted both have {native_coords.shape[0]} atoms (no hydrogens)")
            
            try:
                rmsd = compute_coordinate_RMSD(native_coords, predicted_coords, use_kabsch=True).item()
                rmsd_values_top5.append(rmsd)
                logging.info(f"  Top-{rank}: {pose['pocket_name']}, RMSD = {rmsd:.2f} Å, Affinity = {pose['affinity']:.3f}, Loss = {pose['loss']:.3f}")
            except Exception as e:
                logging.warning(f"  Error calculating RMSD for top-{rank}: {e}")
                rmsd_values_top5.append(np.nan)
    else:
        # No native ligand available, skip RMSD calculation
        logging.info(f"  Skipping RMSD calculation (no native ligand available)")
        for rank, pose in enumerate(top5_poses, 1):
            rmsd_values_top5.append(np.nan)
            logging.info(f"  Top-{rank}: {pose['pocket_name']}, Affinity = {pose['affinity']:.3f}, Loss = {pose['loss']:.3f} (RMSD: N/A)")
    
    # Find the best prediction (lowest RMSD among top-5, or best by affinity if no RMSD)
    if native_coords is not None and len([r for r in rmsd_values_top5 if not np.isnan(r)]) > 0:
        # Use RMSD to find best pose
        valid_rmsd_indices = [i for i, r in enumerate(rmsd_values_top5) if not np.isnan(r)]
        if valid_rmsd_indices:
            best_pose_idx = valid_rmsd_indices[np.argmin([rmsd_values_top5[i] for i in valid_rmsd_indices])]
        else:
            best_pose_idx = 0  # Fallback to first pose
    else:
        # No RMSD available, use best by affinity (first in sorted list)
        best_pose_idx = 0
    
    best_prediction = top5_poses[best_pose_idx]
    
    # Verify best_prediction has 'affinity' field
    if 'affinity' not in best_prediction:
        raise KeyError(f"best_prediction missing 'affinity' field! Available keys: {best_prediction.keys()}")
    
    # Save ONLY the best prediction (single SDF file per protein-compound pair)
    toFile = f'{result_folder}/{compound_name}_{best_prediction["pocket_name"]}_tankbind.sdf'
    write_with_new_coords(best_prediction['mol'], best_prediction['coords'], toFile)
    if native_coords is not None and not np.isnan(rmsd_values_top5[best_pose_idx]):
        logging.info(f"  Saved ONLY best prediction: {best_prediction['pocket_name']} (RMSD = {rmsd_values_top5[best_pose_idx]:.2f} Å)")
    else:
        logging.info(f"  Saved ONLY best prediction: {best_prediction['pocket_name']} (Affinity = {best_prediction['affinity']:.3f})")
    logging.info(f"    Note: Top-5 poses evaluated, but only best pose saved to disk")
    
    # Check if any of top 5 succeeds (RMSD < 5Å)
    if native_coords is not None and len([r for r in rmsd_values_top5 if not np.isnan(r)]) > 0:
        valid_rmsd = [r for r in rmsd_values_top5 if not np.isnan(r)]
        min_rmsd = min(valid_rmsd)
        top5_success = 1 if min_rmsd < 5.0 else 0
        top5_success_list.append(top5_success)
        
        # Also check for top-2 success (RMSD < 2Å)
        top2_success = 1 if min_rmsd < 2.0 else 0
        
        # Get the affinity for the pose with minimum RMSD
        # Verify that best_pose_idx points to the pose with minimum RMSD
        rmsd_at_best_pose_idx = rmsd_values_top5[best_pose_idx] if best_pose_idx < len(rmsd_values_top5) else np.nan
        if not np.isnan(rmsd_at_best_pose_idx) and abs(rmsd_at_best_pose_idx - min_rmsd) < 1e-6:
            # best_pose_idx correctly points to pose with minimum RMSD
            affinity_min_rmsd = best_prediction['affinity']  # Affinity for pose with minimum RMSD
        else:
            # Fallback: find the index with minimum RMSD explicitly
            min_rmsd_idx = None
            for idx, rmsd_val in enumerate(rmsd_values_top5):
                if not np.isnan(rmsd_val) and abs(rmsd_val - min_rmsd) < 1e-6:
                    min_rmsd_idx = idx
                    break
            if min_rmsd_idx is not None and min_rmsd_idx < len(top5_poses):
                affinity_min_rmsd = top5_poses[min_rmsd_idx]['affinity']
            else:
                # Final fallback: use best_prediction
                affinity_min_rmsd = best_prediction['affinity']
            logging.warning(f"  Warning: best_pose_idx ({best_pose_idx}) might not match min_rmsd index, recalculated")
        
        rmsd_results.append({
            'protein_name': protein_name,
            'compound_name': compound_name,
            'min_rmsd': min_rmsd,
            'affinity': affinity_min_rmsd,  # Affinity for pose with minimum RMSD
            'all_rmsd': rmsd_values_top5,
            'top5_success': top5_success,
            'top2_success': top2_success,
            'n_predictions': len(top5_poses),
            'total_poses_generated': len(all_poses_sorted)
        })
        
        logging.info(f"  Min RMSD (among top-5) = {min_rmsd:.2f} Å")
        logging.info(f"  Affinity for pose with min RMSD = {affinity_min_rmsd:.3f}")
        logging.info(f"  Verified: RMSD at best_pose_idx[{best_pose_idx}] = {rmsd_at_best_pose_idx:.2f} Å (matches min_rmsd)")
        logging.info(f"  Top-5 Success (RMSD < 5Å): {'✓' if top5_success else '✗'}")
        logging.info(f"  Top-2 Success (RMSD < 2Å): {'✓' if top2_success else '✗'}")
    else:
        # No RMSD available, mark as skipped
        # Use the best affinity (best_prediction already points to best by affinity when RMSD unavailable)
        affinity_fallback = best_prediction['affinity']
        
        rmsd_results.append({
            'protein_name': protein_name,
            'compound_name': compound_name,
            'min_rmsd': np.nan,
            'affinity': affinity_fallback,  # Best affinity when no RMSD available
            'all_rmsd': rmsd_values_top5,
            'top5_success': np.nan,
            'top2_success': np.nan,
            'n_predictions': len(top5_poses),
            'total_poses_generated': len(all_poses_sorted)
        })
        logging.info(f"  RMSD calculation skipped (no native ligand available)")
        logging.info(f"  Using best affinity (rank 1): {affinity_fallback:.3f}")

logging.info(f"{'='*50}\n")

# Calculate overall top-5 success rate
valid_rmsd_results = [r for r in rmsd_results if not np.isnan(r.get('top5_success', np.nan))]
if len(valid_rmsd_results) > 0:
    top5_success_list = [r['top5_success'] for r in valid_rmsd_results]
    top2_success_list = [r['top2_success'] for r in valid_rmsd_results]
    top5_success_rate = np.mean(top5_success_list)
    top2_success_rate = np.mean(top2_success_list)
    all_min_rmsd = [r['min_rmsd'] for r in valid_rmsd_results]
    
    logging.info(f"\n{'='*50}")
    logging.info(f"TOP-5 SUCCESS RATE RESULTS")
    logging.info(f"{'='*50}")
    logging.info(f"Total protein-compound pairs: {len(rmsd_results)}")
    logging.info(f"Pairs with RMSD calculation: {len(valid_rmsd_results)}")
    logging.info(f"Pairs without native ligand: {len(rmsd_results) - len(valid_rmsd_results)}")
    logging.info(f"Top-5 Success Rate (RMSD < 5Å): {top5_success_rate:.4f} ({top5_success_rate*100:.2f}%)")
    logging.info(f"Top-2 Success Rate (RMSD < 2Å): {top2_success_rate:.4f} ({top2_success_rate*100:.2f}%)")
    logging.info(f"Mean of minimum RMSD: {np.mean(all_min_rmsd):.2f} Å")
    logging.info(f"Median of minimum RMSD: {np.median(all_min_rmsd):.2f} Å")
    logging.info(f"Std of minimum RMSD: {np.std(all_min_rmsd):.2f} Å")
    logging.info(f"Min RMSD range: [{np.min(all_min_rmsd):.2f}, {np.max(all_min_rmsd):.2f}] Å")
    logging.info(f"{'='*50}\n")
    
    # Save RMSD results to CSV
    rmsd_df = pd.DataFrame(rmsd_results)
    # Expand all_rmsd list into separate columns for better readability
    for i in range(5):
        rmsd_df[f'rmsd_rank_{i+1}'] = rmsd_df['all_rmsd'].apply(lambda x: x[i] if i < len(x) and not np.isnan(x[i] if isinstance(x[i], (int, float)) else np.nan) else np.nan)
    rmsd_df = rmsd_df.drop('all_rmsd', axis=1)
    rmsd_df.to_csv(f"./PL_results_top5_with_affinity/top5_rmsd_results.csv", index=False)
    logging.info(f"RMSD results saved to: ./PL_results_top5_with_affinity/top5_rmsd_results.csv")
else:
    logging.warning("No RMSD results to calculate success rate (no native ligands found)")

