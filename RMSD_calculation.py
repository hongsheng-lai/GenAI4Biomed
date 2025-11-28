import os
from rdkit import Chem
import pandas as pd
import numpy as np

def compute_coordinate_RMSD(predicted_coords, true_coords, use_kabsch=True):
    """
    Compute coordinate-based RMSD between predicted and true coordinates with proper alignment.
    This function aligns the structures using Kabsch algorithm before calculating RMSD.
    
    Args:
        predicted_coords: [n_atoms, 3] predicted coordinates (list or numpy array)
        true_coords: [n_atoms, 3] true/native coordinates (list or numpy array)
        use_kabsch: If True, align coordinates using Kabsch algorithm (recommended)
    
    Returns:
        rmsd: RMSD value in Angstroms (float)
    
    Note:
        This is the standard way to calculate RMSD for molecular docking.
        It removes translational and rotational differences before comparing structures.
        Uses numpy (CPU only, no GPU required).
    """
    # Convert to numpy arrays if needed
    pred = np.array(predicted_coords, dtype=np.float64)
    true = np.array(true_coords, dtype=np.float64)
    
    # Center coordinates (translate to origin)
    pred = pred - pred.mean(axis=0)
    true = true - true.mean(axis=0)
    
    if use_kabsch:
        # Kabsch algorithm for optimal rotation alignment
        # This finds the rotation that minimizes RMSD
        H = pred.T @ true
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        # Apply rotation to predicted coordinates
        pred_aligned = pred @ R.T
    else:
        pred_aligned = pred
    
    # Calculate RMSD: sqrt(mean(sum((aligned_pred - true)^2)))
    rmsd = np.sqrt(np.mean(np.sum((pred_aligned - true) ** 2, axis=1)))
    return float(rmsd)


def extract_protein_list(result_folder):
    protein_set = set()
    
    if not os.path.exists(result_folder):
        raise ValueError(f"Result folder does not exist")
    
    for filename in os.listdir(result_folder):
        if filename.endswith('.sdf'):
            # Extract first 4 characters as protein ID
            pdb_id = filename[:4]
            protein_set.add(pdb_id)
    
    protein_list = sorted(list(protein_set))
    return protein_list


def RMSD_calculation(predicted_sdf_folder, true_sdf_folder, save_path, save_rdkit_path):
    """
    Main function for RMSD calculation.
    Extracts protein list from SDF files in the folder.
    """
    rmsd_list = []
    pose_rank_for_rmsd = []
    protein_list = extract_protein_list(predicted_sdf_folder)
    protein_name_list = []
    for protein in protein_list:
        pose_rank = 0
        # save the location of all the predicted sdf files for the protein
        predicted_sdf_files = [f for f in os.listdir(predicted_sdf_folder) if f.startswith(protein)]
        # save the location of all the true sdf files for the protein
        true_sdf_file = None
        for root, dirs, files in os.walk(true_sdf_folder):
            for file in files:
                if file.startswith(protein) and file.endswith(".sdf"):
                    true_sdf_file = os.path.join(root, file)
                    break


        # Load true SDF (may have hydrogens from PDB)
        true_sdf = Chem.MolFromMolFile(true_sdf_file)
        true_sdf_noH = Chem.RemoveHs(Chem.Mol(true_sdf))
        true_sdf_conf = true_sdf_noH.GetConformer()
        if true_sdf_conf is None:
            print(f"Warning: No conformer found for {true_sdf_file}")
            continue
        
        # Load RDKit template
        rdkit_mol_template = None
        rdkitMolFile = f"{save_rdkit_path}/{protein}_mol_from_rdkit.sdf"
        if os.path.exists(rdkitMolFile):
            rdkit_mol_template = Chem.MolFromMolFile(rdkitMolFile)
        else:
            smiles = Chem.MolToSmiles(true_sdf_noH)
            rdkit_mol_template = Chem.MolFromSmiles(smiles)
        
        rdkit_mol_template = Chem.RemoveHs(rdkit_mol_template)
        
        true_smiles = Chem.MolToSmiles(true_sdf_noH)
        rdkit_smiles = Chem.MolToSmiles(rdkit_mol_template)

        if true_smiles != rdkit_smiles:
            print(f"Warning: True and RDKit mol are not the same")
            continue
        else:
            match_reverse = true_sdf_noH.GetSubstructMatch(rdkit_mol_template)
            if len(match_reverse) == rdkit_mol_template.GetNumAtoms():
                true_coords_reordered = [list(true_sdf_conf.GetAtomPosition(match_reverse[j])) for j in range(rdkit_mol_template.GetNumAtoms())]
            else:
                print(f"Warning: Atom mapping mismatch: match_len={len(match_reverse)}, native={true_sdf_noH.GetNumAtoms()}, rdkit={rdkit_mol_template.GetNumAtoms()}")
                continue

        for predicted_sdf_file in predicted_sdf_files:
            pose_rank += 1
            # read the predicted sdf file
            predicted_sdf = Chem.MolFromMolFile(os.path.join(predicted_sdf_folder, predicted_sdf_file))
            predicted_sdf = Chem.RemoveHs(Chem.Mol(predicted_sdf))
            predicted_sdf_conf = predicted_sdf.GetConformer()
            if predicted_sdf_conf is None:
                print(f"Warning: No conformer found for {predicted_sdf_file}")
                continue
            
            predicted_smiles = Chem.MolToSmiles(predicted_sdf)
            predicted_match = predicted_sdf.GetSubstructMatch(rdkit_mol_template)
            expected_match = list(range(rdkit_mol_template.GetNumAtoms()))  
            if predicted_smiles != true_smiles:
                print(f"Warning: Predicted and true mol are not the same")
                continue                
            elif predicted_match != expected_match:
                predicted_coords_reordered = [list(predicted_sdf_conf.GetAtomPosition(predicted_match[j])) for j in range(rdkit_mol_template.GetNumAtoms())]
            else:
                predicted_coords_reordered = [list(predicted_sdf_conf.GetAtomPosition(predicted_match[j])) for j in range(predicted_sdf.GetNumAtoms())]

            rmsd = compute_coordinate_RMSD(true_coords_reordered, predicted_coords_reordered)

            rmsd_list.append(rmsd)
            pose_rank_for_rmsd.append(pose_rank)
            protein_name_list.append(protein)

    df = pd.DataFrame({'protein': protein_name_list, 'rmsd': rmsd_list, 'pose_rank_for_rmsd': pose_rank_for_rmsd})
    df.to_csv(f"{save_path}/rmsd_list.csv", index=False)
    return df


if __name__ == "__main__":
    # Example usage
    predicted_sdf_folder = "/ocean/projects/cis250160p/wli27/TankBind/tankbind/PL_results_save_sdf/result"
    true_sdf_folder = "/ocean/projects/cis250160p/wli27/P-L"
    save_path = "/ocean/projects/cis250160p/wli27/TankBind/tankbind/PL_results_save_sdf"
    save_rdkit_path = "/ocean/projects/cis250160p/wli27/TankBind/tankbind/PL_results_save_sdf/PL_rdkit"
    
    df = RMSD_calculation(predicted_sdf_folder, true_sdf_folder, save_path, save_rdkit_path)
    print(df.head())
