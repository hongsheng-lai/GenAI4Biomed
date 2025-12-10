import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr

#def load_data_affinity(filename):
#    df = pd.read_csv(filename, index_col=0)
#    return df

#def load_data_RMSD(filename):
#    df = pd.read_csv(filename, index_col=0)
#    return df

def load_data_RMSD_affinity(filename):
    df = pd.read_csv(filename) 
    return df

def calculate_average_RMSD(df_RMSD):
    return df_RMSD['best_rmsd'].dropna().mean()


def calculate_top5_success_rate(df_RMSD):
    groups_len = len(df_RMSD['protein'].unique())
    success = 0
    for i in range(groups_len):
        group = df_RMSD[df_RMSD['protein'] == df_RMSD['protein'].unique()[i]]
        group_len = len(group)
        for j in range(min(group_len, 5)):  # Only check top 5 poses
            if group['rmsd'].iloc[j] < 5:
                success += 1
                break
    return success / groups_len

def select_best_rmsd(df_RMSD):
    groups_len = len(df_RMSD['protein'].unique())
    best_rmsd_list = []
    best_pose_rank_list = []
    for i in range(groups_len):
        group = df_RMSD[df_RMSD['protein'] == df_RMSD['protein'].unique()[i]]
        group_len = len(group)
        best_rmsd = group['rmsd'].iloc[0]
        best_pose_rank = 1  # Initialize to 1 (1-indexed: first pose is rank 1)
        for j in range(group_len):
            if group['rmsd'].iloc[j] < best_rmsd:
                best_rmsd = group['rmsd'].iloc[j]
                best_pose_rank = j + 1
        best_rmsd_list.append(best_rmsd)
        best_pose_rank_list.append(best_pose_rank)
    df_best_rmsd = pd.DataFrame({'protein': df_RMSD['protein'].unique(), 'best_rmsd': best_rmsd_list, 'best_pose_rank': best_pose_rank_list})
    return df_best_rmsd

def select_affinity_by_pose_rank(df_affinity, best_pose_rank_list, protein_name_list):
    affinity_list = []
    for i in range(len(protein_name_list)):
        group = df_affinity[df_affinity['protein_name'] == protein_name_list[i]]
        affinity_list.append(group['affinity'].iloc[best_pose_rank_list[i] - 1])
    df_best_affinity = pd.DataFrame({'protein': protein_name_list, 'affinity': affinity_list})
    return df_best_affinity

def read_index_affinity(filename):
    data = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Skip comment lines
            if line.strip().startswith('#'):
                continue
            
            # Skip empty lines
            if not line.strip():
                continue
            
            # Split by whitespace
            parts = line.strip().split()
            
            if len(parts) < 4:
                continue
            
            # Extract basic fields
            protein_name = parts[0]
            neg_log_kd_ki = float(parts[3])  # -logKd/Ki (affinity in log units)            
            
            
            data.append({
                'protein': protein_name,
                'neg_log_kd_ki': neg_log_kd_ki,  # Main affinity metric (-logKd/Ki)
            })
    
    df = pd.DataFrame(data)
    return df

#def check_predicted_affinity_in_index(df_affinity):
#    affinity = df_affinity['affinity'].sorted

def calculate_PCC_by_affinity(df_affinity, df_true_affinity):
    # Merge dataframes by protein_name to align predictions with true values
    merged_df = df_affinity.merge(df_true_affinity, on='protein', how='inner')
    
    if len(merged_df) == 0:
        raise ValueError("No matching proteins found between predicted and true affinity data!")
    
    # Sort by protein_name to ensure consistent ordering
    merged_df = merged_df.sort_values('protein').reset_index(drop=True)
    
    # Extract aligned affinity values
    true_label = merged_df['neg_log_kd_ki']
    predicted_label = merged_df['affinity']
    
    PCC, p_value = pearsonr(true_label, predicted_label)
    return PCC, p_value

def calculate_SCC_by_affinity(df_affinity, df_true_affinity):
    # Merge dataframes by protein_name to align predictions with true values
    merged_df = df_affinity.merge(df_true_affinity, on='protein', how='inner')
    
    if len(merged_df) == 0:
        raise ValueError("No matching proteins found between predicted and true affinity data!")
    
    # Sort by protein_name to ensure consistent ordering
    merged_df = merged_df.sort_values('protein').reset_index(drop=True)
    
    # Extract aligned affinity values
    true_label = merged_df['neg_log_kd_ki']
    predicted_label = merged_df['affinity']
    
    SCC, p_value = spearmanr(true_label, predicted_label)
    return SCC, p_value

def main():
#    df_affinity = load_data_affinity("/ocean/projects/cis250160p/wli27/TankBind/tankbind/PL_results_top5_new/info_with_predicted_affinity.csv")
#    df_RMSD = load_data_RMSD("/ocean/projects/cis250160p/wli27/TankBind/tankbind/PL_results_top5_new/top5_rmsd_results.csv")
    df_RMSD = load_data_RMSD_affinity("/ocean/projects/cis250160p/wli27/TankBind/tankbind/PL_results_save_sdf/rmsd_list.csv")
    df_best_rmsd = select_best_rmsd(df_RMSD)
    protein_name_list = df_best_rmsd['protein'].unique()
    df_affinity = load_data_RMSD_affinity("/ocean/projects/cis250160p/wli27/TankBind/tankbind/PL_results_save_sdf/top5_pose_affinities.csv")
    df_best_affinity = select_affinity_by_pose_rank(df_affinity, df_best_rmsd['best_pose_rank'], protein_name_list)
    average_RMSD = calculate_average_RMSD(df_best_rmsd)
    top5_success_rate = calculate_top5_success_rate(df_RMSD)
    print(f"Average RMSD: {average_RMSD}")
    print(f"Top5 Success Rate: {top5_success_rate}")
    df_true_affinity = read_index_affinity("/ocean/projects/cis250160p/wli27/P-L/INDEX_core_data.2016")
    print(f"Number of true affinity entries: {len(df_true_affinity)}")
    print(f"Number of predicted affinity entries: {len(df_best_affinity)}")
    
    # Merge to check alignment
    merged_check = df_best_affinity.merge(df_true_affinity, on='protein', how='inner')
    print(f"Number of matched proteins (for correlation): {len(merged_check)}")
    
    PCC, p_value = calculate_PCC_by_affinity(df_best_affinity, df_true_affinity)
    SCC, p_value = calculate_SCC_by_affinity(df_best_affinity, df_true_affinity)
    print(f"PCC: {PCC:.4f}, p_value: {p_value:.4e}")
    print(f"SCC: {SCC:.4f}, p_value: {p_value:.4e}")


if __name__ == "__main__":
    main()