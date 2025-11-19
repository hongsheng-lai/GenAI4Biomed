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
    df = pd.read_csv(filename)  # Don't use index_col=0 to keep protein_name as a column
    return df

def calculate_average_RMSD(df_RMSD):
    return df_RMSD['min_rmsd'].dropna().mean()

def calculate_top5_success_rate(df_RMSD):
    N = len(df_RMSD)
    success = df_RMSD['top5_success'].dropna().sum()
    return success / N

def read_index_affinity(filename):
    """
    Read INDEX_core_data.2016 file and extract affinity data.
    
    Returns:
        pandas.DataFrame with columns: protein_name, neg_log_kd_ki
    """
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
                'protein_name': protein_name,
                'neg_log_kd_ki': neg_log_kd_ki,  # Main affinity metric (-logKd/Ki)
            })
    
    df = pd.DataFrame(data)
    return df

#def check_predicted_affinity_in_index(df_affinity):
#    affinity = df_affinity['affinity'].sorted

def calculate_PCC_by_affinity(df_affinity, df_true_affinity):
    """
    Calculate Pearson Correlation Coefficient between predicted and true affinities.
    Ensures alignment by protein_name through merge and sort.
    """
    # Merge dataframes by protein_name to align predictions with true values
    merged_df = df_affinity.merge(df_true_affinity, on='protein_name', how='inner')
    
    if len(merged_df) == 0:
        raise ValueError("No matching proteins found between predicted and true affinity data!")
    
    # Sort by protein_name to ensure consistent ordering
    merged_df = merged_df.sort_values('protein_name').reset_index(drop=True)
    
    # Extract aligned affinity values
    true_label = merged_df['neg_log_kd_ki']
    predicted_label = merged_df['affinity']
    
    PCC, p_value = pearsonr(true_label, predicted_label)
    return PCC, p_value

def calculate_SCC_by_affinity(df_affinity, df_true_affinity):
    """
    Calculate Spearman Correlation Coefficient between predicted and true affinities.
    Ensures alignment by protein_name through merge and sort.
    """
    # Merge dataframes by protein_name to align predictions with true values
    merged_df = df_affinity.merge(df_true_affinity, on='protein_name', how='inner')
    
    if len(merged_df) == 0:
        raise ValueError("No matching proteins found between predicted and true affinity data!")
    
    # Sort by protein_name to ensure consistent ordering
    merged_df = merged_df.sort_values('protein_name').reset_index(drop=True)
    
    # Extract aligned affinity values
    true_label = merged_df['neg_log_kd_ki']
    predicted_label = merged_df['affinity']
    
    SCC, p_value = spearmanr(true_label, predicted_label)
    return SCC, p_value

def main():
#    df_affinity = load_data_affinity("/ocean/projects/cis250160p/wli27/TankBind/tankbind/PL_results_top5_new/info_with_predicted_affinity.csv")
#    df_RMSD = load_data_RMSD("/ocean/projects/cis250160p/wli27/TankBind/tankbind/PL_results_top5_new/top5_rmsd_results.csv")
    df_RMSD_affinity = load_data_RMSD_affinity("/ocean/projects/cis250160p/wli27/TankBind/tankbind/PL_results_top5_with_affinity/top5_rmsd_results.csv")
    print(df_RMSD_affinity.head())
    print(df_RMSD_affinity.columns)  # Check available columns
    print(df_RMSD_affinity['min_rmsd'])
    print(df_RMSD_affinity['affinity'])
    print(df_RMSD_affinity['protein_name'])
    average_RMSD = calculate_average_RMSD(df_RMSD_affinity)
    top5_success_rate = calculate_top5_success_rate(df_RMSD_affinity)
    print(f"Average RMSD: {average_RMSD}")
    print(f"Top5 Success Rate: {top5_success_rate}")
    df_true_affinity = read_index_affinity("/ocean/projects/cis250160p/wli27/P-L/INDEX_core_data.2016")
    print(f"Number of true affinity entries: {len(df_true_affinity)}")
    print(f"Number of predicted affinity entries: {len(df_RMSD_affinity)}")
    
    # Merge to check alignment
    merged_check = df_RMSD_affinity.merge(df_true_affinity, on='protein_name', how='inner')
    print(f"Number of matched proteins (for correlation): {len(merged_check)}")
    
    PCC, p_value = calculate_PCC_by_affinity(df_RMSD_affinity, df_true_affinity)
    SCC, p_value = calculate_SCC_by_affinity(df_RMSD_affinity, df_true_affinity)
    print(f"PCC: {PCC:.4f}, p_value: {p_value:.4e}")
    print(f"SCC: {SCC:.4f}, p_value: {p_value:.4e}")


if __name__ == "__main__":
    main()