import numpy as np
import pandas as pd

def sample_class0(n, n_features, n_intrinsic_dim, var_0, var_noise, rot): 
    """
    Generates samples for class 0.
    
    Parameters:
    - n: Number of samples
    - n_features: Total number of features
    - n_intrinsic_dim: Intrinsic dimension of the data
    - var_0: Variance of the intrinsic dimension
    - var_noise: Noise variance for nuisance dimensions
    - rot: Rotation matrix
    
    Returns:
    - Rotated data for class 0
    """
    n_nuissance_dim = n_features - n_intrinsic_dim
    data = np.zeros([n, n_features])
    mean = np.zeros(n_intrinsic_dim)
    cov = np.eye(n_intrinsic_dim)
    
    data[:, :n_intrinsic_dim] = np.random.multivariate_normal(mean, var_0 * cov, size=n)
    data[:, n_intrinsic_dim:] = np.random.normal(0., np.sqrt(var_noise), [n, n_nuissance_dim])
    
    return data @ rot

def sample_class1(n, n_features, n_intrinsic_dim, r_mean_1, r_var_1, var_noise, rot):      
    """
    Generates samples for class 1.
    
    Parameters:
    - n: Number of samples
    - n_features: Total number of features
    - n_intrinsic_dim: Intrinsic dimension of the data
    - r_mean_1: Mean radius for class 1
    - r_var_1: Variance of radius for class 1
    - var_noise: Noise variance for nuisance dimensions
    - rot: Rotation matrix
    
    Returns:
    - Rotated data for class 1
    """
    n_nuissance_dim = n_features - n_intrinsic_dim
    data = np.zeros([n, n_features])
    
    data_1 = np.random.multivariate_normal(np.zeros(n_intrinsic_dim), np.eye(n_intrinsic_dim) * 2, size=n)
    data_1 = data_1 / np.linalg.norm(data_1, axis=-1)[:, None]
    r = r_mean_1 + np.random.normal(0, np.sqrt(r_var_1), [n, 1])
    
    data[:, :n_intrinsic_dim] = r * data_1
    data[:, n_intrinsic_dim:] = np.random.normal(0., np.sqrt(var_noise), [n, n_nuissance_dim])
    
    return data @ rot

def generate_data(n_features, n_machines, n_datapoints, var_0=1., r_mean_1=3., r_var_1=1., var_noise=1., split_ratio=None, pctg_of_0=None):
    """
    Generates synthetic datasets for multiple machines.
    
    Parameters:
    - n_features: List with total number of features and effective dimension
    - n_machines: Number of machines (partitions)
    - n_datapoints: Total number of datapoints
    - var_0: Variance for class 0
    - r_mean_1: Mean radius for class 1
    - r_var_1: Variance of radius for class 1
    - var_noise: Noise variance for nuisance dimensions
    - split_ratio: List with ratio of data points per machine
    - pctg_of_0: List with percentage of class 0 samples per machine
    
    Returns:
    - Data: Dictionary containing data for each machine
    - rot: Rotation matrix used
    """
    n_feat = n_features[0]
    n_effective_dim = n_features[-1]
    
    # Generate a random rotation matrix
    M = np.random.randn(n_feat, n_feat)
    rot, _ = np.linalg.qr(M)   
    
    dataset = pd.DataFrame()
    Data = {}
    
    # Determine the number of data points per machine
    if split_ratio is None:
        n_data_per_machine = [int(n_datapoints / n_machines)] * n_machines
    else:
        ratios = np.array(split_ratio) / sum(split_ratio)
        n_data_per_machine = [int(n_datapoints * ratios[i]) for i in range(ratios.shape[0])]
    
    for i in range(n_machines):
        if pctg_of_0 is None:
            n_0 = int(0.5 * n_data_per_machine[i])
            n_1 = n_data_per_machine[i] - n_0
        else:
            n_0 = int(pctg_of_0[i] * n_data_per_machine[i])
            n_1 = n_data_per_machine[i] - n_0
        
        # Initialize local data array
        local_data = np.zeros([n_0 + n_1, n_feat + 2])
        local_data[:, -1] = i  # Machine identifier
        local_data[:n_0, -2] = 0  # Class 0
        local_data[n_0:, -2] = 1  # Class 1
        
        # Generate class 0 and class 1 data
        class0_data = sample_class0(n_0, n_feat, n_effective_dim, var_0, var_noise, rot)
        class1_data = sample_class1(n_1, n_feat, n_effective_dim, r_mean_1, r_var_1, var_noise, rot)
        
        # Apply affine transformation
        A = np.random.randn(n_feat, n_feat)
        b = np.random.randn(n_feat) * (i + 1) * 5
        class0_data = class0_data @ A + b
        class1_data = class1_data @ A + b
        
        # Assign data to local_data array
        local_data[:n_0, :-2] = class0_data
        local_data[n_0:, :-2] = class1_data
        
        # Shuffle data
        np.random.shuffle(local_data)
        
        # Store data
        Data[f"DataSet_{i}"] = local_data
        df = pd.DataFrame(local_data)
        df.columns = [f"Feature_{j}" for j in range(n_feat)] + ["ClassCategory_0", "Machine"]
        
        # Save data to CSV files
        df.to_csv(f"./datasets/synthetic/partitions_{i}.csv", index=False)
        
def generate_data_from_config(cfg):
    """
    Generates synthetic data based on the given configuration.

    Parameters:
    - cfg: Hydra configuration object

    Returns:
    - Data: Dictionary containing data for each machine
    - rot: Rotation matrix used
    """
    n_features = cfg.datasets.n_features
    n_machines = cfg.datasets.n_machines
    n_datapoints = cfg.datasets.n_datapoints
    var_0 = cfg.datasets.var_0
    r_mean_1 = cfg.datasets.r_mean_1
    r_var_1 = cfg.datasets.r_var_1
    var_noise = cfg.datasets.var_noise
    split_ratio = cfg.datasets.split_ratio
    pctg_of_0 = cfg.datasets.pctg_of_0
    
    generate_data(n_features, n_machines, n_datapoints, 
                         var_0, r_mean_1, r_var_1, var_noise, 
                         split_ratio=split_ratio, pctg_of_0=pctg_of_0)