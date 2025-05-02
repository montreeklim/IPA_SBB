import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# For example, using the occupancy_detection dataset (ID: 357)
datasets = {
    # 'occupancy_detection': 357,
    # 'covertype': 31,
    # 'adult': 2,
    'bank_marketing': 222,
    'skin_segmentation': 229,
    'online_shoppers': 468
}

for name, dataset_id in datasets.items():
    print(f"Processing dataset: {name} (ID: {dataset_id})")
    try:
        # Fetch the dataset from UCI repository using the real ID
        dataset = fetch_ucirepo(id=dataset_id)
        
        # Extract features and targets as pandas DataFrames
        X_df = dataset.data.features.copy()  # use copy() to avoid SettingWithCopyWarning
        y_df = dataset.data.targets.copy()
        
        # Drop any feature columns that contain NaN values
        X_df = X_df.dropna(axis=1)
        
        # If there is a non-numeric column like 'date', drop it
        if 'date' in X_df.columns:
            X_df = X_df.drop(columns=['date'])
        
        # Convert each column to numeric (if not already) - non-convertible values will become NaN
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        
        # Now keep only columns with numeric data (should be all remaining ones)
        X_df = X_df.select_dtypes(include=[np.number])
        
        # Drop any columns that turned completely into NaN after conversion
        X_df = X_df.dropna(axis=1, how='all')
        
        # Print columns to check what's remaining
        print("Remaining columns:", X_df.columns.tolist())
        
        # Convert target DataFrame to a 1D pandas Series (assumes a single column)
        y_df = y_df.iloc[:, 0]
        
        # Drop rows where y_df is NaN and apply the same mask to X_df
        valid_mask = ~y_df.isna()
        X_df = X_df[valid_mask]
        y_df = y_df[valid_mask]
        
        # Convert to NumPy arrays
        X = X_df.to_numpy()
        y = y_df.to_numpy()
        
        # Verify that there are exactly 2 unique classes in the target vector
        classes = np.unique(y)
        # if len(classes) != 2:
        #     print(f"Skipping dataset '{name}': expected exactly 2 classes, but got {classes}")
        #     continue
        
        # Create arrays A and B using boolean indexing based on the two target classes
        A = X[y == classes[0]]
        B = X[y == classes[1]]
        
        # Save the arrays to disk as .npy files
        np.save(f"{name}_A.npy", A)
        np.save(f"{name}_B.npy", B)
        
        print(f"Saved '{name}_A.npy' (shape: {A.shape}) and '{name}_B.npy' (shape: {B.shape})")
    
    except Exception as e:
        print(f"Error processing dataset '{name}' with ID {dataset_id}: {e}")
