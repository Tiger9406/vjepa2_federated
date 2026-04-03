import os
import pandas as pd

def verify_dataset_csv(csv_path):
    print(f"--- Verifying {csv_path} ---")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found at: {csv_path}\n")
        return
        
    try:
        df = pd.read_csv(csv_path, header=None, delimiter=" ")
    except Exception as e:
        print(f"[ERROR] Pandas failed to read CSV. Check formatting. Error: {e}\n")
        return
        
    if len(df.columns) < 2:
        print(f"[ERROR] Expected 2 columns (path, label), but got {len(df.columns)}.\n")
        return
        
    paths = df.values[:, 0]
    labels = df.values[:, 1]
    
    print(f"Successfully parsed {len(paths)} rows.")
    print(f"Sample Row 0 -> Path: {paths[0]} | Label: {labels[0]}")
    
    if not os.path.isabs(str(paths[0])):
        print(f"[WARNING] V-JEPA expects absolute paths. Found relative path: {paths[0]}")
    else:
        print("Paths r absolute")
        
    try:
        int(labels[0])
        print("Labels are valid integers.")
    except ValueError:
        print(f"[ERROR] Labels must be integers. Found: {labels[0]}")
        
    missing_count = 0
    sample_size = min(100, len(paths))
    for p in paths[:sample_size]:
        if not os.path.exists(str(p)):
            missing_count += 1
            
    if missing_count > 0:
        print(f"[ERROR] {missing_count} out of the first {sample_size} video files DO NOT EXIST on disk!")
        print(f"Example missing file: {paths[0]}")
    else:
        print(f"First {sample_size} video files successfully located on disk.")
        
    print("------------------------------------------\n")

if __name__ == "__main__":
    verify_dataset_csv("./eval_ucf101/ucf101_train.csv")
    verify_dataset_csv("./eval_ucf101/ucf101_val.csv")