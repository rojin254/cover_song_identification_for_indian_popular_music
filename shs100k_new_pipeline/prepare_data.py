import os
import pandas as pd

def prepare_shs100k(csv_path, out_txt_path):
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path)
    
    with open(out_txt_path, 'w') as f:
        # Group by clique to find the minimum version
        for clique, group in df.groupby('clique'):
            min_version = group['version'].min()
            for _, row in group.iterrows():
                # Extract filename without extension
                filename = row['filename'].rsplit('.', 1)[0]
                is_original = 1 if row['version'] == min_version else 0
                f.write(f"{filename},{clique},{is_original}\n")
                
    print(f"Saved to {out_txt_path}")

if __name__ == '__main__':
    base_dir = '/teamspace/studios/this_studio/PP4'
    os.makedirs(os.path.join(base_dir, 'shs100k_new_pipeline'), exist_ok=True)
    
    # Process train
    train_csv = os.path.join(base_dir, 'data/shs100k-train.csv')
    if os.path.exists(train_csv):
        prepare_shs100k(train_csv, os.path.join(base_dir, 'shs100k_new_pipeline/shs100k_train.txt'))
    else:
        print(f"{train_csv} not found")
        
    # Process test
    test_csv = os.path.join(base_dir, 'data/shs100k-test.csv')
    if os.path.exists(test_csv):
        prepare_shs100k(test_csv, os.path.join(base_dir, 'shs100k_new_pipeline/shs100k-test.txt'))
    else:
        print(f"{test_csv} not found")
