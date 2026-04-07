import os
import time
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class PhysioNetDataset(Dataset):
    def __init__(self, samples, labels):
        self.data_x = samples
        self.data_y = labels
        
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    
    def __len__(self):
        return len(self.data_x)

def custom_collate(batch):
    """Custom collate function that handles padding and creates attention masks"""
    # Separate sequences and labels
    sequences, labels = zip(*batch)
    
    # Get max length in this batch
    max_len = max(seq.size(0) for seq in sequences)
    
    # Get feature dimension from first sequence
    feature_dim = sequences[0].size(1)
    
    # Create padded batch tensor
    padded_sequences = torch.zeros(len(sequences), max_len, feature_dim)
    
    # Create attention mask [batch_size, seq_len]
    # 1 = actual data, 0 = padding
    attention_mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    
    # Fill padded tensor and attention mask
    for i, seq in enumerate(sequences):
        end = seq.size(0)
        padded_sequences[i, :end] = seq
        attention_mask[i, :end] = True  # Mark actual data positions
    
    # Stack labels
    labels = torch.stack(labels)
    
    return padded_sequences, labels, attention_mask

def load_physionet_data(data_path, outcomes_path, config):
    """Load PhysioNet data preserving temporal measurements"""
    # Load outcomes
    outcomes_df = pd.read_csv(outcomes_path)
    
    # Use features from config
    dynamic_features = config.feature_list
    feature_map = {f: i for i, f in enumerate(dynamic_features)}
    
    samples = []
    labels = []
    
    # Process each patient file
    for filename in os.listdir(data_path):
        if not filename.endswith('.txt') or filename.startswith('Outcomes'):
            continue
        
        try:
            record_id = int(filename.split('.')[0])
            df = pd.read_csv(os.path.join(data_path, filename), engine='python')
            
            # Keep only dynamic measurements
            df = df[df['Parameter'].isin(dynamic_features)]
            
            # Convert time to hours since admission
            df['Time'] = df['Time'].apply(
                lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 if ':' in str(x) else 0
            )
            
            # Sort by time
            df = df.sort_values('Time')
            
            # Create sequence of measurements [time_steps, features]
            time_points = df.groupby('Time')
            sequence = []
            
            for _, group in time_points:
                features = torch.zeros(len(dynamic_features))
                for _, row in group.iterrows():
                    feat_idx = feature_map[row['Parameter']]
                    features[feat_idx] = row['Value']
                sequence.append(features)
            
            # Convert to tensor [time_steps, features]
            sequence = torch.stack(sequence)
            
            samples.append(sequence)
            labels.append(
                outcomes_df[outcomes_df['RecordID'] == record_id]['In-hospital_death'].iloc[0]
            )
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue
    
    # Convert labels to tensor
    labels = torch.tensor(labels)
    
    return samples, labels

def physionet_data_pipeline(data_path, outcomes_path, config, split='train'):
    """Main pipeline function with attention masking"""
    # Load data
    samples, labels = load_physionet_data(data_path, outcomes_path, config)
    
    # Create dataset
    dataset = PhysioNetDataset(samples, labels)
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == 'train'),
        drop_last=config.use_grad_accum,
        collate_fn=custom_collate
    )
    
    return dataset, dataloader