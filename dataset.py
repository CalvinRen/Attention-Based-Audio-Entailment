import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import ast

# Prepare data loaders
def prepare_dataloaders(config):
    train_csv = config['train_csv']
    val_csv = config['val_csv']
    test_csv = config['test_csv']

    train_audio_dir = config['train_audio_dir']
    val_audio_dir = config['val_audio_dir']
    test_audio_dir = config['test_audio_dir']

    for dir_path in [train_audio_dir, val_audio_dir, test_audio_dir]:
        if not os.path.isdir(dir_path):
            raise ValueError(f"Audio directory does not exist: {dir_path}")

    train_dataset = ClothoEntailmentDataset(train_csv, train_audio_dir)
    val_dataset = ClothoEntailmentDataset(val_csv, val_audio_dir)
    test_dataset = ClothoEntailmentDataset(test_csv, test_audio_dir)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader



# Define the dataset class
class ClothoEntailmentDataset(Dataset):
    def __init__(self, csv_file, audio_dir):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.samples = []

        for idx, row in self.data.iterrows():
            audio_file_name = row['Audio file']
            audio_captions_str = row['Caption']
            if pd.isna(audio_file_name) or not isinstance(audio_file_name, str) or audio_file_name.strip() == '':
                continue  # Skip invalid entries

            audio_file = os.path.join(self.audio_dir, audio_file_name.strip())
            if not os.path.isfile(audio_file):
                continue  # Skip if file does not exist

            # Convert caption string to list
            try:
                audio_captions = ast.literal_eval(audio_captions_str)
            except (SyntaxError, ValueError):
                continue  # Skip if captions cannot be parsed

            # Append samples with labels
            for label, text_column in enumerate(['Entailment', 'Neutral', 'Contradiction']):
                text = row[text_column]
                if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
                    continue  # Skip invalid texts
                sample = {
                    'audio': audio_file,
                    'captions': audio_captions,
                    'text': text.strip(),
                    'label': label
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample

# Custom collate_fn function to ensure that batch samples remain as list of dictionaries
def collate_fn(batch):
    batch_samples = {}
    for key in batch[0]:
        if key == 'captions':
            # Correctly aggregate captions for each sample, keeping it as [batch_size, 5]
            batch_samples[key] = [sample[key] for sample in batch]
        else:
            batch_samples[key] = [sample[key] for sample in batch]

    return batch_samples