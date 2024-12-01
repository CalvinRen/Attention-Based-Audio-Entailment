import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import gc
import argparse
import yaml

from msclap import CLAP
from train import train, validate, test
from dataset import prepare_dataloaders
from mlp import AudioTextClassifier
from attention_model import AttentionBasedModel


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Audio-Text Entailment Training and Testing')
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'test'], default='train',
                        help='Mode to run: train or test')
    parser.add_argument('-c', '--config', type=str, default='config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('-md', '--model', type=str, choices=['mlp', 'attention'], default='mlp',
                        help='Model to use: mlp or attention')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set the start method to 'spawn' for multiprocessing
    mp.set_start_method('spawn', force=True)
    gc.collect()
    torch.cuda.empty_cache()

    # Main training and validation loop with test evaluation
    try:
        clap_model = CLAP(version='2023', use_cuda=config['use_cuda'])
    except Exception as e:
        print(f"Error initializing CLAP model: {e}")
        return

    try:
        train_loader, val_loader, test_loader = prepare_dataloaders(config)
    except ValueError as ve:
        print(f"DataLoader preparation error: {ve}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu')

    # Model selection
    if args.model == 'mlp':
        model_input_dim = config['audio_embed_dim'] + config['text_embed_dim'] * 2  # 1024 * 3 = 3072
        model = AudioTextClassifier(model_input_dim, config['hidden_dim'], config['num_classes'])
    elif args.model == 'attention':
        model = AttentionBasedModel(config)
    else:
        raise ValueError("Invalid model selection. Choose 'mlp' or 'attention'.")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scaler = torch.cuda.amp.GradScaler()

    if args.mode == 'train':
        best_model = None
        best_val_f1 = 0.0

        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            train_loss, train_acc = train(model, train_loader, clap_model, optimizer, criterion, scaler, device, args.model)
            print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc * 100:.2f}%")

            val_loss, val_acc, val_f1 = validate(model, val_loader, clap_model, criterion, device, args.model)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc * 100:.2f}%, F1 Score: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model = model.state_dict()

                # Save the best model
                torch.save(best_model, config['model_save_path'])
                print("Best model saved.")

        if best_model:
            model.load_state_dict(best_model)
            print("Best model loaded based on validation F1 score.")

    if args.mode == 'test' or (args.mode == 'train' and best_model):
        if args.mode == 'test':
            # Load the best model
            if os.path.exists(config['model_save_path']):
                model.load_state_dict(torch.load(config['model_save_path']))
                print("Model loaded for testing.")
            else:
                print("Model path does not exist. Please train the model first.")
                return

        # Test the best model on the test set
        test_loss, test_acc, test_f1 = test(model, test_loader, clap_model, criterion, device, args.model)
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%, F1 Score: {test_f1:.4f}")

if __name__ == "__main__":
    main()