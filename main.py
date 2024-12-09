import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import gc
import argparse
import yaml
import wandb

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

    # Initialize W&B
    if args.mode == 'train':
        wandb.init(
            project='attention-based-audio-text-entailment',
            config=config,
            name=config['wandb_run_name']
        )
        wandb.config.update(config)

    # Set the start method to 'spawn' for multiprocessing
    mp.set_start_method('spawn', force=True)
    gc.collect()
    torch.cuda.empty_cache()

    # Main training and validation loop with test evaluation
    try:
        clap_model = CLAP(version='2023', use_cuda=True)
    except Exception as e:
        print(f"Error initializing CLAP model: {e}")
        return

    try:
        train_loader, val_loader, test_loader = prepare_dataloaders(config)
    except ValueError as ve:
        print(f"DataLoader preparation error: {ve}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model selection
    if args.model == 'mlp':
        audio_embed_dim = config['audio_embed_dim']  # Example: 1024
        text_embed_dim = config['text_embed_dim']   # Example: 1024
        
        model_input_dim = audio_embed_dim + text_embed_dim + text_embed_dim  # Dynamically compute input size
        model = AudioTextClassifier(model_input_dim, config['hidden_dim'], config['num_classes'])
    elif args.model == 'attention':
        model = AttentionBasedModel(config)
    else:
        raise ValueError("Invalid model selection. Choose 'mlp' or 'attention'.")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler()

    if args.mode == 'train':
        best_model = None
        best_val_f1 = 0.0

        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

            train_loss, train_precision, train_recall, train_f1, train_accuracy = train(
                model, train_loader, clap_model, optimizer, criterion, scaler, device, args.model, epoch
            )
            print(f"Train Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}")

            # Log training metrics to WandB
            wandb.log({
                'train_loss': train_loss,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'train_accuracy': train_accuracy,
                'epoch': epoch + 1
            })

            val_loss, val_precision, val_recall, val_f1, val_accuracy = validate(
                model, val_loader, clap_model, criterion, device, args.model, epoch
            )
            print(f"Validation Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")

            # Log validation metrics to WandB
            wandb.log({
                'val_loss': val_loss,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'val_accuracy': val_accuracy,
                'epoch': epoch + 1
            })

            # Save metrics and models as needed
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model = model.state_dict()

                # Save the best model
                torch.save(best_model, config['model_save_path'])
                print("Best model saved.")

        if best_model:
            model.load_state_dict(best_model)
            print("Best model loaded based on validation F1 score.")

    wandb.finish()

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
        test_loss, test_precision, test_recall, test_f1, test_accuracy = test(model, test_loader, clap_model, criterion, device, args.model)
        print(f"Test Loss: {test_loss:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}") 

if __name__ == "__main__":
    main()