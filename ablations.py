import os
import subprocess
import time

def run_ablation_experiments(config_dir, script_path, model_type='mlp', mode='train'):
    """
    Runs ablation experiments sequentially using different configs.

    Args:
        config_dir (str): Directory containing the configuration files.
        script_path (str): Path to the main script to execute experiments.
        model_type (str): Model type ('mlp' or 'attention').
        mode (str): Mode to run ('train' or 'test').
    """
    # Get a list of all config files in the directory
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    config_files.sort()  # Optional: sort files for consistent ordering

    # Ensure output directories exist
    os.makedirs("logs", exist_ok=True)

    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        log_file = f"logs/{os.path.splitext(config_file)[0]}_{mode}.log"

        print(f"Running experiment with config: {config_file}")

        # Construct the command to run the script
        command = [
            "python", script_path,
            "-c", config_path,
            "-md", model_type,
            "-m", mode
        ]

        # Run the command and log output
        with open(log_file, "w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            process.wait()  # Wait for the process to complete

        print(f"Experiment with config {config_file} completed. Log saved to {log_file}.")

        # Optional: Add a delay between experiments
        time.sleep(2)

if __name__ == "__main__":
    # Define paths and parameters
    CONFIG_DIR = "configs/"         # Directory containing the config files
    SCRIPT_PATH = "main.py"         # Path to the main script
    MODEL_TYPE = "attention"              # Model type ('mlp' or 'attention')
    MODE = "train"                  # Mode to run ('train' or 'test')

    # Run the experiments
    run_ablation_experiments(CONFIG_DIR, SCRIPT_PATH, MODEL_TYPE, MODE)