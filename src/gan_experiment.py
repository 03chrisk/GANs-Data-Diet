import os
import torch
import numpy as np
import pandas as pd
import argparse
import time
from datetime import datetime
from pathlib import Path

# Import your modules
from train_gan import train_gan
from utils.data_utils import set_random_seed
import configs.mnist_config as mnist_config
import configs.cifar_config as cifar_config
from utils.eval_utils import calculate_fid


class GANExperiment:
    """Class to manage experiments for training GANs on different subsets."""
    
    def __init__(self, dataset_type, subset_strategy, num_trials=3, seed=64):
        """
        Initialize the experiment.
        
        Args:
            dataset_type: Type of dataset ('digits', 'fashion', 'cifar10')
            subset_strategy: Type of subset strategy ('random', 'easiest', 'hardest', 'easiest_balanced')
            num_trials: Number of trials to run for each configuration
            seed: Base random seed
        """
        self.dataset_type = dataset_type
        self.subset_strategy = subset_strategy
        self.num_trials = num_trials
        self.base_seed = seed
        
        # Select the appropriate config
        self.config = cifar_config if dataset_type == 'cifar10' else mnist_config
        
        # Create timestamp for this experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        self.results_dir = f"results/{dataset_type}_{subset_strategy}_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Results file path
        self.results_file = os.path.join(self.results_dir, "results.csv")
        
        # Model save directory
        self.models_dir = f"models/{dataset_type}_{subset_strategy}"
        
        # Initialize results list
        self.results = []
        
        print(f"Experiment initialized: {dataset_type}, {subset_strategy}")
        print(f"Results will be saved to: {self.results_dir}")
    
    def get_subset_percentages(self):
        """
        Get the list of percentages to test based on the subset strategy.
        
        Returns:
            List of percentage values
        """
        if self.subset_strategy == 'random':
            # For random strategy, use standard percentages
            return [50, 60, 70, 80, 90, 100]
        else:
            return [50, 60, 70, 80, 90]

    
    def run_trial(self, percentage, trial):
        """
        Run a single training and evaluation trial.
        
        Args:
            percentage: Percentage of data to use
            trial: Trial number
            
        Returns:
            Dictionary with trial results
        """
        print(f"\nTrial {trial}/{self.num_trials}")
        
        # Set seed for reproducibility (different for each trial)
        trial_seed = self.base_seed + trial
        set_random_seed(trial_seed)
        print(f"Using seed: {trial_seed}")
        
        # Define model save path
        model_save_path = os.path.join(self.models_dir, f"{percentage}p/trial_{trial}")
        os.makedirs(model_save_path, exist_ok=True)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Train the GAN
            g_losses, d_losses, generator = train_gan(
                subset_percentage=percentage,
                dataset_type=self.dataset_type,
                subset_strategy=self.subset_strategy,
                save_path=model_save_path,
                config = self.config,
            )
            
            train_time = time.time() - start_time

            
            # Calculate FID score
            fid_start = time.time()
            fid_score = calculate_fid(
                generator=generator,
                dataset_type=self.dataset_type,
                num_samples=10000,
            )
            fid_time = time.time() - fid_start
            
            # Record results
            result = {
                'dataset': self.dataset_type,
                'strategy': self.subset_strategy,
                'percentage': percentage,
                'trial': trial,
                'fid_score': fid_score,
                'train_time': train_time,
                'fid_time': fid_time,
                'total_time': train_time + fid_time
            }
            
            print(f"Trial {trial} complete. FID: {fid_score:.4f}")
            
            return result
            
        except Exception as e:
            print(f"Error in trial {trial}: {str(e)}")
            # Log the error but continue with next trial
            return {
                'dataset': self.dataset_type,
                'strategy': self.subset_strategy,
                'percentage': percentage,
                'trial': trial,
                'error': str(e)
            }
    
    def save_results(self):
        """Save current results to CSV files."""
        if not self.results:
            return
            
        # Save detailed results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.results_file, index=False)
        print(f"Results saved to {self.results_file}")
    
    
    def run(self):
        """Run the complete experiment for all percentages and trials."""
        percentages = self.get_subset_percentages()
        
        if not percentages:
            print("No percentages to test. Experiment cannot proceed.")
            return
        
        print(f"Running experiment with percentages: {percentages}")
        
        total_start_time = time.time()
        
        for percentage in percentages:
            print(f"\n{'='*80}")
            print(f"Dataset: {self.dataset_type}, Strategy: {self.subset_strategy}, Percentage: {percentage}%")
            print(f"{'='*80}")
            
            for trial in range(1, self.num_trials + 1):
                # Run the trial
                result = self.run_trial(percentage, trial)
                
                # Add result to our collection
                self.results.append(result)
                
                # Save after each trial in case of interruption
                self.save_results()
        
        total_time = time.time() - total_start_time
        print(f"\nExperiment complete. Total time: {total_time/60:.2f} minutes")
        
        return self.results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run GAN experiments on different data subsets')
    
    parser.add_argument('--dataset', type=str, default='digits',
                        choices=['digits', 'fashion', 'cifar10'],
                        help='Dataset type (default: digits)')
    
    parser.add_argument('--strategy', type=str, default='random',
                        choices=['random', 'easiest', 'hardest', 'easiest_balanced'],
                        help='Subset selection strategy (default: random)')
    
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials for each configuration (default: 3)')
    
    parser.add_argument('--seed', type=int, default=64,
                        help='Base random seed (default: 64)')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Display configuration
    print("\nExperiment Configuration:")
    print(f"Dataset: {args.dataset}")
    print(f"Subset Strategy: {args.strategy}")
    print(f"Number of Trials: {args.trials}")
    print(f"Training Epochs: {mnist_config.NUM_EPOCHS}")
    print(f"Base Random Seed: {args.seed}")
    
    # Create and run the experiment
    experiment = GANExperiment(
        dataset_type=args.dataset,
        subset_strategy=args.strategy,
        num_trials=args.trials,
        seed=args.seed
    )
    
    # Run the experiment
    results = experiment.run()
    
    print("\nExperiment completed successfully!")