import os
import sys
import argparse
from src.models.models_manager import ModelManager


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train machine learning models for traffic prediction")
    parser.add_argument('input_dir', nargs='?', default=os.path.join('data', 'processed'),
                        help="Directory containing processed data files")
    parser.add_argument('output_dir', nargs='?', default=os.path.join('data', 'models'),
                        help="Directory to save trained models")
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Verify input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)
    
    # Create model manager instance
    manager = ModelManager(train_data_dir=args.input_dir, models_dir=args.output_dir)
    
    # Train all models
    print(f"Training models using data from {args.input_dir}")
    print(f"Trained models will be saved to {args.output_dir}")
    
    try:
        manager.train_all_models()
        print("\nTraining complete!")
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()