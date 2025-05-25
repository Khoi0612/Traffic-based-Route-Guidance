import sys
import argparse
from src.models.models_manager import ModelManager


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Evaluate trained traffic prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model selection argument
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('model_name', nargs='?', help='Name of the model file to evaluate')
    group.add_argument('-a', '--all', action='store_true', help='Evaluate all available models')
    
    # Display plots flag
    parser.add_argument('-d', '--display', action='store_true', help='Display plots/figures')
    
    return parser.parse_args()


def list_available_models(model_manager):
    models = model_manager.get_available_models()
    
    if not models:
        print("No trained models found. Please train models first.")
        return None
    
    print(f"\nAvailable models ({len(models)}):")
    print("-" * 40)
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    return models


def find_model_by_partial_name(model_manager, partial_name):
    available_models = model_manager.get_available_models()
    
    # Try exact match first
    if partial_name in available_models:
        return partial_name
    
    # Try partial matches
    matches = [model for model in available_models if partial_name.lower() in model.lower()]
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"\nMultiple models match '{partial_name}':")
        for i, model in enumerate(matches, 1):
            print(f"{i}. {model}")
        return None
    else:
        return None


def evaluate_single_model(model_manager, model_name, show_plots):
    try:
        # Try to find the model
        found_model = find_model_by_partial_name(model_manager, model_name)
        
        if not found_model:
            print(f"Model '{model_name}' not found.")
            available_models = list_available_models(model_manager)
            if available_models:
                print(f"\nDid you mean one of these models?")
            return False
        
        print(f"Evaluating model: {found_model}")
        if show_plots:
            print("Plot will be displayed...")
        
        # Evaluate the model - only create plot if we need it
        rmse, predictions, y_test, dates_test, fig = model_manager.evaluate_model(
            found_model, 
            show_plot=show_plots,
            create_plot=show_plots  # Only create if we're going to show it
        )
        
        print(f"\nEvaluation completed successfully!")
        print(f"Model: {found_model}")
        print(f"RMSE: {rmse:.2f}")
        
        if show_plots and fig is not None:
            import matplotlib.pyplot as plt
            plt.show()
            plt.close(fig)  # Clean up after showing
        elif fig is not None:
            # This shouldn't happen with our new logic, but just in case
            import matplotlib.pyplot as plt
            plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"Error evaluating model '{model_name}': {str(e)}")
        # Clean up any figures that might have been created
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        return False


def evaluate_all_models(model_manager, show_plots):
    import matplotlib.pyplot as plt
    
    try:
        available_models = model_manager.get_available_models()
        
        if not available_models:
            print("No trained models found. Please train models first.")
            return False
        
        print(f"Evaluating all {len(available_models)} available models...")
        if show_plots:
            print("Plots will be displayed for each model...")
        
        # Set matplotlib backend for better memory management
        if show_plots:
            plt.ion()  # Turn on interactive mode
        else:
            # Use non-interactive backend to save memory
            import matplotlib
            matplotlib.use('Agg')
        
        results, figures = model_manager.evaluate_all_models(show_plots=show_plots)
        
        if results:
            print(f"\nSuccessfully evaluated {len(results)} models.")
        else:
            print("\nNo models were successfully evaluated.")
            return False
        
        # Clean up any remaining figures
        plt.close('all')
            
        return True
        
    except Exception as e:
        print(f"Error evaluating all models: {str(e)}")
        # Clean up any remaining figures
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        return False


def main():
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Check if any models exist
        available_models = model_manager.get_available_models()
        if not available_models:
            print("No trained models found in the models directory.")
            print("Please train models first using the training script.")
            sys.exit(1)
        
        success = False
        
        if args.all:
            # Evaluate all models
            success = evaluate_all_models(model_manager, args.display)
        else:
            # Evaluate specific model
            success = evaluate_single_model(model_manager, args.model_name, args.display)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()