import logging
import os
import traceback
from typing import Dict, List, Tuple
import sys
import wandb


import pytorch_lightning as pl
# torchaudio gives wierd errors
# use this thread to fix: https://github.com/pytorch/audio/issues/62
# pip install -U torch torchaudio --no-cache-dir
import torch
from torch.cuda import OutOfMemoryError

from args.training_args import parser, apply_subset_arguments
from models.models import DummyModel, load_model_from_run_name
from utils.wandb_utils import create_wandb_logger
from utils.data_module import GeneralDataModule
from utils.dataset_utils import CustomDataModule, get_dataloaders, get_dummy_dataloader
from utils.training_utils import train_model
from utils.inference_utils import save_samples, generate_samples
from utils.evaluation_utils import load_samples, evaluate_metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Run Experiment Pipeline for Generative Audio Model Training!')


def process_results(args):
    # If anything needs to be processed after training/testing, it should go here
    return


def main():
    try:
        # ================================
        # SET UP ARGS
        # ================================
        args = parser.parse_args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ================================
        # CONFIGURE WANDB
        # ================================
        if args.disable_wandb:
            os.environ['WANDB_MODE'] = 'disabled'
        wandb.init(project=args.wandb_project_name, config=args)
    
        wandb_logger = create_wandb_logger(args, args.wandb_project_name)
        wandb.run.name = args.wandb_run_name
        
        if args.experiment_type == 'training':
            # Initialize the data module
            data_module = GeneralDataModule(
                data_dir=args.data_dir, 
                dataset_name=args.dataset_name, 
                batch_size=args.batch_size, 
                image_size=args.image_size
            )
            data_module.prepare_data()
            data_module.setup()
    
            train_loader, val_loader, test_loader = get_dataloaders(args)
            data_module = CustomDataModule(train_loader, val_loader, test_loader)
            
            train_model(args, data_module, wandb_logger)
            
            process_results(args)
            
        
        elif args.experiment_type == 'inference':
            return
            
        elif args.experiment_type == 'evaluation':
            return
            
        # ================================
        # FINISH
        # ================================
        wandb.finish()
        

    except OutOfMemoryError as oom_error:
        # Log the error to wandb
        logger.warning(str(oom_error))
        wandb.log({"error": str(oom_error)})

        # Mark the run as failed
        wandb.run.fail()
        
        wandb.finish(exit_code=-1)
        

    # except Exception as e:
    #     print(traceback.print_exc(), file=sys.stderr)
    #     print(f"An error occurred: {e}\n Terminating run here.")
    #     # Log error message to wandb
    #     wandb.log({"critical_error": str(e)})
    #     # Finish the wandb run without specifying exit_code if fail() is not available
    #     wandb.finish(exit_code=-1)

        
if __name__ == '__main__':
    main()