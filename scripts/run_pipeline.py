#!/usr/bin/env python3
"""
Main pipeline runner for DevEthOps-LLM-CICD.
This script orchestrates the entire ethical ML pipeline.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_pipeline(config_path, dataset, stage, model_type):
    """Run the DevEthOps pipeline."""
    logger = setup_logging()
    logger.info("Starting DevEthOps Pipeline")
    
    try:
        from devethops.config import ConfigManager
        from devethops.pipeline import EthicalMLPipeline
        
        # Load configuration
        config = ConfigManager(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        
        # Initialize pipeline
        pipeline = EthicalMLPipeline(config)
        logger.info("Pipeline initialized")
        
        # Run specified stage or full pipeline
        if stage == 'all':
            logger.info("Running full pipeline (Build -> Test -> Deploy -> Monitor)")
            results = pipeline.run_full_pipeline(dataset_name=dataset, model_type=model_type)
        elif stage == 'build':
            logger.info("Running Build stage")
            results = pipeline.run_build_stage(dataset_name=dataset, model_type=model_type)
        elif stage == 'test':
            logger.info("Running Test stage")
            results = pipeline.run_test_stage(dataset_name=dataset, model_type=model_type)
        elif stage == 'deploy':
            logger.info("Running Deploy stage")
            results = pipeline.run_deploy_stage(model_type=model_type)
        elif stage == 'monitor':
            logger.info("Running Monitor stage")
            results = pipeline.run_monitor_stage()
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        logger.info("Pipeline completed successfully")
        logger.info(f"Results: {results}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='DevEthOps Ethical ML Pipeline')
    
    parser.add_argument(
        '--config', 
        default='configs/settings.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--dataset', 
        choices=['ibm_hr', 'adult', 'mimic'],
        default='ibm_hr',
        help='Dataset to use'
    )
    
    parser.add_argument(
        '--stage', 
        choices=['all', 'build', 'test', 'deploy', 'monitor'],
        default='all',
        help='Pipeline stage to run'
    )
    
    parser.add_argument(
        '--model', 
        choices=['logistic_regression', 'random_forest', 'xgboost', 'roberta'],
        default='logistic_regression',
        help='Model type to train'
    )
    
    parser.add_argument(
        '--test-only', 
        action='store_true',
        help='Run quick test to verify pipeline works'
    )
    
    args = parser.parse_args()
    
    if args.test_only:
        print("Running pipeline test...")
        # Import and run the test
        sys.path.insert(0, os.path.dirname(__file__))
        import test_pipeline
        success = test_pipeline.main()
        sys.exit(0 if success else 1)
    
    try:
        results = run_pipeline(
            config_path=args.config,
            dataset=args.dataset,
            stage=args.stage,
            model_type=args.model
        )
        
        print("\n" + "="*50)
        print("🎉 Pipeline completed successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
