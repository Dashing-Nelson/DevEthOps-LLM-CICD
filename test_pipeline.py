#!/usr/bin/env python3
"""
Test script to verify the DevEthOps pipeline works with the IBM HR dataset.
This script runs a simplified version of the pipeline to check if everything is working.
"""

import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_data_loading():
    """Test if the dataset can be loaded correctly."""
    print("🔍 Testing data loading...")
    
    try:
        from devethops.data_loader import DataLoader
        from devethops.config import ConfigManager
        
        # Load configuration
        config = ConfigManager('configs/settings.yaml')
        
        # Initialize data loader
        data_loader = DataLoader(config)
        
        # Load IBM HR dataset
        features, target = data_loader.load_ibm_hr_data()
        
        print(f"✅ Data loaded successfully!")
        print(f"   - Features shape: {features.shape}")
        print(f"   - Target shape: {target.shape}")
        print(f"   - Feature columns: {list(features.columns)}")
        print(f"   - Target values: {target.value_counts().to_dict()}")
        
        # Combine for preprocessing
        data = features.copy()
        data['Attrition'] = target
        
        return True, data
        
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return False, None

def test_preprocessing(data):
    """Test preprocessing pipeline."""
    print("\n🔧 Testing preprocessing...")
    
    try:
        from devethops.preprocess import DataPreprocessor
        from devethops.config import ConfigManager
        
        config = ConfigManager('configs/settings.yaml')
        preprocessor = DataPreprocessor(config)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(data, 'Attrition')
        
        print(f"✅ Preprocessing completed!")
        print(f"   - Training set: {X_train.shape}")
        print(f"   - Test set: {X_test.shape}")
        print(f"   - Features: {X_train.shape[1]}")
        
        return True, (X_train, X_test, y_train, y_test)
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        return False, None

def test_model_training(data_splits):
    """Test model training."""
    print("\n🤖 Testing model training...")
    
    try:
        from devethops.models_tabular import TabularModelTrainer
        from devethops.config import ConfigManager
        
        config = ConfigManager('configs/settings.yaml')
        trainer = TabularModelTrainer(config)
        
        X_train, X_test, y_train, y_test = data_splits
        
        # Train a simple logistic regression model
        training_results = trainer.train_logistic_regression(X_train, y_train, hyperparameter_search=False)
        model = training_results['model_object']
        
        # Make predictions
        predictions = model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        
        print(f"✅ Model training completed!")
        print(f"   - Model type: {training_results['model_type']}")
        print(f"   - Test accuracy: {accuracy:.3f}")
        print(f"   - Number of features: {len(training_results.get('feature_names', []))}")
        
        return True, model
        
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        return False, None

def test_fairness_check(data, model, data_splits):
    """Test fairness evaluation."""
    print("\n⚖️ Testing fairness checks...")
    
    try:
        from devethops.fairness_checks import FairnessEvaluator
        from devethops.config import ConfigManager
        
        config = ConfigManager('configs/settings.yaml')
        evaluator = FairnessEvaluator(config)
        
        X_train, X_test, y_train, y_test = data_splits
        
        # Prepare data with predictions
        test_data = X_test.copy()
        test_data['Attrition'] = y_test
        test_data['predictions'] = model.predict(X_test)
        
        # Run fairness evaluation (focusing on Gender)
        if 'Gender' in test_data.columns:
            fairness_results = evaluator.evaluate_fairness(
                test_data, 
                'Gender', 
                'Attrition', 
                'predictions'
            )
            
            print(f"✅ Fairness evaluation completed!")
            print(f"   - Protected attribute: Gender")
            print(f"   - Metrics computed: {list(fairness_results.keys())}")
            
            # Check if any thresholds are violated
            violations = evaluator.check_fairness_thresholds(fairness_results, 'ibm_hr')
            if violations:
                print(f"   - ⚠️ Fairness violations detected: {len(violations)}")
            else:
                print(f"   - ✅ No fairness violations detected")
        else:
            print(f"   - ⚠️ Gender column not found, skipping detailed fairness check")
        
        return True
        
    except Exception as e:
        print(f"❌ Fairness evaluation failed: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🚀 DevEthOps Pipeline Test")
    print("=" * 50)
    
    # Test 1: Data Loading
    success, data = test_data_loading()
    if not success:
        print("\n❌ Pipeline test failed at data loading stage")
        return False
    
    # Test 2: Preprocessing
    success, data_splits = test_preprocessing(data)
    if not success:
        print("\n❌ Pipeline test failed at preprocessing stage")
        return False
    
    # Test 3: Model Training
    success, model = test_model_training(data_splits)
    if not success:
        print("\n❌ Pipeline test failed at model training stage")
        return False
    
    # Test 4: Fairness Check
    success = test_fairness_check(data, model, data_splits)
    if not success:
        print("\n❌ Pipeline test failed at fairness evaluation stage")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! The DevEthOps pipeline is working correctly.")
    print("\nNext steps:")
    print("1. Install required dependencies: pip install -r requirements.txt")
    print("2. Run full pipeline: python scripts/run_pipeline.py")
    print("3. Check configs/settings.yaml for configuration options")
    
    return True

if __name__ == "__main__":
    main()
