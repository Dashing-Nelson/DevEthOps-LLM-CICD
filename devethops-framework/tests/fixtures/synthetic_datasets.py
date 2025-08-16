"""
Test fixtures for DevEthOps framework.
Provides synthetic datasets with known bias patterns for testing.
"""

import numpy as np
import pandas as pd
import json
import tempfile
import os
from typing import Dict, Tuple, List
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class BiasInjector:
    """Utility class for injecting various types of bias into datasets."""
    
    @staticmethod
    def inject_selection_bias(data: pd.DataFrame, 
                             protected_attr: str, 
                             privileged_group: str,
                             bias_strength: float = 0.3) -> pd.DataFrame:
        """
        Inject selection bias by underrepresenting certain groups.
        
        Args:
            data: Original dataset
            protected_attr: Protected attribute column name
            privileged_group: Value representing privileged group
            bias_strength: Strength of bias (0-1)
        
        Returns:
            Biased dataset
        """
        biased_data = data.copy()
        
        # Reduce representation of non-privileged groups
        mask_privileged = biased_data[protected_attr] == privileged_group
        mask_unprivileged = ~mask_privileged
        
        # Keep all privileged group samples
        privileged_samples = biased_data[mask_privileged]
        
        # Reduce unprivileged group samples
        unprivileged_samples = biased_data[mask_unprivileged]
        n_keep = int(len(unprivileged_samples) * (1 - bias_strength))
        unprivileged_reduced = unprivileged_samples.sample(n=n_keep, random_state=42)
        
        return pd.concat([privileged_samples, unprivileged_reduced], ignore_index=True)
    
    @staticmethod
    def inject_label_bias(data: pd.DataFrame, 
                         protected_attr: str,
                         privileged_group: str,
                         bias_strength: float = 0.3) -> pd.DataFrame:
        """
        Inject label bias by systematically mislabeling certain groups.
        
        Args:
            data: Original dataset with 'label' column
            protected_attr: Protected attribute column name  
            privileged_group: Value representing privileged group
            bias_strength: Strength of bias (0-1)
            
        Returns:
            Dataset with biased labels
        """
        biased_data = data.copy()
        
        # Flip labels for unprivileged group with certain probability
        mask_unprivileged = biased_data[protected_attr] != privileged_group
        flip_mask = np.random.random(mask_unprivileged.sum()) < bias_strength
        
        unprivileged_indices = biased_data[mask_unprivileged].index
        flip_indices = unprivileged_indices[flip_mask]
        
        # Flip positive labels to negative (systematic disadvantage)
        positive_flip_mask = biased_data.loc[flip_indices, 'label'] == 1
        biased_data.loc[flip_indices[positive_flip_mask], 'label'] = 0
        
        return biased_data
    
    @staticmethod
    def inject_feature_bias(data: pd.DataFrame,
                           protected_attr: str,
                           feature_cols: List[str],
                           privileged_group: str,
                           bias_strength: float = 0.3) -> pd.DataFrame:
        """
        Inject feature bias by modifying feature distributions for certain groups.
        
        Args:
            data: Original dataset
            protected_attr: Protected attribute column name
            feature_cols: List of feature columns to bias
            privileged_group: Value representing privileged group  
            bias_strength: Strength of bias (0-1)
            
        Returns:
            Dataset with biased features
        """
        biased_data = data.copy()
        
        mask_unprivileged = biased_data[protected_attr] != privileged_group
        
        for col in feature_cols:
            if col in biased_data.columns:
                # Reduce feature values for unprivileged group
                reduction_factor = 1 - bias_strength
                biased_data.loc[mask_unprivileged, col] *= reduction_factor
                
        return biased_data


class SyntheticDataGenerator:
    """Generate synthetic datasets with controlled bias patterns."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_credit_dataset(self, 
                               n_samples: int = 5000,
                               inject_bias: bool = True,
                               bias_types: List[str] = None) -> pd.DataFrame:
        """
        Generate synthetic credit approval dataset.
        
        Args:
            n_samples: Number of samples to generate
            inject_bias: Whether to inject bias
            bias_types: Types of bias to inject ['selection', 'label', 'feature']
            
        Returns:
            Synthetic credit dataset
        """
        if bias_types is None:
            bias_types = ['label', 'feature']
        
        # Generate base features
        X, y = make_classification(
            n_samples=n_samples,
            n_features=8,
            n_informative=6,
            n_redundant=2,
            n_clusters_per_class=2,
            class_sep=0.8,
            random_state=self.random_state
        )
        
        # Create meaningful feature names
        feature_names = [
            'income_score', 'credit_history', 'employment_length', 
            'debt_ratio', 'education_score', 'age_group',
            'savings_score', 'loan_amount'
        ]
        
        data = pd.DataFrame(X, columns=feature_names)
        data['label'] = y  # 1 = approved, 0 = denied
        
        # Add protected attributes
        data['gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
        data['race'] = np.random.choice(
            ['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
            n_samples, 
            p=[0.60, 0.15, 0.15, 0.08, 0.02]
        )
        data['age'] = np.random.randint(18, 75, n_samples)
        data['age_category'] = pd.cut(data['age'], 
                                    bins=[0, 25, 35, 50, 100], 
                                    labels=['Young', 'Adult', 'Middle', 'Senior'])
        
        if inject_bias:
            # Inject various types of bias
            if 'selection' in bias_types:
                data = BiasInjector.inject_selection_bias(
                    data, 'gender', 'Male', bias_strength=0.2
                )
                
            if 'label' in bias_types:
                data = BiasInjector.inject_label_bias(
                    data, 'gender', 'Male', bias_strength=0.25
                )
                data = BiasInjector.inject_label_bias(
                    data, 'race', 'White', bias_strength=0.20
                )
                
            if 'feature' in bias_types:
                data = BiasInjector.inject_feature_bias(
                    data, 'gender', ['income_score', 'credit_history'], 
                    'Male', bias_strength=0.15
                )
                data = BiasInjector.inject_feature_bias(
                    data, 'race', ['income_score', 'education_score'], 
                    'White', bias_strength=0.20
                )
        
        return data
    
    def generate_hiring_dataset(self, 
                               n_samples: int = 3000,
                               inject_bias: bool = True) -> pd.DataFrame:
        """
        Generate synthetic hiring dataset.
        
        Args:
            n_samples: Number of samples to generate
            inject_bias: Whether to inject bias
            
        Returns:
            Synthetic hiring dataset
        """
        # Generate base features
        X, y = make_classification(
            n_samples=n_samples,
            n_features=6,
            n_informative=5,
            n_redundant=1,
            n_clusters_per_class=2,
            class_sep=0.7,
            random_state=self.random_state
        )
        
        feature_names = [
            'experience_years', 'education_level', 'skill_score',
            'interview_score', 'portfolio_quality', 'referral_strength'
        ]
        
        data = pd.DataFrame(X, columns=feature_names)
        data['label'] = y  # 1 = hired, 0 = not hired
        
        # Add protected attributes
        data['gender'] = np.random.choice(['Male', 'Female', 'Non-binary'], 
                                        n_samples, p=[0.52, 0.46, 0.02])
        data['ethnicity'] = np.random.choice(
            ['White', 'Black', 'Hispanic', 'Asian', 'Native', 'Other'],
            n_samples,
            p=[0.58, 0.16, 0.18, 0.06, 0.01, 0.01]
        )
        data['age'] = np.random.randint(22, 65, n_samples)
        
        if inject_bias:
            # Inject hiring bias
            data = BiasInjector.inject_label_bias(
                data, 'gender', 'Male', bias_strength=0.30
            )
            data = BiasInjector.inject_feature_bias(
                data, 'gender', ['interview_score', 'skill_score'], 
                'Male', bias_strength=0.20
            )
            data = BiasInjector.inject_label_bias(
                data, 'ethnicity', 'White', bias_strength=0.25
            )
        
        return data
    
    def generate_healthcare_dataset(self, 
                                   n_samples: int = 4000,
                                   inject_bias: bool = True) -> pd.DataFrame:
        """
        Generate synthetic healthcare dataset.
        
        Args:
            n_samples: Number of samples to generate
            inject_bias: Whether to inject bias
            
        Returns:
            Synthetic healthcare dataset
        """
        # Generate base features
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=2,
            class_sep=0.6,
            random_state=self.random_state
        )
        
        feature_names = [
            'symptom_severity', 'vital_signs_score', 'lab_results',
            'medical_history', 'age_factor', 'bmi_score',
            'medication_response', 'lifestyle_score', 'family_history',
            'emergency_indicator'
        ]
        
        data = pd.DataFrame(X, columns=feature_names)
        data['label'] = y  # 1 = high-priority treatment, 0 = standard care
        
        # Add protected attributes
        data['gender'] = np.random.choice(['Male', 'Female'], n_samples)
        data['race'] = np.random.choice(
            ['White', 'Black', 'Hispanic', 'Asian', 'Native', 'Other'],
            n_samples,
            p=[0.62, 0.14, 0.16, 0.06, 0.01, 0.01]
        )
        data['age'] = np.random.randint(18, 90, n_samples)
        data['insurance_type'] = np.random.choice(
            ['Private', 'Medicare', 'Medicaid', 'Uninsured'],
            n_samples,
            p=[0.60, 0.20, 0.15, 0.05]
        )
        
        if inject_bias:
            # Inject healthcare bias
            data = BiasInjector.inject_label_bias(
                data, 'race', 'White', bias_strength=0.20
            )
            data = BiasInjector.inject_feature_bias(
                data, 'insurance_type', ['symptom_severity', 'vital_signs_score'],
                'Private', bias_strength=0.25
            )
            data = BiasInjector.inject_label_bias(
                data, 'gender', 'Male', bias_strength=0.15
            )
        
        return data


def create_test_datasets() -> Dict[str, str]:
    """
    Create all test datasets and return their file paths.
    
    Returns:
        Dictionary mapping dataset names to file paths
    """
    generator = SyntheticDataGenerator(random_state=42)
    dataset_paths = {}
    
    # Generate datasets
    datasets = {
        'credit_biased': generator.generate_credit_dataset(
            n_samples=2000, inject_bias=True, bias_types=['label', 'feature']
        ),
        'credit_fair': generator.generate_credit_dataset(
            n_samples=2000, inject_bias=False
        ),
        'hiring_biased': generator.generate_hiring_dataset(
            n_samples=1500, inject_bias=True
        ),
        'hiring_fair': generator.generate_hiring_dataset(
            n_samples=1500, inject_bias=False
        ),
        'healthcare_biased': generator.generate_healthcare_dataset(
            n_samples=1800, inject_bias=True
        ),
        'healthcare_fair': generator.generate_healthcare_dataset(
            n_samples=1800, inject_bias=False
        )
    }
    
    # Save datasets to temporary files
    temp_dir = tempfile.mkdtemp()
    
    for name, data in datasets.items():
        file_path = os.path.join(temp_dir, f"{name}.csv")
        data.to_csv(file_path, index=False)
        dataset_paths[name] = file_path
    
    # Create dataset metadata
    metadata = {
        'credit': {
            'protected_attributes': ['gender', 'race', 'age_category'],
            'label_column': 'label',
            'feature_columns': [
                'income_score', 'credit_history', 'employment_length',
                'debt_ratio', 'education_score', 'age_group',
                'savings_score', 'loan_amount'
            ],
            'favorable_label': 1,
            'description': 'Credit approval dataset with potential bias in lending decisions'
        },
        'hiring': {
            'protected_attributes': ['gender', 'ethnicity', 'age'],
            'label_column': 'label', 
            'feature_columns': [
                'experience_years', 'education_level', 'skill_score',
                'interview_score', 'portfolio_quality', 'referral_strength'
            ],
            'favorable_label': 1,
            'description': 'Hiring dataset with potential bias in recruitment decisions'
        },
        'healthcare': {
            'protected_attributes': ['gender', 'race', 'insurance_type'],
            'label_column': 'label',
            'feature_columns': [
                'symptom_severity', 'vital_signs_score', 'lab_results',
                'medical_history', 'age_factor', 'bmi_score',
                'medication_response', 'lifestyle_score', 'family_history',
                'emergency_indicator'
            ],
            'favorable_label': 1,
            'description': 'Healthcare triage dataset with potential bias in treatment prioritization'
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(temp_dir, 'dataset_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    dataset_paths['metadata'] = metadata_path
    
    return dataset_paths


def create_validation_dataset() -> str:
    """
    Create a validation dataset for model testing.
    
    Returns:
        Path to validation dataset file
    """
    generator = SyntheticDataGenerator(random_state=123)
    
    # Create balanced dataset for validation
    data = generator.generate_credit_dataset(
        n_samples=1000, inject_bias=False
    )
    
    # Split into train/validation
    train_data, val_data = train_test_split(
        data, test_size=0.3, random_state=123, stratify=data['label']
    )
    
    # Save validation set
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    val_data.to_csv(temp_file.name, index=False)
    
    return temp_file.name


def create_biased_dataset_for_testing() -> str:
    """
    Create a highly biased dataset for testing bias detection capabilities.
    
    Returns:
        Path to biased dataset file
    """
    generator = SyntheticDataGenerator(random_state=456)
    
    # Create dataset with multiple bias types
    data = generator.generate_credit_dataset(
        n_samples=1500, 
        inject_bias=True, 
        bias_types=['selection', 'label', 'feature']
    )
    
    # Add additional bias patterns
    # Create extreme bias for testing edge cases
    extreme_bias_mask = (data['gender'] == 'Female') & (data['race'] == 'Black')
    data.loc[extreme_bias_mask, 'label'] = 0  # Systematic rejection
    
    # Modify features to amplify bias
    data.loc[extreme_bias_mask, 'income_score'] *= 0.5
    data.loc[extreme_bias_mask, 'credit_history'] *= 0.6
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    data.to_csv(temp_file.name, index=False)
    
    return temp_file.name


if __name__ == '__main__':
    # Create test datasets when run directly
    print("Creating test datasets...")
    
    dataset_paths = create_test_datasets()
    print("Created datasets:")
    for name, path in dataset_paths.items():
        print(f"  {name}: {path}")
    
    print("\nDatasets are ready for testing!")
    print("Note: These are temporary files and will need to be recreated for each test run.")
