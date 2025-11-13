import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

class DataPreprocessor:
    """
    Preprocesses page access data for ML model training
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, file_path='page_access_data.csv'):
        """
        Load raw data from CSV
        """
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records")
        return df
    
    def engineer_features(self, df):
        """
        Create additional features from raw data
        """
        print("Engineering features...")
        
        # Create derived features
        df['recency_frequency_ratio'] = df['recency'] / (df['frequency'] + 1)
        df['time_frequency_ratio'] = df['time_in_memory'] / (df['frequency'] + 1)
        
        # Logarithmic transformations for skewed features
        df['log_recency'] = np.log1p(df['recency'])
        df['log_frequency'] = np.log1p(df['frequency'])
        
        # Interaction features
        df['recency_distance'] = df['recency'] * df['page_distance']
        df['freq_recent_10'] = df['frequency'] * df['recent_10']
        
        # Priority score (lower is better for eviction)
        df['priority_score'] = (
            df['frequency'] * 0.3 + 
            (100 - df['recency']) * 0.3 + 
            df['recent_10'] * 20 +
            df['recent_50'] * 10 +
            df['recent_100'] * 5
        )
        
        print(f"Created {len(df.columns)} total features")
        return df
    
    def handle_missing_values(self, df):
        """
        Handle any missing values in the dataset
        """
        print("Checking for missing values...")
        missing = df.isnull().sum()
        
        if missing.sum() > 0:
            print(f"Found {missing.sum()} missing values")
            # Fill missing values with median
            df = df.fillna(df.median())
        else:
            print("No missing values found")
        
        return df
    
    def remove_outliers(self, df, columns=None, threshold=3):
        """
        Remove outliers using z-score method
        """
        if columns is None:
            columns = ['recency', 'frequency', 'page_distance']
        
        print(f"Removing outliers from {columns}...")
        initial_size = len(df)
        
        for col in columns:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        
        print(f"Removed {initial_size - len(df)} outliers ({(initial_size - len(df))/initial_size*100:.2f}%)")
        return df
    
    def balance_dataset(self, df):
        """
        Balance the dataset if victim/non-victim classes are imbalanced
        """
        print("Checking class balance...")
        
        victim_count = df['is_victim'].sum()
        non_victim_count = len(df) - victim_count
        
        print(f"Victim samples: {victim_count}")
        print(f"Non-victim samples: {non_victim_count}")
        print(f"Imbalance ratio: {non_victim_count/victim_count:.2f}:1")
        
        # If highly imbalanced, undersample majority class
        if non_victim_count / victim_count > 3:
            print("Balancing dataset...")
            victim_df = df[df['is_victim'] == 1]
            non_victim_df = df[df['is_victim'] == 0].sample(n=int(victim_count * 2), random_state=42)
            df = pd.concat([victim_df, non_victim_df]).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"Balanced dataset size: {len(df)}")
        
        return df
    
    def normalize_features(self, df, feature_columns):
        """
        Normalize features using StandardScaler
        """
        print("Normalizing features...")
        
        df_normalized = df.copy()
        df_normalized[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        print("Features normalized")
        return df_normalized
    
    def split_data(self, df, test_size=0.2, val_size=0.1):
        """
        Split data into train, validation, and test sets
        """
        print(f"Splitting data (train: {1-test_size-val_size:.0%}, val: {val_size:.0%}, test: {test_size:.0%})...")
        
        # Define feature columns (exclude target and metadata)
        self.feature_columns = [col for col in df.columns 
                               if col not in ['is_victim', 'page_id', 'timestamp', 'incoming_page']]
        
        X = df[self.feature_columns]
        y = df['is_victim']
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                               output_prefix='preprocessed'):
        """
        Save preprocessed data and scaler
        """
        print("Saving preprocessed data...")
        
        # Save datasets
        np.save(f'{output_prefix}_X_train.npy', X_train.values)
        np.save(f'{output_prefix}_X_val.npy', X_val.values)
        np.save(f'{output_prefix}_X_test.npy', X_test.values)
        np.save(f'{output_prefix}_y_train.npy', y_train.values)
        np.save(f'{output_prefix}_y_val.npy', y_val.values)
        np.save(f'{output_prefix}_y_test.npy', y_test.values)
        
        # Save scaler and feature columns
        with open(f'{output_prefix}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'{output_prefix}_feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print("Preprocessed data saved successfully!")
        print(f"Feature columns: {self.feature_columns}")
    
    def preprocess_pipeline(self, input_file='page_access_data.csv', 
                          output_prefix='preprocessed'):
        """
        Complete preprocessing pipeline
        """
        print("="*60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load data
        df = self.load_data(input_file)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Remove outliers
        df = self.remove_outliers(df)
        
        # Balance dataset
        df = self.balance_dataset(df)
        
        # Define feature columns
        feature_columns = [col for col in df.columns 
                          if col not in ['is_victim', 'page_id', 'timestamp', 'incoming_page']]
        
        # Normalize features
        df = self.normalize_features(df, feature_columns)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        
        # Save preprocessed data
        self.save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, output_prefix)
        
        print("="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    # Create preprocessor
    preprocessor = DataPreprocessor()
    
    # Run preprocessing pipeline
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline(
        input_file='page_access_data.csv',
        output_prefix='preprocessed'
    )
    
    print("\nPreprocessing Summary:")
    print(f"Training features shape: {X_train.shape}")
    print(f"Training labels distribution: {np.bincount(y_train.values.astype(int))}")
    print(f"\nFeature columns: {preprocessor.feature_columns}")