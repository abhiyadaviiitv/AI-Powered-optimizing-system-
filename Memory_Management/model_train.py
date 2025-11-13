import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PageReplacementModel:
    """
    Trains ML models for page replacement prediction
    """
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        self.scaler = None
        
    def load_preprocessed_data(self, prefix='preprocessed'):
        """
        Load preprocessed data from files
        """
        print("Loading preprocessed data...")
        
        X_train = np.load(f'{prefix}_X_train.npy')
        X_val = np.load(f'{prefix}_X_val.npy')
        X_test = np.load(f'{prefix}_X_test.npy')
        y_train = np.load(f'{prefix}_y_train.npy')
        y_val = np.load(f'{prefix}_y_val.npy')
        y_test = np.load(f'{prefix}_y_test.npy')
        
        with open(f'{prefix}_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f'{prefix}_feature_columns.pkl', 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        print(f"Loaded data - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def initialize_models(self):
        """
        Initialize different ML models for comparison
        """
        print("Initializing models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=7,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }
        
        print(f"Initialized {len(self.models)} models")
    
    def train_models(self, X_train, y_train):
        """
        Train all models
        """
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"{name} training complete")
        
        return trained_models
    
    def evaluate_model(self, model, X, y, dataset_name=''):
        """
        Evaluate a single model
        """
        y_pred = model.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }
        
        return metrics, y_pred
    
    def evaluate_all_models(self, trained_models, X_val, y_val, X_test, y_test):
        """
        Evaluate all trained models
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        results = {}
        best_f1 = 0
        
        for name, model in trained_models.items():
            print(f"\nEvaluating {name}...")
            
            # Validation metrics
            val_metrics, val_pred = self.evaluate_model(model, X_val, y_val, 'Validation')
            
            # Test metrics
            test_metrics, test_pred = self.evaluate_model(model, X_test, y_test, 'Test')
            
            results[name] = {
                'model': model,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'val_pred': val_pred,
                'test_pred': test_pred
            }
            
            print(f"\nValidation Metrics:")
            print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f}")
            print(f"  Recall:    {val_metrics['recall']:.4f}")
            print(f"  F1 Score:  {val_metrics['f1']:.4f}")
            
            print(f"\nTest Metrics:")
            print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall:    {test_metrics['recall']:.4f}")
            print(f"  F1 Score:  {test_metrics['f1']:.4f}")
            
            # Track best model based on validation F1 score
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.best_model = model
                self.best_model_name = name
        
        print("\n" + "="*60)
        print(f"BEST MODEL: {self.best_model_name} (Validation F1: {best_f1:.4f})")
        print("="*60)
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path=None):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Victim', 'Victim'],
                   yticklabels=['Not Victim', 'Victim'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results, save_path=None):
        """
        Plot comparison of all models
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            val_scores = [results[name]['val_metrics'][metric] for name in model_names]
            test_scores = [results[name]['test_metrics'][metric] for name in model_names]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            axes[idx].bar(x - width/2, val_scores, width, label='Validation', alpha=0.8)
            axes[idx].bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(f'{metric.capitalize()} Comparison')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
            axes[idx].legend()
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model, model_name, top_n=15, save_path=None):
        """
        Plot feature importance for tree-based models
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(top_n), importances[indices])
            plt.yticks(range(top_n), [self.feature_columns[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importances - {model_name}')
            plt.gca().invert_yaxis()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_best_model(self, output_path='best_page_replacement_model.pkl'):
        """
        Save the best model
        """
        print(f"\nSaving best model ({self.best_model_name})...")
        
        model_package = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"Model saved to {output_path}")
    
    def train_pipeline(self, data_prefix='preprocessed', save_model=True):
        """
        Complete training pipeline
        """
        print("="*60)
        print("ML PAGE REPLACEMENT MODEL TRAINING")
        print("="*60)
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_preprocessed_data(data_prefix)
        
        # Initialize models
        self.initialize_models()
        
        # Train models
        trained_models = self.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_all_models(trained_models, X_val, y_val, X_test, y_test)
        
        # Visualizations
        print("\nGenerating visualizations...")
        self.plot_model_comparison(results, save_path='model_comparison.png')
        
        # Plot confusion matrix for best model
        self.plot_confusion_matrix(
            y_test, 
            results[self.best_model_name]['test_pred'],
            self.best_model_name,
            save_path='confusion_matrix_best.png'
        )
        
        # Plot feature importance for best model
        # if hasattr(self.best_model, 'feature_importances_'):
        #     self.plot_feature_importance(
        #         self.best_model,
        #         self.best_model_name,
        #         save_path='feature_importance_best.png'
        #     )
        
        # Save best model
        if save_model:
            self.save_best_model()
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETE!")
        print("="*60)
        
        return results

if __name__ == "__main__":
    # Create trainer
    trainer = PageReplacementModel()
    
    # Run training pipeline
    results = trainer.train_pipeline(
        data_prefix='preprocessed',
        save_model=True
    )