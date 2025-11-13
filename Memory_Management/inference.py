import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import deque
import random

class MLPageReplacementSystem:
    """
    Real-time ML-based page replacement system with visualization
    """
    def __init__(self, model_path='best_page_replacement_model.pkl', memory_size=20, num_pages=100):
        self.memory_size = memory_size
        self.num_pages = num_pages
        self.memory = []
        self.access_history = deque(maxlen=1000)
        self.page_faults = 0
        self.total_accesses = 0
        
        # Load model
        self.load_model(model_path)
        
        # Statistics
        self.stats = {
            'page_faults': [],
            'hit_rate': [],
            'victim_predictions': [],
            'comparison': {'ML': [], 'LRU': [], 'FIFO': [], 'Optimal': []}
        }
        
    def load_model(self, model_path):
        """
        Load trained model and preprocessing components
        """
        print(f"Loading model from {model_path}...")
        
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        self.model = model_package['model']
        self.model_name = model_package['model_name']
        self.feature_columns = model_package['feature_columns']
        self.scaler = model_package['scaler']
        
        print(f"Loaded {self.model_name} model successfully!")
    
    def extract_features_for_page(self, page_id, memory_page):
        """
        Extract features for a single page in memory
        """
        # Calculate recency
        last_access_idx = -1
        for i in range(len(self.access_history) - 1, -1, -1):
            if self.access_history[i] == memory_page:
                last_access_idx = i
                break
        
        recency = len(self.access_history) - last_access_idx if last_access_idx != -1 else len(self.access_history)
        
        # Calculate frequency
        frequency = list(self.access_history).count(memory_page)
        
        # Time in memory
        time_in_memory = min(100, list(self.access_history[-100:]).count(memory_page))
        
        # Distance from current access
        page_distance = abs(memory_page - page_id)
        
        # Recent access patterns
        recent_10 = 1 if memory_page in list(self.access_history)[-10:] else 0
        recent_50 = 1 if memory_page in list(self.access_history)[-50:] else 0
        recent_100 = 1 if memory_page in list(self.access_history)[-100:] else 0
        
        # Base features
        features = {
            'recency': recency,
            'frequency': frequency,
            'time_in_memory': time_in_memory,
            'page_distance': page_distance,
            'recent_10': recent_10,
            'recent_50': recent_50,
            'recent_100': recent_100
        }
        
        # Derived features (matching preprocessing)
        features['recency_frequency_ratio'] = recency / (frequency + 1)
        features['time_frequency_ratio'] = time_in_memory / (frequency + 1)
        features['log_recency'] = np.log1p(recency)
        features['log_frequency'] = np.log1p(frequency)
        features['recency_distance'] = recency * page_distance
        features['freq_recent_10'] = frequency * recent_10
        features['priority_score'] = (
            frequency * 0.3 + 
            (100 - recency) * 0.3 + 
            recent_10 * 20 +
            recent_50 * 10 +
            recent_100 * 5
        )
        
        return features
    
    def predict_victim(self, incoming_page):
        """
        Use ML model to predict which page to evict
        """
        if len(self.memory) < self.memory_size:
            return None
        
        # Extract features for all pages in memory
        features_list = []
        for mem_page in self.memory:
            features = self.extract_features_for_page(incoming_page, mem_page)
            features_list.append(features)
        
        # Create DataFrame and ensure correct feature order
        df = pd.DataFrame(features_list)
        df = df[self.feature_columns]
        
        # Normalize features
        features_normalized = self.scaler.transform(df)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(features_normalized)[:, 1]
        
        # Return page with highest probability of being victim
        victim_idx = np.argmax(probabilities)
        return self.memory[victim_idx], probabilities[victim_idx]
    
    def access_page(self, page_id):
        """
        Access a page with ML-based replacement
        """
        self.total_accesses += 1
        self.access_history.append(page_id)
        
        # Check if page is in memory
        if page_id in self.memory:
            # Page hit
            return 'HIT', None
        else:
            # Page fault
            self.page_faults += 1
            
            victim = None
            if len(self.memory) >= self.memory_size:
                # Need to evict - use ML prediction
                victim, confidence = self.predict_victim(page_id)
                self.memory.remove(victim)
                self.stats['victim_predictions'].append({
                    'victim': victim,
                    'confidence': confidence,
                    'incoming': page_id
                })
            
            self.memory.append(page_id)
            return 'FAULT', victim
    
    def lru_replacement(self, memory_copy, access_history_copy, page_id):
        """
        Least Recently Used replacement for comparison
        """
        if page_id in memory_copy:
            return 'HIT', None
        
        if len(memory_copy) >= self.memory_size:
            # Find LRU page
            lru_page = None
            lru_time = float('inf')
            
            for page in memory_copy:
                try:
                    last_access = len(access_history_copy) - 1 - list(access_history_copy)[::-1].index(page)
                except ValueError:
                    last_access = -1
                
                if last_access < lru_time:
                    lru_time = last_access
                    lru_page = page
            
            memory_copy.remove(lru_page)
            memory_copy.append(page_id)
            return 'FAULT', lru_page
        
        memory_copy.append(page_id)
        return 'FAULT', None
    
    def fifo_replacement(self, memory_copy, page_id):
        """
        First In First Out replacement for comparison
        """
        if page_id in memory_copy:
            return 'HIT', None
        
        victim = None
        if len(memory_copy) >= self.memory_size:
            victim = memory_copy.pop(0)
        
        memory_copy.append(page_id)
        return 'FAULT', victim
    
    def simulate_and_compare(self, access_pattern, verbose=True):
        """
        Simulate page accesses and compare with traditional algorithms
        """
        # Initialize comparison memories
        lru_memory = []
        fifo_memory = []
        lru_access_history = deque(maxlen=1000)
        
        ml_faults = 0
        lru_faults = 0
        fifo_faults = 0
        
        if verbose:
            print("="*60)
            print("SIMULATION STARTED")
            print("="*60)
        
        for i, page_id in enumerate(access_pattern):
            # ML-based replacement
            result, victim = self.access_page(page_id)
            if result == 'FAULT':
                ml_faults += 1
            
            # LRU replacement
            lru_access_history.append(page_id)
            lru_result, _ = self.lru_replacement(lru_memory, lru_access_history, page_id)
            if lru_result == 'FAULT':
                lru_faults += 1
            
            # FIFO replacement
            fifo_result, _ = self.fifo_replacement(fifo_memory, page_id)
            if fifo_result == 'FAULT':
                fifo_faults += 1
            
            # Update statistics
            if (i + 1) % 100 == 0:
                ml_rate = 1 - (ml_faults / (i + 1))
                lru_rate = 1 - (lru_faults / (i + 1))
                fifo_rate = 1 - (fifo_faults / (i + 1))
                
                self.stats['comparison']['ML'].append(ml_rate)
                self.stats['comparison']['LRU'].append(lru_rate)
                self.stats['comparison']['FIFO'].append(fifo_rate)
                
                if verbose and (i + 1) % 500 == 0:
                    print(f"\nProgress: {i+1}/{len(access_pattern)}")
                    print(f"ML Hit Rate: {ml_rate*100:.2f}%")
                    print(f"LRU Hit Rate: {lru_rate*100:.2f}%")
                    print(f"FIFO Hit Rate: {fifo_rate*100:.2f}%")
        
        if verbose:
            print("\n" + "="*60)
            print("SIMULATION COMPLETE")
            print("="*60)
            print(f"\nFinal Results:")
            print(f"ML Page Faults: {ml_faults} (Hit Rate: {(1-ml_faults/len(access_pattern))*100:.2f}%)")
            print(f"LRU Page Faults: {lru_faults} (Hit Rate: {(1-lru_faults/len(access_pattern))*100:.2f}%)")
            print(f"FIFO Page Faults: {fifo_faults} (Hit Rate: {(1-fifo_faults/len(access_pattern))*100:.2f}%)")
            
            improvement_lru = ((lru_faults - ml_faults) / lru_faults) * 100
            improvement_fifo = ((fifo_faults - ml_faults) / fifo_faults) * 100
            
            print(f"\nML Improvement over LRU: {improvement_lru:.2f}%")
            print(f"ML Improvement over FIFO: {improvement_fifo:.2f}%")
        
        return ml_faults, lru_faults, fifo_faults
    
    def visualize_memory_state(self, save_path='memory_state.png'):
        """
        Visualize current memory state
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Memory layout visualization
        rows = 4
        cols = self.memory_size // rows
        
        for i, page_id in enumerate(self.memory):
            row = i // cols
            col = i % cols
            
            # Color based on recency
            recent_count = list(self.access_history[-100:]).count(page_id)
            color_intensity = min(1.0, recent_count / 10)
            color = plt.cm.RdYlGn(color_intensity)
            
            rect = Rectangle((col, rows - row - 1), 1, 1, 
                           facecolor=color, edgecolor='black', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(col + 0.5, rows - row - 0.5, str(page_id),
                    ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax1.set_xlim(0, cols)
        ax1.set_ylim(0, rows)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Memory State (Size: {self.memory_size})\nColor: Recent Access Frequency', 
                     fontsize=14, fontweight='bold')
        
        # Hit rate comparison
        if len(self.stats['comparison']['ML']) > 0:
            x = np.arange(len(self.stats['comparison']['ML'])) * 100
            ax2.plot(x, np.array(self.stats['comparison']['ML']) * 100, 
                    label='ML-Based', linewidth=2, marker='o', markersize=4)
            ax2.plot(x, np.array(self.stats['comparison']['LRU']) * 100, 
                    label='LRU', linewidth=2, marker='s', markersize=4)
            ax2.plot(x, np.array(self.stats['comparison']['FIFO']) * 100, 
                    label='FIFO', linewidth=2, marker='^', markersize=4)
            
            ax2.set_xlabel('Number of Accesses', fontsize=12)
            ax2.set_ylabel('Hit Rate (%)', fontsize=12)
            ax2.set_title('Page Replacement Performance Comparison', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_access_pattern(self, num_accesses=5000):
        """
        Generate realistic page access pattern with locality
        """
        pattern = []
        current_page = random.randint(0, self.num_pages - 1)
        
        for _ in range(num_accesses):
            rand = random.random()
            
            if rand < 0.7 and len(pattern) > 0:  # Temporal locality
                recent_pages = pattern[-50:]
                current_page = random.choice(recent_pages)
            elif rand < 0.9:  # Spatial locality
                offset = random.randint(-5, 5)
                current_page = max(0, min(self.num_pages - 1, current_page + offset))
            else:  # Random access
                current_page = random.randint(0, self.num_pages - 1)
            
            pattern.append(current_page)
        
        return pattern

if __name__ == "__main__":
    # Create ML page replacement system
    print("Initializing ML Page Replacement System...")
    system = MLPageReplacementSystem(
        model_path='best_page_replacement_model.pkl',
        memory_size=20,
        num_pages=100
    )
    
    # Generate access pattern
    print("\nGenerating page access pattern...")
    access_pattern = system.generate_access_pattern(num_accesses=5000)
    
    # Run simulation and comparison
    print("\nRunning simulation...\n")
    ml_faults, lru_faults, fifo_faults = system.simulate_and_compare(access_pattern)
    
    # Visualize results
    print("\nGenerating visualization...")
    system.visualize_memory_state(save_path='memory_management_results.png')
    
    print("\nVisualization saved to 'memory_management_results.png'")
    print("\nSystem ready for real-time inference!")