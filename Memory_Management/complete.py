import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import deque
import random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

class EnhancedAdaptiveMLPageReplacement:
    """
    Enhanced Adaptive ML with advanced features for better accuracy
    """
    def __init__(self, memory_size=20, num_pages=100, adaptation_rate=30):
        self.memory_size = memory_size
        self.num_pages = num_pages
        self.memory = []
        self.access_history = deque(maxlen=2000)  # Increased history
        self.page_faults = 0
        self.total_accesses = 0
        self.adaptation_rate = adaptation_rate
        
        # Enhanced training buffer with priority sampling
        self.training_buffer = deque(maxlen=1000)  # Larger buffer
        self.retrain_counter = 0
        
        # Track page access patterns per page
        self.page_stats = {}  # Per-page statistics
        self.access_intervals = {}  # Time between accesses
        
        # Enhanced model - Gradient Boosting performs better
        self.model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=10,
            subsample=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Extended feature set
        self.feature_columns = [
            'recency', 'frequency', 'time_in_memory', 'page_distance',
            'recent_10', 'recent_50', 'recent_100', 'recent_200',
            'recency_frequency_ratio', 'time_frequency_ratio',
            'log_recency', 'log_frequency', 'recency_distance',
            'freq_recent_10', 'priority_score',
            'access_interval_mean', 'access_interval_std', 'last_interval',
            'forward_distance', 'backward_distance', 'position_in_memory',
            'frequency_trend', 'recency_normalized', 'frequency_normalized',
            'working_set_membership', 'cyclic_pattern_score'
        ]
        
        # Statistics
        self.stats = {
            'page_faults': [],
            'hit_rate': [],
            'retraining_points': [],
            'comparison': {'ML': [], 'LRU': [], 'FIFO': []},
            'confidence_scores': []
        }
        
        print(f"Initialized Enhanced Adaptive ML (retrains every {adaptation_rate} accesses)")
        print(f"Using Gradient Boosting with {len(self.feature_columns)} features")
    
    def update_page_statistics(self, page_id):
        """Track detailed per-page statistics"""
        current_time = self.total_accesses
        
        if page_id not in self.page_stats:
            self.page_stats[page_id] = {
                'first_access': current_time,
                'last_access': current_time,
                'access_count': 1,
                'access_times': [current_time]
            }
            self.access_intervals[page_id] = []
        else:
            stats = self.page_stats[page_id]
            interval = current_time - stats['last_access']
            self.access_intervals[page_id].append(interval)
            
            stats['last_access'] = current_time
            stats['access_count'] += 1
            stats['access_times'].append(current_time)
            
            # Keep only recent access times
            if len(stats['access_times']) > 50:
                stats['access_times'] = stats['access_times'][-50:]
    
    def extract_enhanced_features(self, page_id, memory_page):
        """Extract comprehensive feature set"""
        history_list = list(self.access_history)
        current_time = self.total_accesses
        
        # Basic features
        last_access_idx = -1
        for i in range(len(history_list) - 1, -1, -1):
            if history_list[i] == memory_page:
                last_access_idx = i
                break
        
        recency = len(history_list) - last_access_idx if last_access_idx != -1 else len(history_list)
        frequency = history_list.count(memory_page)
        time_in_memory = min(200, history_list[-200:].count(memory_page))
        page_distance = abs(memory_page - page_id)
        
        # Recent access patterns (extended)
        recent_10 = 1 if memory_page in history_list[-10:] else 0
        recent_50 = 1 if memory_page in history_list[-50:] else 0
        recent_100 = 1 if memory_page in history_list[-100:] else 0
        recent_200 = 1 if memory_page in history_list[-200:] else 0
        
        # Access interval statistics
        intervals = self.access_intervals.get(memory_page, [0])
        access_interval_mean = np.mean(intervals) if intervals else 0
        access_interval_std = np.std(intervals) if len(intervals) > 1 else 0
        last_interval = intervals[-1] if intervals else 0
        
        # Forward and backward distance in page space
        forward_distance = (page_id - memory_page) if page_id > memory_page else self.num_pages
        backward_distance = (memory_page - page_id) if memory_page > page_id else self.num_pages
        
        # Position in memory (recently added pages are at end)
        try:
            position_in_memory = self.memory.index(memory_page) / self.memory_size
        except ValueError:
            position_in_memory = 0.5
        
        # Frequency trend (recent vs overall)
        recent_freq = history_list[-100:].count(memory_page)
        frequency_trend = recent_freq / (frequency + 1)
        
        # Normalized features
        max_recency = len(history_list) if len(history_list) > 0 else 1
        recency_normalized = recency / max_recency
        
        max_frequency = max([history_list.count(p) for p in set(self.memory)]) if self.memory else 1
        frequency_normalized = frequency / (max_frequency + 1)
        
        # Working set membership (frequently accessed pages)
        working_set_pages = [p for p in history_list[-100:] if history_list[-100:].count(p) >= 3]
        working_set_membership = 1 if memory_page in working_set_pages else 0
        
        # Cyclic pattern detection
        if memory_page in self.page_stats and len(self.page_stats[memory_page]['access_times']) > 3:
            access_times = self.page_stats[memory_page]['access_times']
            intervals_list = [access_times[i+1] - access_times[i] for i in range(len(access_times)-1)]
            if len(intervals_list) >= 3:
                # Check for regularity in intervals
                mean_interval = np.mean(intervals_list)
                std_interval = np.std(intervals_list)
                cyclic_pattern_score = 1 - min(1.0, std_interval / (mean_interval + 1))
            else:
                cyclic_pattern_score = 0
        else:
            cyclic_pattern_score = 0
        
        # Combine all features
        features = {
            'recency': recency,
            'frequency': frequency,
            'time_in_memory': time_in_memory,
            'page_distance': page_distance,
            'recent_10': recent_10,
            'recent_50': recent_50,
            'recent_100': recent_100,
            'recent_200': recent_200,
            'recency_frequency_ratio': recency / (frequency + 1),
            'time_frequency_ratio': time_in_memory / (frequency + 1),
            'log_recency': np.log1p(recency),
            'log_frequency': np.log1p(frequency),
            'recency_distance': recency * page_distance,
            'freq_recent_10': frequency * recent_10,
            'priority_score': (
                frequency * 0.3 + 
                (100 - min(100, recency)) * 0.3 + 
                recent_10 * 20 + recent_50 * 10 + recent_100 * 5
            ),
            'access_interval_mean': access_interval_mean,
            'access_interval_std': access_interval_std,
            'last_interval': last_interval,
            'forward_distance': forward_distance,
            'backward_distance': backward_distance,
            'position_in_memory': position_in_memory,
            'frequency_trend': frequency_trend,
            'recency_normalized': recency_normalized,
            'frequency_normalized': frequency_normalized,
            'working_set_membership': working_set_membership,
            'cyclic_pattern_score': cyclic_pattern_score
        }
        
        return features
    
    def find_optimal_victim(self, incoming_page, memory):
        """Enhanced optimal victim selection using multiple heuristics"""
        scores = {}
        history_list = list(self.access_history)
        
        for page in memory:
            score = 0
            
            # Recency score (higher = worse)
            if page in history_list:
                last_idx = len(history_list) - 1 - history_list[::-1].index(page)
                recency_score = (len(history_list) - last_idx) * 2
            else:
                recency_score = 1000
            
            # Frequency score (lower = worse)
            freq = history_list[-200:].count(page)
            frequency_score = -freq * 10
            
            # Working set score
            if page not in history_list[-50:]:
                working_set_score = 200
            else:
                working_set_score = 0
            
            # Distance score
            distance_score = abs(page - incoming_page) * 0.5
            
            # Combine scores (higher = better victim)
            scores[page] = recency_score + frequency_score + working_set_score + distance_score
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def predict_victim_with_confidence(self, incoming_page):
        """Predict with confidence-based fallback"""
        if len(self.memory) < self.memory_size:
            return None, 0.0
        
        # Confidence threshold - use ML only if confident enough
        confidence_threshold = 0.55
        
        if not self.model_trained or len(self.training_buffer) < 100:
            return self.lru_victim(), 0.5
        
        try:
            features_list = []
            for mem_page in self.memory:
                features = self.extract_enhanced_features(incoming_page, mem_page)
                features_list.append(features)
            
            df = pd.DataFrame(features_list)
            df = df[self.feature_columns]
            
            features_normalized = self.scaler.transform(df)
            probabilities = self.model.predict_proba(features_normalized)[:, 1]
            
            victim_idx = np.argmax(probabilities)
            confidence = probabilities[victim_idx]
            
            # If not confident, blend with LRU
            if confidence < confidence_threshold:
                lru_page = self.lru_victim()
                # Return LRU if ML is uncertain
                return lru_page, confidence
            
            return self.memory[victim_idx], confidence
        
        except Exception as e:
            return self.lru_victim(), 0.5
    
    def lru_victim(self):
        """Enhanced LRU with tie-breaking"""
        history_list = list(self.access_history)
        candidates = []
        
        for page in self.memory:
            try:
                last_access = len(history_list) - 1 - history_list[::-1].index(page)
                freq = history_list[-100:].count(page)
                candidates.append((page, last_access, freq))
            except ValueError:
                return page  # Never accessed, definitely evict
        
        # Sort by last access, then by frequency
        candidates.sort(key=lambda x: (x[1], -x[2]))
        return candidates[0][0]
    
    def collect_training_sample(self, incoming_page, victim_page):
        """Collect high-quality training samples"""
        if len(self.memory) < self.memory_size:
            return
        
        optimal_victim = self.find_optimal_victim(incoming_page, self.memory + [victim_page])
        
        for mem_page in self.memory + [victim_page]:
            features = self.extract_enhanced_features(incoming_page, mem_page)
            features['is_victim'] = 1 if mem_page == optimal_victim else 0
            
            # Priority sampling - add important samples multiple times
            if features['is_victim'] == 1:
                self.training_buffer.append(features)
            self.training_buffer.append(features)
    
    def retrain_model(self):
        """Enhanced retraining with data balancing"""
        if len(self.training_buffer) < 100:
            return
        
        try:
            df = pd.DataFrame(list(self.training_buffer))
            
            X = df[self.feature_columns]
            y = df['is_victim']
            
            if len(y.unique()) < 2:
                return
            
            # Balance dataset
            victim_samples = df[df['is_victim'] == 1]
            non_victim_samples = df[df['is_victim'] == 0]
            
            if len(non_victim_samples) > len(victim_samples) * 1.5:
                non_victim_samples = non_victim_samples.sample(n=int(len(victim_samples) * 1.5))
                df = pd.concat([victim_samples, non_victim_samples]).sample(frac=1.0)
                X = df[self.feature_columns]
                y = df['is_victim']
            
            if not self.model_trained:
                self.scaler.fit(X)
            
            X_normalized = self.scaler.transform(X)
            self.model.fit(X_normalized, y)
            self.model_trained = True
            
            self.stats['retraining_points'].append(self.total_accesses)
            
        except Exception as e:
            print(f"Retraining error: {e}")
    
    def access_page(self, page_id):
        """Access page with enhanced tracking"""
        self.total_accesses += 1
        self.retrain_counter += 1
        
        # Update statistics
        self.update_page_statistics(page_id)
        self.access_history.append(page_id)
        
        if page_id in self.memory:
            return 'HIT', None
        else:
            self.page_faults += 1
            
            victim = None
            confidence = 0.0
            
            if len(self.memory) >= self.memory_size:
                victim, confidence = self.predict_victim_with_confidence(page_id)
                self.collect_training_sample(page_id, victim)
                self.memory.remove(victim)
                self.stats['confidence_scores'].append(confidence)
            
            self.memory.append(page_id)
            
            # Adaptive retraining
            if self.retrain_counter >= self.adaptation_rate:
                self.retrain_model()
                self.retrain_counter = 0
            
            return 'FAULT', victim
    
    def lru_replacement(self, memory_copy, access_history_copy, page_id):
        """LRU for comparison"""
        if page_id in memory_copy:
            return 'HIT', None
        
        if len(memory_copy) >= self.memory_size:
            history_list = list(access_history_copy)
            lru_page = None
            lru_time = float('inf')
            
            for page in memory_copy:
                try:
                    last_access = len(history_list) - 1 - history_list[::-1].index(page)
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
        """FIFO for comparison"""
        if page_id in memory_copy:
            return 'HIT', None
        
        victim = None
        if len(memory_copy) >= self.memory_size:
            victim = memory_copy.pop(0)
        
        memory_copy.append(page_id)
        return 'FAULT', victim
    
    def simulate_and_compare(self, access_pattern, verbose=True):
        """Run enhanced simulation"""
        lru_memory = []
        fifo_memory = []
        lru_history = deque(maxlen=2000)
        
        ml_faults = 0
        lru_faults = 0
        fifo_faults = 0
        
        if verbose:
            print("="*60)
            print("ENHANCED ADAPTIVE ML SIMULATION")
            print("="*60)
        
        for i, page_id in enumerate(access_pattern):
            result, victim = self.access_page(page_id)
            if result == 'FAULT':
                ml_faults += 1
            
            lru_history.append(page_id)
            lru_result, _ = self.lru_replacement(lru_memory, lru_history, page_id)
            if lru_result == 'FAULT':
                lru_faults += 1
            
            fifo_result, _ = self.fifo_replacement(fifo_memory, page_id)
            if fifo_result == 'FAULT':
                fifo_faults += 1
            
            if (i + 1) % 100 == 0:
                ml_rate = 1 - (ml_faults / (i + 1))
                lru_rate = 1 - (lru_faults / (i + 1))
                fifo_rate = 1 - (fifo_faults / (i + 1))
                
                self.stats['comparison']['ML'].append(ml_rate)
                self.stats['comparison']['LRU'].append(lru_rate)
                self.stats['comparison']['FIFO'].append(fifo_rate)
                
                if verbose and (i + 1) % 500 == 0:
                    status = "✓ Trained" if self.model_trained else "⚠ Learning"
                    avg_conf = np.mean(self.stats['confidence_scores'][-50:]) if self.stats['confidence_scores'] else 0
                    print(f"\nProgress: {i+1}/{len(access_pattern)} [{status}] Confidence: {avg_conf:.2f}")
                    print(f"ML: {ml_rate*100:.2f}% | LRU: {lru_rate*100:.2f}% | FIFO: {fifo_rate*100:.2f}%")
        
        if verbose:
            print("\n" + "="*60)
            print("SIMULATION COMPLETE")
            print("="*60)
            print(f"\nFinal Results:")
            print(f"ML: {ml_faults} faults ({(1-ml_faults/len(access_pattern))*100:.2f}% hit rate)")
            print(f"LRU: {lru_faults} faults ({(1-lru_faults/len(access_pattern))*100:.2f}% hit rate)")
            print(f"FIFO: {fifo_faults} faults ({(1-fifo_faults/len(access_pattern))*100:.2f}% hit rate)")
            print(f"\nRetrains: {len(self.stats['retraining_points'])}")
        
        return ml_faults, lru_faults, fifo_faults
    
    def visualize_results(self, save_path='enhanced_ml_results.png'):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Memory state
        ax1 = fig.add_subplot(gs[0, 0])
        rows, cols = 4, self.memory_size // 4
        history_list = list(self.access_history)
        
        for i, page_id in enumerate(self.memory):
            row, col = i // cols, i % cols
            recent_count = history_list[-100:].count(page_id)
            color_intensity = min(1.0, recent_count / 10)
            color = plt.cm.RdYlGn(color_intensity)
            
            rect = Rectangle((col, rows - row - 1), 1, 1, 
                           facecolor=color, edgecolor='black', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(col + 0.5, rows - row - 0.5, str(page_id),
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax1.set_xlim(0, cols)
        ax1.set_ylim(0, rows)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title('Memory State', fontweight='bold')
        
        # Performance comparison
        ax2 = fig.add_subplot(gs[0, 1:])
        x = np.arange(len(self.stats['comparison']['ML'])) * 100
        ax2.plot(x, np.array(self.stats['comparison']['ML']) * 100, 
                label='Enhanced ML', linewidth=2.5, color='#2ecc71')
        ax2.plot(x, np.array(self.stats['comparison']['LRU']) * 100, 
                label='LRU', linewidth=2, color='#3498db')
        ax2.plot(x, np.array(self.stats['comparison']['FIFO']) * 100, 
                label='FIFO', linewidth=2, color='#e74c3c')
        
        for pt in self.stats['retraining_points']:
            ax2.axvline(x=pt, color='orange', linestyle='--', alpha=0.2, linewidth=1)
        
        ax2.set_xlabel('Accesses', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Hit Rate (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Performance Over Time', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Hit rate bars
        ax3 = fig.add_subplot(gs[1, 0])
        ml_final = self.stats['comparison']['ML'][-1] * 100
        lru_final = self.stats['comparison']['LRU'][-1] * 100
        fifo_final = self.stats['comparison']['FIFO'][-1] * 100
        
        bars = ax3.bar(['Enhanced ML', 'LRU', 'FIFO'], 
                      [ml_final, lru_final, fifo_final],
                      color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Hit Rate (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Final Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Confidence over time
        ax4 = fig.add_subplot(gs[1, 1])
        if self.stats['confidence_scores']:
            window = 50
            conf = self.stats['confidence_scores']
            smoothed = np.convolve(conf, np.ones(window)/window, mode='valid')
            ax4.plot(smoothed, linewidth=2, color='purple')
            ax4.fill_between(range(len(smoothed)), smoothed, alpha=0.3, color='purple')
            ax4.axhline(y=0.55, color='red', linestyle='--', label='Confidence Threshold')
            ax4.set_xlabel('Decisions', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Confidence', fontsize=11, fontweight='bold')
            ax4.set_title('Model Confidence', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Statistics
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        improvement_lru = ((ml_final - lru_final) / lru_final * 100) if lru_final > 0 else 0
        improvement_fifo = ((ml_final - fifo_final) / fifo_final * 100) if fifo_final > 0 else 0
        
        stats_text = f"""
ENHANCED ML STATISTICS
{'='*28}

Accesses: {self.total_accesses:,}
Retrains: {len(self.stats['retraining_points'])}
Features: {len(self.feature_columns)}

Hit Rates:
ML:   {ml_final:.2f}%
LRU:  {lru_final:.2f}%
FIFO: {fifo_final:.2f}%

Improvement:
vs LRU:  {improvement_lru:+.2f}%
vs FIFO: {improvement_fifo:+.2f}%

Avg Confidence: {np.mean(self.stats['confidence_scores']) if self.stats['confidence_scores'] else 0:.2f}
        """
        
        ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightgreen', alpha=0.5))
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {save_path}")
        plt.show()
    
    def generate_access_pattern(self, num_accesses=5000):
        """Generate realistic pattern"""
        pattern = []
        current_page = random.randint(0, self.num_pages - 1)
        
        for _ in range(num_accesses):
            rand = random.random()
            
            if rand < 0.7 and len(pattern) > 0:
                recent_pages = pattern[-50:]
                current_page = random.choice(recent_pages)
            elif rand < 0.9:
                offset = random.randint(-5, 5)
                current_page = max(0, min(self.num_pages - 1, current_page + offset))
            else:
                current_page = random.randint(0, self.num_pages - 1)
            
            pattern.append(current_page)
        
        return pattern


if __name__ == "__main__":
    print("="*60)
    print("ENHANCED ADAPTIVE ML PAGE REPLACEMENT")
    print("="*60)
    
    system = EnhancedAdaptiveMLPageReplacement(
        memory_size=20,
        num_pages=100,
        adaptation_rate=30  # More frequent retraining
    )
    
    print("\nGenerating access pattern...")
    access_pattern = system.generate_access_pattern(num_accesses=5000)
    
    print("\nStarting enhanced simulation...\n")
    ml_faults, lru_faults, fifo_faults = system.simulate_and_compare(access_pattern)
    
    print("\nGenerating visualization...")
    system.visualize_results(save_path='enhanced_ml_results.png')
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)