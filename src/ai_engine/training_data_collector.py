# src/ai_engine/training_data_collector.py
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from collections import deque

class TrainingDataCollector:
    def __init__(self, process_monitor=None, max_samples: int = 1000):
        self.monitor = process_monitor
        self.training_dataset = deque(maxlen=max_samples)
        self.sample_count = 0
        
    def collect_labeled_example(self, process_metrics: Dict, system_metrics: Dict, optimal_decisions: List[Dict]):
        """Collect one labeled training example"""
        features = self._extract_features(process_metrics, system_metrics)
        labels = self._extract_labels(optimal_decisions)
        
        training_example = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'labels': labels,
            'context': {
                'system_load': system_metrics.get('cpu_percent', 0),
                'process_count': len(process_metrics),
                'time_of_day': datetime.now().hour,
                'day_of_week': datetime.now().weekday(),
                'sample_id': self.sample_count
            }
        }
        
        self.training_dataset.append(training_example)
        self.sample_count += 1
        
        if self.sample_count % 50 == 0:
            print(f"📊 Collected {self.sample_count} training samples")
    
    def _extract_features(self, process_metrics: Dict, system_metrics: Dict) -> Dict[str, Any]:
        """Extract features for ML model training"""
        features = {}
        
        # System-level features
        features.update({
            'system_cpu_percent': system_metrics.get('cpu_percent', 0),
            'system_ram_percent': system_metrics.get('ram_percent', 0),
            'system_temperature': system_metrics.get('cpu_temperature', 0) or 0,
            'battery_percent': system_metrics.get('battery_percent', 100) or 100,
            'battery_plugged': 1 if system_metrics.get('battery_plugged', False) else 0,
            'load_average': system_metrics.get('load_avg', 0),
            'total_processes': len(process_metrics),
            'running_processes': system_metrics.get('running_processes', 0),
            'hour_of_day': datetime.now().hour,
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
            'is_night': 1 if 22 <= datetime.now().hour <= 6 else 0
        })
        
        # Process-level features (aggregated statistics)
        if process_metrics:
            cpu_usages = [p.get('cpu_percent', 0) for p in process_metrics.values()]
            memory_usages = [p.get('memory_percent', 0) for p in process_metrics.values()]
            priorities = [p.get('nice', 0) for p in process_metrics.values()]
            threads = [p.get('num_threads', 1) for p in process_metrics.values()]
            
            features.update({
                'avg_cpu_usage': sum(cpu_usages) / len(cpu_usages),
                'max_cpu_usage': max(cpu_usages) if cpu_usages else 0,
                'std_cpu_usage': (sum((x - (sum(cpu_usages)/len(cpu_usages)))**2 for x in cpu_usages) / len(cpu_usages))**0.5 if cpu_usages else 0,
                'avg_memory_usage': sum(memory_usages) / len(memory_usages),
                'max_memory_usage': max(memory_usages) if memory_usages else 0,
                'high_priority_count': sum(1 for p in priorities if p < 0),
                'low_priority_count': sum(1 for p in priorities if p > 0),
                'cpu_intensive_count': sum(1 for c in cpu_usages if c > 30),
                'memory_intensive_count': sum(1 for m in memory_usages if m > 10),
                'total_threads': sum(threads),
                'avg_threads_per_process': sum(threads) / len(threads) if threads else 0
            })
        else:
            # Default values if no processes
            features.update({
                'avg_cpu_usage': 0, 'max_cpu_usage': 0, 'std_cpu_usage': 0,
                'avg_memory_usage': 0, 'max_memory_usage': 0,
                'high_priority_count': 0, 'low_priority_count': 0,
                'cpu_intensive_count': 0, 'memory_intensive_count': 0,
                'total_threads': 0, 'avg_threads_per_process': 0
            })
        
        return features
    
    def _extract_labels(self, optimal_decisions: List[Dict]) -> Dict[str, Any]:
        """Extract labels from expert decisions for ML training"""
        labels = {}
        
        # For each process decision, create labeled features
        for decision in optimal_decisions:
            pid = decision['pid']
            prefix = f"proc_{pid}"
            
            labels.update({
                f"{prefix}_quantum": decision.get('optimal_quantum_ms', 20),
                f"{prefix}_priority": decision.get('optimal_priority', 0.5),
                f"{prefix}_type": self._process_type_to_number(decision.get('process_type', 'background'))
            })
        
        # Also store aggregated labels for system-wide optimization
        if optimal_decisions:
            quantums = [d.get('optimal_quantum_ms', 20) for d in optimal_decisions]
            priorities = [d.get('optimal_priority', 0.5) for d in optimal_decisions]
            
            labels.update({
                'system_avg_quantum': sum(quantums) / len(quantums),
                'system_avg_priority': sum(priorities) / len(priorities),
                'system_quantum_std': (sum((x - (sum(quantums)/len(quantums)))**2 for x in quantums) / len(quantums))**0.5,
                'interactive_ratio': sum(1 for p in priorities if p > 0.7) / len(priorities) if priorities else 0
            })
        
        return labels
    
    def _process_type_to_number(self, process_type: str) -> int:
        """Convert process type to numerical label"""
        type_mapping = {
            'web_browser': 0,
            'developer_tool': 1,
            'gaming': 2,
            'cpu_intensive': 3,
            'database': 4,
            'ide': 5,
            'media_player': 6,
            'high_priority_system': 7,
            'cpu_bound': 8,
            'memory_intensive': 9,
            'parallel_worker': 10,
            'background': 11
        }
        return type_mapping.get(process_type, 11)
    
    def export_training_data(self, filename: str = "data/training_data.json"):
        """Export collected training data to JSON file"""
        try:
            with open(filename, 'w') as f:
                # Convert deque to list for JSON serialization
                json.dump(list(self.training_dataset), f, indent=2)
            print(f"💾 Training data exported to {filename} ({len(self.training_dataset)} samples)")
        except Exception as e:
            print(f"❌ Error exporting training data: {e}")
    
    def load_training_data(self, filename: str = "data/training_data.json"):
        """Load existing training data from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.training_dataset = deque(data, maxlen=self.training_dataset.maxlen)
                self.sample_count = len(data)
            print(f"📂 Training data loaded from {filename} ({len(self.training_dataset)} samples)")
        except FileNotFoundError:
            print(f"📂 No existing training data found at {filename}")
            self.training_dataset = deque(maxlen=self.training_dataset.maxlen)
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the collected training data"""
        if not self.training_dataset:
            return {"total_samples": 0}
        
        total_samples = len(self.training_dataset)
        
        # Calculate some basic stats
        system_loads = [sample['context']['system_load'] for sample in self.training_dataset]
        process_counts = [sample['context']['process_count'] for sample in self.training_dataset]
        
        return {
            "total_samples": total_samples,
            "avg_system_load": sum(system_loads) / len(system_loads),
            "avg_process_count": sum(process_counts) / len(process_counts),
            "data_collection_period": f"{total_samples * 2 / 3600:.1f} hours",
            "first_sample": self.training_dataset[0]['timestamp'] if total_samples > 0 else None,
            "last_sample": self.training_dataset[-1]['timestamp'] if total_samples > 0 else None
        }
    
    def clear_data(self):
        """Clear all collected training data"""
        self.training_dataset.clear()
        self.sample_count = 0
        print("🗑️ Training data cleared")

# Quick test function
def test_data_collector():
    """Test the training data collector"""
    collector = TrainingDataCollector()
    
    # Mock data
    mock_processes = {
        1234: {'name': 'code', 'cpu_percent': 15.0, 'memory_percent': 8.0, 'nice': 0, 'num_threads': 10},
        5678: {'name': 'chrome', 'cpu_percent': 25.0, 'memory_percent': 12.0, 'nice': 0, 'num_threads': 15}
    }
    
    mock_system = {
        'cpu_percent': 45.0, 'ram_percent': 65.0, 'cpu_temperature': 62.0,
        'battery_percent': 85, 'battery_plugged': True, 'load_avg': 1.5,
        'running_processes': 2
    }
    
    mock_decisions = [
        {'pid': 1234, 'optimal_quantum_ms': 18, 'optimal_priority': 0.85, 'process_type': 'developer_tool'},
        {'pid': 5678, 'optimal_quantum_ms': 15, 'optimal_priority': 0.8, 'process_type': 'web_browser'}
    ]
    
    # Collect sample
    collector.collect_labeled_example(mock_processes, mock_system, mock_decisions)
    
    # Show stats
    stats = collector.get_dataset_stats()
    print("📊 Data Collector Test:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export to test
    collector.export_training_data("test_training_data.json")
    
    return collector

if __name__ == "__main__":
    test_data_collector()