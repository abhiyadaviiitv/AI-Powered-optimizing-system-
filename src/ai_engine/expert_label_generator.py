# src/ai_engine/expert_label_generator.py
import psutil
from typing import Dict, List, Any
import time

class ExpertLabelGenerator:
    def __init__(self):
        self.expert_rules = self._load_expert_rules()
    
    def generate_optimal_decisions(self, process_metrics: Dict, system_metrics: Dict) -> List[Dict]:
        """Generate optimal scheduling decisions using expert rules"""
        decisions = []
        
        for pid, metrics in process_metrics.items():
            process_type = self._classify_process(metrics)
            optimal_quantum = self._calculate_optimal_quantum(process_type, metrics, system_metrics)
            optimal_priority = self._calculate_optimal_priority(process_type, metrics, system_metrics)
            optimal_cores = self._calculate_optimal_cores(process_type, system_metrics)
            
            decisions.append({
                'pid': pid,
                'process_name': metrics.get('name', 'unknown'),
                'process_type': process_type,
                'optimal_quantum_ms': optimal_quantum,
                'optimal_priority': optimal_priority,
                'recommended_cores': optimal_cores,
                'reasoning': self._explain_decision(process_type, optimal_quantum, system_metrics),
                'timestamp': time.time()
            })
        
        return decisions
    
    def _classify_process(self, metrics: Dict) -> str:
        """Expert process classification based on behavior and name"""
        name = metrics.get('name', '').lower()
        cpu_usage = metrics.get('cpu_percent', 0)
        memory_usage = metrics.get('memory_percent', 0)
        nice_value = metrics.get('nice', 0)
        num_threads = metrics.get('num_threads', 1)
        
        # Expert classification rules
        if any(term in name for term in ['chrome', 'firefox', 'brave', 'safari', 'browser']):
            return 'web_browser'
        elif any(term in name for term in ['code', 'pycharm', 'vim', 'nano', 'sublime', 'atom', 'editor']):
            return 'developer_tool'
        elif any(term in name for term in ['game', 'steam', 'wine', 'lutris', 'heroic']):
            return 'gaming'
        elif any(term in name for term in ['ffmpeg', 'render', 'blender', 'compile', 'make', 'gcc', 'g++']):
            return 'cpu_intensive'
        elif any(term in name for term in ['mysql', 'postgres', 'redis', 'database', 'mongodb']):
            return 'database'
        elif any(term in name for term in ['pycharm', 'intellij', 'webstorm', 'clion']):
            return 'ide'
        elif any(term in name for term in ['spotify', 'vlc', 'music', 'video', 'audio']):
            return 'media_player'
        elif nice_value < 0:
            return 'high_priority_system'
        elif cpu_usage > 50:
            return 'cpu_bound'
        elif memory_usage > 20:
            return 'memory_intensive'
        elif num_threads > 10:
            return 'parallel_worker'
        else:
            return 'background'
    
    def _calculate_optimal_quantum(self, process_type: str, metrics: Dict, system_metrics: Dict) -> int:
        """Calculate optimal time quantum based on process type and system state"""
        # Base quanta for different process types (in milliseconds)
        base_quanta = {
            'web_browser': 15,
            'developer_tool': 18,
            'gaming': 12,
            'cpu_intensive': 45,
            'database': 25,
            'ide': 20,
            'media_player': 22,
            'high_priority_system': 10,
            'cpu_bound': 40,
            'memory_intensive': 30,
            'parallel_worker': 35,
            'background': 60
        }
        
        quantum = base_quanta.get(process_type, 20)
        
        # Adjust based on system conditions
        system_load = system_metrics.get('cpu_percent', 0)
        if system_load > 80:
            quantum = int(quantum * 0.7)  # Shorter quanta under high load
        elif system_load < 30:
            quantum = int(quantum * 1.3)  # Longer quanta under low load
            
        # Adjust based on battery level
        battery = system_metrics.get('battery_percent', 100)
        if battery and battery < 20:
            quantum = int(quantum * 1.2)  # Longer quanta to save power
            
        # Adjust based on temperature
        temperature = system_metrics.get('cpu_temperature', 0)
        if temperature and temperature > 75:
            quantum = int(quantum * 0.9)  # Slightly shorter to reduce heat
            
        # Ensure quantum stays in reasonable bounds (10-100ms)
        return max(10, min(100, quantum))
    
    def _calculate_optimal_priority(self, process_type: str, metrics: Dict, system_metrics: Dict) -> float:
        """Calculate optimal priority score (0.0 to 1.0)"""
        # Base priorities for different process types
        base_priority = {
            'web_browser': 0.8,
            'developer_tool': 0.85,
            'gaming': 0.9,
            'cpu_intensive': 0.5,
            'database': 0.7,
            'ide': 0.88,
            'media_player': 0.6,
            'high_priority_system': 1.0,
            'cpu_bound': 0.6,
            'memory_intensive': 0.4,
            'parallel_worker': 0.55,
            'background': 0.2
        }
        
        priority = base_priority.get(process_type, 0.5)
        
        # Adjust based on system load
        system_load = system_metrics.get('cpu_percent', 0)
        if system_load > 70 and process_type in ['web_browser', 'developer_tool', 'gaming']:
            priority = min(0.95, priority + 0.1)  # Boost interactive tasks under load
            
        # Adjust based on user activity (simplified - would use mouse/keyboard events)
        hour = time.localtime().tm_hour
        if 9 <= hour <= 18:  # Working hours
            if process_type in ['web_browser', 'developer_tool', 'ide']:
                priority = min(0.95, priority + 0.05)
        
        return max(0.1, min(0.95, priority))
    
    def _calculate_optimal_cores(self, process_type: str, system_metrics: Dict) -> List[int]:
        """Calculate optimal CPU core affinity"""
        total_cores = system_metrics.get('cpu_logical_cores', psutil.cpu_count(logical=True) or 4)
        
        core_strategies = {
            'web_browser': list(range(min(2, total_cores))),  # Prefer first 2 cores
            'developer_tool': list(range(min(2, total_cores))),
            'gaming': list(range(min(4, total_cores))),  # Gaming benefits from more cores
            'cpu_intensive': list(range(total_cores)),  # Use all cores
            'database': list(range(total_cores)),
            'ide': list(range(min(2, total_cores))),
            'media_player': list(range(min(2, total_cores))),
            'high_priority_system': [0],  # Dedicated core for system tasks
            'cpu_bound': list(range(total_cores)),
            'memory_intensive': list(range(max(1, total_cores // 2))),  # Half cores
            'parallel_worker': list(range(total_cores)),
            'background': list(range(max(1, total_cores // 2)))
        }
        
        return core_strategies.get(process_type, list(range(min(2, total_cores))))
    
    def _explain_decision(self, process_type: str, quantum: int, system_metrics: Dict) -> str:
        """Generate human-readable explanation for the decision"""
        explanations = {
            'web_browser': f"Quantum {quantum}ms for responsive browsing and smooth scrolling",
            'developer_tool': f"Quantum {quantum}ms for instant code feedback and typing response",
            'gaming': f"Quantum {quantum}ms for stable frame rates and minimal input lag",
            'cpu_intensive': f"Quantum {quantum}ms for efficient long-running computations",
            'database': f"Quantum {quantum}ms for balanced query processing and I/O operations",
            'ide': f"Quantum {quantum}ms for smooth editing and fast code intelligence",
            'media_player': f"Quantum {quantum}ms for uninterrupted audio/video playback",
            'high_priority_system': f"Quantum {quantum}ms for critical system responsiveness",
            'background': f"Quantum {quantum}ms for efficient background task completion"
        }
        
        # Add system context to explanation
        system_load = system_metrics.get('cpu_percent', 0)
        battery = system_metrics.get('battery_percent', 100)
        
        context = []
        if system_load > 80:
            context.append("high system load")
        if battery and battery < 30:
            context.append("low battery")
        
        base_explanation = explanations.get(process_type, f"Quantum {quantum}ms for optimal performance")
        
        if context:
            return f"{base_explanation} (adjusted for {', '.join(context)})"
        else:
            return base_explanation
    
    def _load_expert_rules(self) -> Dict:
        """Load expert scheduling rules (could be from config file)"""
        return {
            "interactive_boost": 0.1,
            "power_save_multiplier": 1.2,
            "thermal_throttle_multiplier": 0.9,
            "high_load_multiplier": 0.7,
            "min_quantum": 10,
            "max_quantum": 100
        }

# Quick test function
def test_expert_system():
    """Test the expert label generator"""
    expert = ExpertLabelGenerator()
    
    # Create mock data
    mock_process = {
        'name': 'code',
        'cpu_percent': 15.5,
        'memory_percent': 8.2,
        'nice': 0,
        'num_threads': 12
    }
    
    mock_system = {
        'cpu_percent': 45.0,
        'ram_percent': 65.0,
        'cpu_temperature': 62.0,
        'battery_percent': 85,
        'cpu_logical_cores': 8
    }
    
    decisions = expert.generate_optimal_decisions(
        {1234: mock_process}, 
        mock_system
    )
    
    print("🧠 Expert System Test:")
    for decision in decisions:
        print(f"PID {decision['pid']}: {decision['process_name']}")
        print(f"  Type: {decision['process_type']}")
        print(f"  Quantum: {decision['optimal_quantum_ms']}ms")
        print(f"  Priority: {decision['optimal_priority']:.2f}")
        print(f"  Cores: {decision['recommended_cores']}")
        print(f"  Reason: {decision['reasoning']}")
    
    return decisions

if __name__ == "__main__":
    test_expert_system()