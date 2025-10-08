# src/monitoring/metrics_analyzer.py
import numpy as np
from collections import defaultdict, deque
import time

class EnhancedMetricsAnalyzer:
    def __init__(self, process_monitor):
        self.monitor = process_monitor
        self.performance_history = deque(maxlen=100)
        
    def comprehensive_analysis(self):
        """Perform comprehensive system analysis"""
        current_data = self.monitor.get_current_metrics()
        if not current_data:
            return {
                'timestamp': time.time(),
                'system_health': {'overall_health_score': 0, 'load_status': 'unknown'},
                'process_intelligence': {'critical_processes': [], 'process_categories': {}},
                'performance_metrics': {'throughput': 0, 'avg_response_time': 0, 'cpu_efficiency': 0},
                'bottleneck_analysis': [],
                'resource_forecasting': {'cpu_forecast': 'no_data'},
                'scheduling_recommendations': [],
                'time_quanta_analysis': {'efficiency_metrics': {}},
                'energy_efficiency': {'energy_efficiency_score': 0}
            }
        
        system_metrics = current_data['system']
        process_metrics = current_data['processes']
        
        analysis = {
            'timestamp': time.time(),
            'system_health': self._analyze_system_health(system_metrics),
            'process_intelligence': self._analyze_process_intelligence(process_metrics),
            'performance_metrics': self._calculate_performance_metrics(system_metrics, process_metrics),
            'bottleneck_analysis': self._identify_bottlenecks(system_metrics, process_metrics),
            'resource_forecasting': self._forecast_resources(system_metrics),
            'scheduling_recommendations': self._generate_recommendations(system_metrics, process_metrics),
            'time_quanta_analysis': self._analyze_time_quanta_efficiency(process_metrics),
            'energy_efficiency': self._analyze_energy_efficiency(system_metrics)
        }
        
        # Store for historical analysis
        self.performance_history.append(analysis)
        
        return analysis
    
    def _analyze_system_health(self, system_metrics):
        """Analyze overall system health"""
        health = {
            'cpu_usage': system_metrics.get('cpu_percent', 0),
            'ram_usage': system_metrics.get('ram_percent', 0),
            'swap_usage': system_metrics.get('swap_percent', 0),
            'cpu_temperature': system_metrics.get('cpu_temperature', 0),
            'load_status': 'normal',
            'temperature_status': 'normal',
            'memory_status': 'normal',
            'battery_status': 'normal',
            'overall_health_score': 0
        }
        
        # CPU Load Analysis
        cpu_usage = health['cpu_usage']
        if cpu_usage > 90:
            health['load_status'] = 'critical'
        elif cpu_usage > 70:
            health['load_status'] = 'high'
        elif cpu_usage > 50:
            health['load_status'] = 'moderate'
        else:
            health['load_status'] = 'normal'
        
        # Temperature Analysis
        temp = health['cpu_temperature']
        if temp:
            if temp > 85:
                health['temperature_status'] = 'critical'
            elif temp > 75:
                health['temperature_status'] = 'high'
            elif temp > 65:
                health['temperature_status'] = 'elevated'
            else:
                health['temperature_status'] = 'normal'
        
        # Memory Analysis
        ram_usage = health['ram_usage']
        swap_usage = health['swap_usage']
        if ram_usage > 90 or swap_usage > 50:
            health['memory_status'] = 'critical'
        elif ram_usage > 80:
            health['memory_status'] = 'high'
        elif ram_usage > 70:
            health['memory_status'] = 'moderate'
        else:
            health['memory_status'] = 'normal'
        
        # Battery Analysis
        battery = system_metrics.get('battery_percent', 100)
        if battery and battery < 10:
            health['battery_status'] = 'critical'
        elif battery and battery < 20:
            health['battery_status'] = 'low'
        elif battery and battery < 50:
            health['battery_status'] = 'moderate'
        else:
            health['battery_status'] = 'normal'
        
        # Calculate overall health score (0-100)
        health_score = 100
        health_score -= min(30, cpu_usage * 0.3)
        health_score -= min(20, ram_usage * 0.2)
        health_score -= min(20, (temp or 0) / 4)
        if battery and battery < 20:
            health_score -= 10
        
        health['overall_health_score'] = max(0, min(100, health_score))
        
        return health
    
    def _analyze_process_intelligence(self, process_metrics):
        """Intelligent process analysis"""
        intelligence = {
            'critical_processes': [],
            'resource_efficiency': [],
            'behavioral_patterns': [],
            'process_categories': defaultdict(list),
            'suspicious_activity': []
        }
        
        for pid, metrics in process_metrics.items():
            # Critical process detection
            criticality_score = self._calculate_criticality_score(metrics)
            if criticality_score > 0.6:
                intelligence['critical_processes'].append({
                    'pid': pid,
                    'name': metrics.get('name', 'unknown'),
                    'criticality_score': criticality_score,
                    'cpu_usage': metrics.get('cpu_percent', 0),
                    'memory_usage': metrics.get('memory_percent', 0),
                    'priority': metrics.get('nice', 0),
                    'reasons': self._get_criticality_reasons(metrics)
                })
            
            # Categorize processes
            category = self._categorize_process(metrics)
            intelligence['process_categories'][category].append(pid)
            
            # Behavioral patterns
            pattern = self._classify_behavioral_pattern(metrics)
            intelligence['behavioral_patterns'].append({
                'pid': pid,
                'name': metrics.get('name', 'unknown'),
                'pattern_type': pattern
            })
        
        # Sort by severity
        intelligence['critical_processes'].sort(key=lambda x: x['criticality_score'], reverse=True)
        
        return intelligence
    
    def _calculate_criticality_score(self, metrics):
        """Calculate how critical a process is"""
        score = 0
        
        # CPU usage factor
        cpu_usage = metrics.get('cpu_percent', 0)
        score += min(0.4, cpu_usage / 250)
        
        # Memory usage factor
        memory_usage = metrics.get('memory_percent', 0)
        score += min(0.3, memory_usage / 333)
        
        # I/O activity factor
        io_activity = (metrics.get('io_read_bytes', 0) + metrics.get('io_write_bytes', 0)) / (1024 * 1024)
        score += min(0.2, io_activity / 500)
        
        # Priority factor
        nice = metrics.get('nice', 0)
        if nice < 0:
            score += 0.1
        
        return min(1.0, score)
    
    def _get_criticality_reasons(self, metrics):
        """Get reasons why a process is critical"""
        reasons = []
        
        if metrics.get('cpu_percent', 0) > 30:
            reasons.append("high_cpu")
        if metrics.get('memory_percent', 0) > 10:
            reasons.append("high_memory")
        if metrics.get('nice', 0) < 0:
            reasons.append("high_priority")
        if metrics.get('num_threads', 0) > 20:
            reasons.append("many_threads")
            
        return reasons if reasons else ["normal"]
    
    def _categorize_process(self, metrics):
        """Categorize process by type"""
        cpu_usage = metrics.get('cpu_percent', 0)
        memory_usage = metrics.get('memory_percent', 0)
        io_rate = (metrics.get('io_read_bytes', 0) + metrics.get('io_write_bytes', 0)) / 1024
        name = metrics.get('name', '').lower()
        
        if 'chrome' in name or 'firefox' in name or 'brave' in name:
            return 'browser'
        elif 'code' in name or 'pycharm' in name or 'vim' in name or 'nano' in name:
            return 'developer_tool'
        elif cpu_usage > 50:
            return 'cpu_intensive'
        elif memory_usage > 20:
            return 'memory_intensive'
        elif io_rate > 1000:
            return 'io_intensive'
        elif metrics.get('nice', 0) < 0:
            return 'interactive'
        else:
            return 'background'
    
    def _classify_behavioral_pattern(self, metrics):
        """Classify process behavioral pattern"""
        cpu_usage = metrics.get('cpu_percent', 0)
        memory_usage = metrics.get('memory_percent', 0)
        ctx_switches = metrics.get('num_voluntary_ctx_switches', 0) + metrics.get('num_involuntary_ctx_switches', 0)
        
        if cpu_usage > 50 and memory_usage < 10:
            return "cpu_intensive"
        elif memory_usage > 20 and cpu_usage < 20:
            return "memory_intensive"
        elif ctx_switches > 1000:
            return "interactive"
        elif cpu_usage < 5 and memory_usage < 5:
            return "background"
        else:
            return "mixed_workload"
    
    def _calculate_performance_metrics(self, system_metrics, process_metrics):
        """Calculate performance metrics"""
        total_processes = len(process_metrics)
        running_processes = sum(1 for p in process_metrics.values() if p.get('status') == 'running')
        
        # Estimated metrics
        throughput = running_processes / max(1, total_processes) * 100
        avg_response_time = self._estimate_response_time(process_metrics)
        cpu_efficiency = max(0, 100 - system_metrics.get('cpu_percent', 0))
        
        total_ctx_switches = sum(
            p.get('num_voluntary_ctx_switches', 0) + p.get('num_involuntary_ctx_switches', 0) 
            for p in process_metrics.values()
        )
        
        return {
            'throughput': throughput,
            'avg_response_time': avg_response_time,
            'cpu_efficiency': cpu_efficiency,
            'total_context_switches': total_ctx_switches
        }
    
    def _estimate_response_time(self, process_metrics):
        """Estimate average response time"""
        if not process_metrics:
            return 0
        
        total_response = 0
        for metrics in process_metrics.values():
            priority_factor = 1.0 - (metrics.get('nice', 0) + 20) / 39
            cpu_time = metrics.get('cpu_time_total', 0)
            response_estimate = (1.0 - priority_factor) * 100 + np.log1p(cpu_time) * 10
            total_response += response_estimate
        
        return total_response / len(process_metrics)
    
    def _identify_bottlenecks(self, system_metrics, process_metrics):
        """Identify system bottlenecks"""
        bottlenecks = []
        
        # CPU Bottleneck
        if system_metrics.get('cpu_percent', 0) > 85:
            bottlenecks.append({
                'type': 'cpu_bottleneck',
                'severity': 'high',
                'description': 'CPU usage exceeding 85%',
                'suggested_actions': ['Limit CPU-intensive processes', 'Adjust process priorities']
            })
        
        # Memory Bottleneck
        if system_metrics.get('ram_percent', 0) > 90:
            bottlenecks.append({
                'type': 'memory_bottleneck',
                'severity': 'high',
                'description': 'RAM usage exceeding 90%',
                'suggested_actions': ['Kill memory-intensive processes', 'Optimize memory usage']
            })
        
        return bottlenecks
    
    def _forecast_resources(self, system_metrics):
        """Forecast resource usage"""
        if len(self.performance_history) < 5:
            return {'cpu_forecast': 'insufficient_data'}
        
        recent_cpu = [point['system_health']['cpu_usage'] for point in list(self.performance_history)[-5:]]
        
        if len(recent_cpu) >= 2:
            trend = np.polyfit(range(len(recent_cpu)), recent_cpu, 1)[0]
            
            if trend > 2:
                forecast = 'increasing'
            elif trend < -2:
                forecast = 'decreasing'
            else:
                forecast = 'stable'
            
            return {
                'cpu_forecast': forecast,
                'trend_slope': trend
            }
        
        return {'cpu_forecast': 'stable'}
    
    def _generate_recommendations(self, system_metrics, process_metrics):
        """Generate scheduling recommendations"""
        recommendations = []
        
        # High CPU recommendation
        if system_metrics.get('cpu_percent', 0) > 80:
            high_cpu_procs = [p for p in process_metrics.values() if p.get('cpu_percent', 0) > 20]
            if high_cpu_procs:
                recommendations.append({
                    'action': 'Limit CPU-intensive processes',
                    'processes': [p['name'] for p in high_cpu_procs[:3]]
                })
        
        # High memory recommendation
        if system_metrics.get('ram_percent', 0) > 85:
            high_mem_procs = [p for p in process_metrics.values() if p.get('memory_percent', 0) > 5]
            if high_mem_procs:
                recommendations.append({
                    'action': 'Optimize memory usage',
                    'processes': [p['name'] for p in high_mem_procs[:2]]
                })
        
        return recommendations
    
    def _analyze_time_quanta_efficiency(self, process_metrics):
        """Analyze time quanta efficiency"""
        analysis = {
            'high_context_switch_processes': [],
            'efficiency_metrics': {}
        }
        
        total_ctx_switches = 0
        total_processes = len(process_metrics)
        
        for pid, metrics in process_metrics.items():
            ctx_switches = (metrics.get('num_voluntary_ctx_switches', 0) + 
                          metrics.get('num_involuntary_ctx_switches', 0))
            total_ctx_switches += ctx_switches
            
            if ctx_switches > 1000:
                analysis['high_context_switch_processes'].append({
                    'pid': pid,
                    'name': metrics.get('name', 'unknown'),
                    'context_switches': ctx_switches
                })
        
        analysis['efficiency_metrics'] = {
            'avg_context_switches_per_process': total_ctx_switches / max(1, total_processes),
            'system_efficiency_score': max(0, 100 - (total_ctx_switches / max(1, total_processes) / 10))
        }
        
        return analysis
    
    def _analyze_energy_efficiency(self, system_metrics):
        """Analyze energy efficiency"""
        cpu_usage = system_metrics.get('cpu_percent', 0)
        cpu_temp = system_metrics.get('cpu_temperature', 0)
        battery = system_metrics.get('battery_percent', 100)
        
        efficiency_score = 100
        
        if cpu_usage > 80:
            efficiency_score -= 20
        elif cpu_usage < 20:
            efficiency_score += 10
        
        if cpu_temp and cpu_temp > 70:
            efficiency_score -= 15
        
        if battery and battery < 30:
            efficiency_score -= 10
        
        return {
            'energy_efficiency_score': max(0, efficiency_score)
        }