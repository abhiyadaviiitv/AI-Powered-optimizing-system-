# src/monitoring/process_monitor.py
import psutil
import time
import threading
import json
from collections import defaultdict, deque
import pandas as pd

class ProcessMonitor:
    def __init__(self, sampling_interval=2.0, history_size=100):
        self.sampling_interval = sampling_interval
        self.process_history = defaultdict(lambda: deque(maxlen=history_size))
        self.system_history = deque(maxlen=history_size)
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Time quanta tracking
        self.time_quanta_data = defaultdict(lambda: deque(maxlen=50))
        
    def start_monitoring(self):
        """Start continuous monitoring in background thread"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 Process monitoring started...")
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("🔍 Process monitoring stopped...")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect all data at once
                system_metrics = self._collect_system_metrics()
                process_metrics = self._collect_process_metrics()
                
                timestamp = time.time()
                
                # Store system metrics
                self.system_history.append({
                    'timestamp': timestamp,
                    **system_metrics
                })
                
                # Store process metrics
                for pid, metrics in process_metrics.items():
                    self.process_history[pid].append({
                        'timestamp': timestamp,
                        **metrics
                    })
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.sampling_interval)
    
    def _collect_system_metrics(self):
        """Collect comprehensive system-wide metrics"""
        # CPU Metrics
        cpu_percent = psutil.cpu_percent(interval=0.5)
        per_core_percent = psutil.cpu_percent(interval=0.5, percpu=True)
        
        # Memory Metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Temperature
        cpu_temp = self._get_cpu_temperature()
        
        # Fans
        fan_speed = self._get_fan_speed()
        
        # Battery
        batt_percent, batt_plugged, batt_time_left = self._get_battery_status()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        
        # Network I/O
        net_io = psutil.net_io_counters()
        
        return {
            # CPU Metrics
            'cpu_percent': cpu_percent,
            'per_core_percent': per_core_percent,
            'cpu_frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'cpu_cores': psutil.cpu_count(logical=False),
            'cpu_logical_cores': psutil.cpu_count(logical=True),
            
            # Memory Metrics
            'ram_percent': memory.percent,
            'ram_used_gb': round(memory.used / (1024**3), 2),
            'ram_total_gb': round(memory.total / (1024**3), 2),
            'swap_percent': swap.percent,
            'swap_used_gb': round(swap.used / (1024**3), 2),
            
            # Temperature & Hardware
            'cpu_temperature': cpu_temp,
            'fan_speed_rpm': fan_speed,
            
            # Battery
            'battery_percent': batt_percent,
            'battery_plugged': batt_plugged,
            'battery_time_left_sec': batt_time_left,
            
            # Disk I/O
            'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
            'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
            
            # Network I/O
            'net_bytes_sent': net_io.bytes_sent if net_io else 0,
            'net_bytes_recv': net_io.bytes_recv if net_io else 0,
            
            # System Load
            'load_avg': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            'total_processes': len(psutil.pids()),
            'running_processes': len([p for p in psutil.process_iter() if p.status() == 'running'])
        }
    
    def _collect_process_metrics(self):
        """Collect comprehensive metrics for all running processes"""
        metrics = {}
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 
                                       'nice', 'status', 'num_threads', 'memory_info',
                                       'io_counters', 'cpu_times', 'create_time']):
            try:
                pid = proc.info['pid']
                
                # Get detailed process info with optimization
                with proc.oneshot():
                    try:
                        cpu_times = proc.cpu_times()
                        io_counters = proc.io_counters() if hasattr(proc, 'io_counters') else None
                        memory_info = proc.memory_info()
                        num_ctx_switches = proc.num_ctx_switches()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                metrics[pid] = {
                    # Basic Info
                    'name': proc.info['name'],
                    'pid': pid,
                    'status': proc.info['status'],
                    'create_time': proc.info['create_time'],
                    
                    # CPU Metrics
                    'cpu_percent': proc.info['cpu_percent'] or 0,
                    'cpu_time_user': cpu_times.user,
                    'cpu_time_system': cpu_times.system,
                    'cpu_time_total': cpu_times.user + cpu_times.system,
                    
                    # Memory Metrics
                    'memory_percent': proc.info['memory_percent'] or 0,
                    'memory_rss_mb': memory_info.rss / (1024 * 1024),
                    'memory_vms_mb': memory_info.vms / (1024 * 1024),
                    
                    # I/O Metrics
                    'io_read_bytes': io_counters.read_bytes if io_counters else 0,
                    'io_write_bytes': io_counters.write_bytes if io_counters else 0,
                    'io_read_count': io_counters.read_count if io_counters else 0,
                    'io_write_count': io_counters.write_count if io_counters else 0,
                    
                    # Process Characteristics
                    'nice': proc.info['nice'],
                    'num_threads': proc.info['num_threads'],
                    'num_voluntary_ctx_switches': num_ctx_switches.voluntary if num_ctx_switches else 0,
                    'num_involuntary_ctx_switches': num_ctx_switches.involuntary if num_ctx_switches else 0,
                    
                    # Time Quanta Related
                    'context_switch_rate': self._calculate_context_switch_rate(pid, num_ctx_switches),
                    'cpu_burst_estimate': self._estimate_cpu_burst(pid, cpu_times.user + cpu_times.system)
                }
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue
                
        return metrics
    
    def _calculate_context_switch_rate(self, pid, num_ctx_switches):
        """Calculate context switch rate for time quanta optimization"""
        if not num_ctx_switches:
            return 0
            
        total_switches = num_ctx_switches.voluntary + num_ctx_switches.involuntary
        
        # Store in history for rate calculation
        current_time = time.time()
        self.time_quanta_data[pid].append({
            'timestamp': current_time,
            'total_switches': total_switches
        })
        
        # Calculate rate (switches per second)
        history = self.time_quanta_data[pid]
        if len(history) >= 2:
            time_diff = history[-1]['timestamp'] - history[0]['timestamp']
            switch_diff = history[-1]['total_switches'] - history[0]['total_switches']
            if time_diff > 0:
                return switch_diff / time_diff
        
        return 0
    
    def _estimate_cpu_burst(self, pid, current_cpu_time):
        """Estimate CPU burst time for time quanta optimization"""
        if pid in self.process_history and self.process_history[pid]:
            last_metrics = self.process_history[pid][-1]
            last_cpu_time = last_metrics.get('cpu_time_total', 0)
            burst_time = max(0, current_cpu_time - last_cpu_time)
            return burst_time
        return 0.1  # Default estimate
    
    def get_current_metrics(self):
        """Get latest metrics snapshot with proper None handling"""
        if not self.system_history:
            return None
            
        latest_system = dict(self.system_history[-1])
        
        # Ensure all required fields have values, not None
        system_defaults = {
            'cpu_percent': 0,
            'ram_percent': 0,
            'ram_used_gb': 0,
            'ram_total_gb': 0,
            'swap_percent': 0,
            'cpu_temperature': None,
            'fan_speed_rpm': None,
            'battery_percent': None,
            'battery_plugged': False,
            'load_avg': 0,
            'total_processes': 0,
            'running_processes': 0
        }
        
        # Replace None values with defaults
        for key, default in system_defaults.items():
            if latest_system.get(key) is None:
                latest_system[key] = default
        
        # Get latest process metrics
        latest_processes = {}
        for pid, history in self.process_history.items():
            if history and history[-1]['timestamp'] == latest_system['timestamp']:
                latest_processes[pid] = dict(history[-1])
        
        return {
            'system': latest_system,
            'processes': latest_processes
        }
    
    # Your existing utility methods
    def _get_cpu_temperature(self):
        try:
            temps = psutil.sensors_temperatures()
            if "coretemp" in temps: 
                return temps["coretemp"][0].current
            elif "cpu_thermal" in temps:
                return temps["cpu_thermal"][0].current
            else:
                return None
        except Exception:
            return None

    def _get_battery_status(self):
        try:
            battery = psutil.sensors_battery()
            if battery:
                return battery.percent, battery.power_plugged, battery.secsleft
            else:
                return None, None, None
        except Exception:
            return None, None, None

    def _get_fan_speed(self):
        try:
            fans = psutil.sensors_fans()
            if fans:
                for name, entries in fans.items():
                    return entries[0].current
            return None
        except Exception:
            return None
    
    def export_to_csv(self, filename="system_usage_complete.csv"):
        """Export complete data to CSV"""
        with open(filename, mode="w", newline="") as file:
            import csv
            writer = csv.writer(file)
            # Write comprehensive header
            writer.writerow([
                "timestamp", "cpu_percent", "per_core_percent", "ram_percent",
                "cpu_temperature", "fan_speed_rpm", "battery_percent", 
                "battery_plugged", "battery_time_left_sec", "pid", "process_name",
                "process_cpu_percent", "process_memory_percent", "process_nice",
                "process_status", "process_threads", "process_io_read_bytes",
                "process_io_write_bytes", "context_switch_rate", "cpu_burst_estimate"
            ])
            
            # Write data
            for system_point in self.system_history:
                timestamp = system_point['timestamp']
                
                # Find processes for this timestamp
                for pid, history in self.process_history.items():
                    for process_point in history:
                        if process_point['timestamp'] == timestamp:
                            writer.writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                                system_point.get('cpu_percent', 0),
                                str(system_point.get('per_core_percent', [])),
                                system_point.get('ram_percent', 0),
                                system_point.get('cpu_temperature', 0),
                                system_point.get('fan_speed_rpm', 0),
                                system_point.get('battery_percent', 0),
                                system_point.get('battery_plugged', False),
                                system_point.get('battery_time_left_sec', 0),
                                pid,
                                process_point.get('name', 'unknown'),
                                process_point.get('cpu_percent', 0),
                                process_point.get('memory_percent', 0),
                                process_point.get('nice', 0),
                                process_point.get('status', 'unknown'),
                                process_point.get('num_threads', 0),
                                process_point.get('io_read_bytes', 0),
                                process_point.get('io_write_bytes', 0),
                                process_point.get('context_switch_rate', 0),
                                process_point.get('cpu_burst_estimate', 0)
                            ])