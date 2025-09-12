import psutil
import pandas as pd
import numpy as np
import time
import os
import threading
import json
from datetime import datetime
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

class SystemDataCollector:
    def __init__(self, collection_interval=1.0):
        """
        Real-time system data collector for CPU scheduling analysis
        
        Args:
            collection_interval: How often to collect data (seconds)
        """
        self.collection_interval = collection_interval
        self.is_collecting = False
        self.data_buffer = []
        self.process_history = defaultdict(list)
        self.system_history = []
        
        # Thread for data collection
        self.collection_thread = None
        
        print("🖥️  System Data Collector Initialized")
        print(f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024**3):.1f}GB RAM")
        print(f"OS: {os.name}, Python PID: {os.getpid()}")
    
    def get_process_info(self, proc):
        """Extract comprehensive process information"""
        try:
            # Get process info
            with proc.oneshot():
                pinfo = proc.as_dict(attrs=[
                    'pid', 'name', 'status', 'create_time', 'ppid',
                    'cpu_percent', 'memory_info', 'memory_percent',
                    'num_threads', 'nice', 'ionice',
                    'cpu_times', 'io_counters', 'cmdline'
                ])
            
            # Calculate additional metrics
            current_time = time.time()
            age = current_time - pinfo['create_time']
            
            # Fix cmdline handling - ensure it's a list before slicing
            cmdline = pinfo.get('cmdline', [])
            if cmdline is None:
                cmdline = []
            
            # Process type classification (heuristic)
            process_type = self._classify_process(pinfo['name'], cmdline)
            
            # Memory usage in MB
            memory_mb = pinfo['memory_info'].rss / (1024 * 1024) if pinfo['memory_info'] else 0
            
            # I/O intensity (if available)
            io_intensity = 0
            if pinfo['io_counters']:
                io_intensity = (pinfo['io_counters'].read_bytes + pinfo['io_counters'].write_bytes) / (1024 * 1024)  # MB
            
            return {
                'timestamp': current_time,
                'pid': pinfo['pid'],
                'name': pinfo['name'],
                'status': pinfo['status'],
                'ppid': pinfo['ppid'],
                'cpu_percent': pinfo['cpu_percent'] or 0,
                'memory_percent': pinfo['memory_percent'] or 0,
                'memory_mb': memory_mb,
                'num_threads': pinfo['num_threads'] or 1,
                'nice': pinfo['nice'] if pinfo['nice'] is not None else 0,
                'process_age': age,
                'process_type': process_type,
                'io_intensity': io_intensity,
                'user_time': pinfo['cpu_times'].user if pinfo['cpu_times'] else 0,
                'system_time': pinfo['cpu_times'].system if pinfo['cpu_times'] else 0,
                'cmdline': ' '.join(cmdline[:3])  # First 3 command args - now safe
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None
    
    def _classify_process(self, name, cmdline):
        """Classify process type based on name and command line"""
        name_lower = name.lower()
        cmdline_str = ' '.join(cmdline).lower() if cmdline else ''
        
        # System processes
        if any(x in name_lower for x in ['kernel', 'kthreadd', 'ksoftirqd', 'systemd', 'init']):
            return 'system'
        
        # Interactive applications
        if any(x in name_lower for x in ['chrome', 'firefox', 'safari', 'code', 'vscode', 'atom', 
                                        'sublime', 'terminal', 'gnome', 'kde', 'xorg', 'wayland']):
            return 'interactive'
        
        # Development/compilation
        if any(x in name_lower for x in ['gcc', 'clang', 'make', 'cmake', 'ninja', 'python', 'java', 'node']):
            return 'development'
        
        # Background services
        if any(x in name_lower for x in ['daemon', 'service', 'server', 'cron', 'ssh', 'docker']):
            return 'service'
        
        # Batch/compute intensive
        if any(x in cmdline_str for x in ['compute', 'process', 'batch', 'simulation']):
            return 'batch'
        
        return 'other'
    
    def get_system_state(self):
        """Get current system-wide metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        
        # Network I/O
        net_io = psutil.net_io_counters()
        
        # Process counts
        process_count = len(psutil.pids())
        
        return {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
            'load_avg_1min': load_avg[0],
            'load_avg_5min': load_avg[1],
            'load_avg_15min': load_avg[2],
            'memory_total_gb': memory.total / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'swap_total_gb': swap.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3),
            'swap_percent': swap.percent,
            'disk_read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
            'disk_write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0,
            'network_sent_mb': net_io.bytes_sent / (1024**2) if net_io else 0,
            'network_recv_mb': net_io.bytes_recv / (1024**2) if net_io else 0,
            'process_count': process_count,
            'context_switches': psutil.cpu_stats().ctx_switches if hasattr(psutil.cpu_stats(), 'ctx_switches') else 0,
            'interrupts': psutil.cpu_stats().interrupts if hasattr(psutil.cpu_stats(), 'interrupts') else 0
        }
    
    def collect_snapshot(self):
        """Collect a single snapshot of system and process data"""
        snapshot_time = time.time()
        
        # Get system state
        system_state = self.get_system_state()
        
        # Get all running processes
        processes = []
        for proc in psutil.process_iter():
            proc_info = self.get_process_info(proc)
            if proc_info:
                processes.append(proc_info)
        
        # Sort processes by CPU usage (simulates scheduler priority)
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        
        # Create combined snapshot
        snapshot = {
            'system': system_state,
            'processes': processes[:50],  # Top 50 processes by CPU
            'snapshot_id': len(self.data_buffer)
        }
        
        return snapshot
    
    def _collection_worker(self):
        """Background thread for continuous data collection"""
        print(f"📊 Starting data collection (interval: {self.collection_interval}s)")
        
        while self.is_collecting:
            try:
                snapshot = self.collect_snapshot()
                self.data_buffer.append(snapshot)
                self.system_history.append(snapshot['system'])
                
                # Update process history
                for proc in snapshot['processes']:
                    self.process_history[proc['pid']].append(proc)
                
                # Print progress
                if len(self.data_buffer) % 10 == 0:
                    print(f"📈 Collected {len(self.data_buffer)} snapshots")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"❌ Collection error: {e}")
                time.sleep(self.collection_interval)
    
    def start_collection(self):
        """Start background data collection"""
        if self.is_collecting:
            print("⚠️  Collection already running!")
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_worker, daemon=True)
        self.collection_thread.start()
    
    def stop_collection(self):
        """Stop data collection"""
        if not self.is_collecting:
            print("⚠️  Collection not running!")
            return
        
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        print(f"🛑 Stopped collection. Total snapshots: {len(self.data_buffer)}")
    
    def export_to_csv(self, filename=None):
        """Export collected data to CSV files"""
        if not self.data_buffer:
            print("❌ No data collected yet!")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            base_filename = f"system_data_{timestamp}"
        else:
            base_filename = filename
        
        # Export system metrics
        system_df = pd.DataFrame(self.system_history)
        system_csv = f"{base_filename}_system.csv"
        system_df.to_csv(system_csv, index=False)
        
        # Export process data (flattened)
        process_records = []
        for snapshot in self.data_buffer:
            snapshot_id = snapshot['snapshot_id']
            system_state = snapshot['system']
            
            for proc in snapshot['processes']:
                record = {
                    'snapshot_id': snapshot_id,
                    'system_timestamp': system_state['timestamp'],
                    'system_cpu_percent': system_state['cpu_percent'],
                    'system_memory_percent': system_state['memory_percent'],
                    'system_load_1min': system_state['load_avg_1min'],
                    'system_process_count': system_state['process_count'],
                    **proc  # Add all process metrics
                }
                process_records.append(record)
        
        process_df = pd.DataFrame(process_records)
        process_csv = f"{base_filename}_processes.csv"
        process_df.to_csv(process_csv, index=False)
        
        # Create scheduling decisions dataset
        scheduling_records = []
        for i, snapshot in enumerate(self.data_buffer[:-1]):
            current_system = snapshot['system']
            next_snapshot = self.data_buffer[i + 1]
            next_system = next_snapshot['system']
            
            if snapshot['processes']:
                # The "scheduled" process is the one with highest CPU usage
                scheduled_proc = max(snapshot['processes'], key=lambda x: x['cpu_percent'])
                
                # Features: current system state + process characteristics
                features = {
                    'decision_timestamp': current_system['timestamp'],
                    'cpu_usage': current_system['cpu_percent'],
                    'memory_usage': current_system['memory_percent'],
                    'load_avg': current_system['load_avg_1min'],
                    'process_count': current_system['process_count'],
                    'context_switches': current_system.get('context_switches', 0),
                    
                    # Queue statistics (top 5 processes)
                    'avg_cpu_demand': np.mean([p['cpu_percent'] for p in snapshot['processes'][:5]]),
                    'max_cpu_demand': max([p['cpu_percent'] for p in snapshot['processes'][:5]]),
                    'avg_memory_usage': np.mean([p['memory_mb'] for p in snapshot['processes'][:5]]),
                    'avg_nice_value': np.mean([p['nice'] for p in snapshot['processes'][:5]]),
                    'interactive_processes': sum(1 for p in snapshot['processes'][:10] if p['process_type'] == 'interactive'),
                    'system_processes': sum(1 for p in snapshot['processes'][:10] if p['process_type'] == 'system'),
                    
                    # Selected process characteristics
                    'selected_pid': scheduled_proc['pid'],
                    'selected_cpu_percent': scheduled_proc['cpu_percent'],
                    'selected_memory_mb': scheduled_proc['memory_mb'],
                    'selected_nice': scheduled_proc['nice'],
                    'selected_threads': scheduled_proc['num_threads'],
                    'selected_age': scheduled_proc['process_age'],
                    'selected_type': scheduled_proc['process_type'],
                    'selected_io_intensity': scheduled_proc['io_intensity'],
                    
                    # Performance outcomes (from next snapshot)
                    'next_cpu_usage': next_system['cpu_percent'],
                    'next_memory_usage': next_system['memory_percent'],
                    'cpu_usage_change': next_system['cpu_percent'] - current_system['cpu_percent'],
                    'memory_usage_change': next_system['memory_percent'] - current_system['memory_percent'],
                    'time_quantum': self.collection_interval
                }
                
                scheduling_records.append(features)
        
        scheduling_df = pd.DataFrame(scheduling_records)
        scheduling_csv = f"{base_filename}_scheduling.csv"
        scheduling_df.to_csv(scheduling_csv, index=False)
        
        print(f"✅ Data exported successfully:")
        print(f"   📊 System metrics: {system_csv} ({len(system_df)} records)")
        print(f"   🔧 Process data: {process_csv} ({len(process_df)} records)")  
        print(f"   🎯 Scheduling decisions: {scheduling_csv} ({len(scheduling_df)} records)")
        
        return {
            'system_file': system_csv,
            'process_file': process_csv,
            'scheduling_file': scheduling_csv,
            'summary': {
                'total_snapshots': len(self.data_buffer),
                'system_records': len(system_df),
                'process_records': len(process_df),
                'scheduling_decisions': len(scheduling_df),
                'collection_duration': self.system_history[-1]['timestamp'] - self.system_history[0]['timestamp'] if self.system_history else 0
            }
        }
    
    def create_visualizations(self, export_files):
        """Create visualizations of collected data"""
        try:
            # Load the data
            system_df = pd.read_csv(export_files['system_file'])
            scheduling_df = pd.read_csv(export_files['scheduling_file'])
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Real-Time System Performance Analysis', fontsize=16)
            
            # CPU usage over time
            system_df['time_relative'] = system_df['timestamp'] - system_df['timestamp'].iloc[0]
            axes[0, 0].plot(system_df['time_relative'], system_df['cpu_percent'], label='CPU %', color='red')
            axes[0, 0].plot(system_df['time_relative'], system_df['memory_percent'], label='Memory %', color='blue')
            axes[0, 0].set_title('System Resource Usage Over Time')
            axes[0, 0].set_xlabel('Time (seconds)')
            axes[0, 0].set_ylabel('Percentage')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Process type distribution in scheduling decisions
            type_counts = scheduling_df['selected_type'].value_counts()
            axes[0, 1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Scheduled Process Types')
            
            # CPU vs Memory usage correlation
            axes[1, 0].scatter(scheduling_df['selected_cpu_percent'], scheduling_df['selected_memory_mb'], 
                              alpha=0.6, c=scheduling_df.index, cmap='viridis')
            axes[1, 0].set_xlabel('Process CPU %')
            axes[1, 0].set_ylabel('Process Memory (MB)')
            axes[1, 0].set_title('CPU vs Memory Usage (Scheduled Processes)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # System load vs scheduling decisions
            axes[1, 1].scatter(scheduling_df['load_avg'], scheduling_df['selected_cpu_percent'], alpha=0.6)
            axes[1, 1].set_xlabel('System Load Average')
            axes[1, 1].set_ylabel('Selected Process CPU %')
            axes[1, 1].set_title('System Load vs Process Selection')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save visualization
            viz_filename = export_files['scheduling_file'].replace('_scheduling.csv', '_analysis.png')
            plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved: {viz_filename}")
            
            plt.show()
            
            return viz_filename
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return None
    
    def print_summary(self):
        """Print collection summary"""
        if not self.data_buffer:
            print("❌ No data collected")
            return
        
        print(f"\n📈 Collection Summary:")
        print(f"   Snapshots collected: {len(self.data_buffer)}")
        print(f"   Unique processes seen: {len(self.process_history)}")
        
        if self.system_history:
            duration = self.system_history[-1]['timestamp'] - self.system_history[0]['timestamp']
            print(f"   Collection duration: {duration:.1f} seconds")
            
            avg_cpu = np.mean([s['cpu_percent'] for s in self.system_history])
            avg_memory = np.mean([s['memory_percent'] for s in self.system_history])
            
            print(f"   Average CPU usage: {avg_cpu:.1f}%")
            print(f"   Average Memory usage: {avg_memory:.1f}%")

# Usage example and demo
def run_collection_demo():
    """Run a demonstration of the system data collector"""
    print("🚀 Starting Real System Data Collection Demo")
    
    # Create collector
    collector = SystemDataCollector(collection_interval=2.0)  # Collect every 2 seconds
    
    # Take a single snapshot first
    print("\n📷 Taking initial snapshot...")
    snapshot = collector.collect_snapshot()
    print(f"   System CPU: {snapshot['system']['cpu_percent']:.1f}%")
    print(f"   System Memory: {snapshot['system']['memory_percent']:.1f}%")
    print(f"   Active processes: {len(snapshot['processes'])}")
    print(f"   Top process: {snapshot['processes'][0]['name']} ({snapshot['processes'][0]['cpu_percent']:.1f}% CPU)")
    
    # Start continuous collection
    collector.start_collection()
    
    print("\n⏰ Collecting data for 30 seconds...")
    print("   (Try running some applications or tasks to generate interesting data)")
    
    # Let it collect for 30 seconds
    time.sleep(30)
    
    # Stop collection
    collector.stop_collection()
    
    # Print summary
    collector.print_summary()
    
    # Export to CSV
    print("\n💾 Exporting data...")
    files = collector.export_to_csv()
    
    if files:
        # Create visualizations
        print("\n📊 Creating visualizations...")
        viz_file = collector.create_visualizations(files)
        
        print(f"\n✅ Demo complete! Files generated:")
        for key, filename in files.items():
            if key != 'summary':
                print(f"   {filename}")
    
    return collector, files

if __name__ == "__main__":
    print("Real System Data Collector for CPU Scheduling Analysis")
    print("=" * 55)
    
    # Run the demo
    collector, files = run_collection_demo()
    
    print("\n🎉 You now have real system data for CPU scheduling analysis!")
    print("\nWhat you can do with this data:")
    print("  • Train machine learning models on actual scheduling patterns")
    print("  • Analyze your system's resource utilization patterns")
    print("  • Identify performance bottlenecks and scheduling inefficiencies")
    print("  • Compare different workload characteristics")
    print("  • Build custom scheduling algorithms based on real behavior")