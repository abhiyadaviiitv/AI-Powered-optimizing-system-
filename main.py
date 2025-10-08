# main.py
from src.monitoring.process_monitor import ProcessMonitor
from src.monitoring.metrics_analyzer import EnhancedMetricsAnalyzer
import time
import signal
import sys
import os

class ProcessMonitorApp:
    def __init__(self):
        self.monitor = ProcessMonitor(sampling_interval=2.0)
        self.analyzer = EnhancedMetricsAnalyzer(self.monitor)
        self.running = False
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        self.stop()
        sys.exit(0)
        
    def start(self):
        """Start the process monitoring application"""
        print("🚀 Starting Advanced Process Monitor")
        print("=" * 70)
        print("Initializing monitoring system...")
        
        # Start monitoring
        self.monitor.start_monitoring()
        time.sleep(3)  # Let some data accumulate
        
        print("Monitoring system ready! Starting real-time analysis...")
        time.sleep(1)
        
        self.running = True
        iteration = 0
        
        try:
            while self.running:
                iteration += 1
                self.display_comprehensive_analysis(iteration)
                time.sleep(3)  # Update every 3 seconds
                
        except KeyboardInterrupt:
            self.stop()
            
    def display_comprehensive_analysis(self, iteration):
        """Display comprehensive system analysis"""
        # Clear screen
        os.system('clear')
        
        # Get current metrics
        current_data = self.monitor.get_current_metrics()
        if not current_data:
            print("No data available yet...")
            return
            
        system_metrics = current_data['system']
        process_metrics = current_data['processes']
        
        # Perform analysis
        analysis = self.analyzer.comprehensive_analysis()
        
        # Display header
        print("🔍 ADVANCED PROCESS MONITOR - COMPREHENSIVE ANALYSIS")
        print("=" * 70)
        print(f"Iteration: {iteration} | Processes: {len(process_metrics)} | {time.strftime('%H:%M:%S')}")
        print("=" * 70)
        
        # System Overview
        self._display_system_overview(system_metrics, analysis)
        
        # Critical Processes
        self._display_critical_processes(analysis)
        
        # Performance Metrics
        self._display_performance_metrics(analysis)
        
        # Process Categories
        self._display_process_categories(analysis)
        
        # Bottlenecks & Recommendations
        self._display_bottlenecks_recommendations(analysis)
        
        # Footer
        print("\n" + "=" * 70)
        print("💡 Press Ctrl+C to stop monitoring and export data")
        print("=" * 70)
    
    def _display_system_overview(self, system_metrics, analysis):
        """Display comprehensive system overview"""
        health = analysis.get('system_health', {})
        
        print("\n📊 SYSTEM OVERVIEW")
        print("-" * 70)
        
        # CPU Section
        cpu_usage = system_metrics.get('cpu_percent', 0)
        cpu_cores = system_metrics.get('cpu_logical_cores', 'N/A')
        cpu_freq = system_metrics.get('cpu_frequency', 'N/A')
        
        print(f"🖥️  CPU:    {cpu_usage:6.1f}% | Cores: {cpu_cores} | Freq: {cpu_freq} MHz")
        
        # Memory Section
        ram_usage = system_metrics.get('ram_percent', 0)
        ram_used = system_metrics.get('ram_used_gb', 0)
        ram_total = system_metrics.get('ram_total_gb', 0)
        swap_usage = system_metrics.get('swap_percent', 0)
        
        print(f"💾 RAM:    {ram_usage:6.1f}% | {ram_used:.1f}/{ram_total:.1f} GB | Swap: {swap_usage:.1f}%")
        
        # Temperature & Hardware - ROBUST handling of None values
        temp = system_metrics.get('cpu_temperature')
        fan_speed = system_metrics.get('fan_speed_rpm')
        battery = system_metrics.get('battery_percent')
        plugged = "🔌" if system_metrics.get('battery_plugged') else "🔋"

        # Safe formatting that handles None values
        temp_display = f"{temp:>6.1f}" if temp is not None else "   N/A"
        fan_display = f"{fan_speed:>6}" if fan_speed is not None else "   N/A"  
        battery_display = f"{battery}" if battery is not None else "N/A"

        print(f"🌡️  TEMP:   {temp_display}°C | Fan: {fan_display} RPM | Battery: {battery_display}% {plugged}")
        
        # System Load
        load_avg = system_metrics.get('load_avg', 0)
        total_procs = system_metrics.get('total_processes', 0)
        running_procs = system_metrics.get('running_processes', 0)
        
        print(f"📈 LOAD:   {load_avg:6.2f} | Processes: {running_procs}/{total_procs} running")
        
        # Health Status
        health_score = health.get('overall_health_score', 0)
        status_emoji = "🟢" if health_score > 80 else "🟡" if health_score > 60 else "🔴"
        
        print(f"❤️  HEALTH: {health_score:5.1f}/100 {status_emoji} | {health.get('load_status', 'UNKNOWN').upper()}")

    def _display_critical_processes(self, analysis):
        """Display critical processes analysis"""
        intelligence = analysis.get('process_intelligence', {})
        critical_processes = intelligence.get('critical_processes', [])
        
        if critical_processes:
            print(f"\n🚨 CRITICAL PROCESSES ({len(critical_processes)})")
            print("-" * 70)
            print(f"{'PID':<8} {'NAME':<20} {'CPU%':<6} {'MEM%':<6} {'PRIO':<6} {'CRITICALITY':<12}")
            print("-" * 70)
            
            for proc in critical_processes[:8]:  # Show top 8
                name = proc['name'][:19] if len(proc['name']) > 19 else proc['name']
                criticality = f"{proc['criticality_score']:.2f}"
                
                print(f"{proc['pid']:<8} {name:<20} {proc['cpu_usage']:<6.1f} "
                      f"{proc['memory_usage']:<6.1f} {proc.get('priority', 0):<6} {criticality:<12}")
        else:
            print(f"\n✅ NO CRITICAL PROCESSES DETECTED")
            print("-" * 70)

    def _display_performance_metrics(self, analysis):
        """Display performance metrics"""
        performance = analysis.get('performance_metrics', {})
        time_quanta = analysis.get('time_quanta_analysis', {})
        
        print(f"\n⚡ PERFORMANCE METRICS")
        print("-" * 70)
        
        # Basic performance metrics
        throughput = performance.get('throughput', 0)
        response_time = performance.get('avg_response_time', 0)
        cpu_efficiency = performance.get('cpu_efficiency', 0)
        
        print(f"📊 Throughput:    {throughput:6.1f} | Response Time: {response_time:6.1f}ms")
        print(f"🎯 CPU Efficiency: {cpu_efficiency:5.1f}% | Context Switches: {performance.get('total_context_switches', 0)}")
        
        # Time quanta efficiency
        efficiency_metrics = time_quanta.get('efficiency_metrics', {})
        avg_ctx_switches = efficiency_metrics.get('avg_context_switches_per_process', 0)
        efficiency_score = efficiency_metrics.get('system_efficiency_score', 0)
        
        print(f"⏰ Avg Ctx/Proc:  {avg_ctx_switches:6.1f} | Efficiency Score: {efficiency_score:5.1f}/100")
        
        # High context switch processes
        high_ctx_procs = time_quanta.get('high_context_switch_processes', [])
        if high_ctx_procs:
            print(f"🔁 High Ctx Switch: {len(high_ctx_procs)} processes need optimization")

    def _display_process_categories(self, analysis):
        """Display process categories analysis"""
        intelligence = analysis.get('process_intelligence', {})
        categories = intelligence.get('process_categories', {})
        behavioral_patterns = intelligence.get('behavioral_patterns', [])
        
        print(f"\n📂 PROCESS CATEGORIES & PATTERNS")
        print("-" * 70)
        
        # Display category counts
        if categories:
            for category, pids in categories.items():
                if pids:  # Only show non-empty categories
                    emoji = self._get_category_emoji(category)
                    print(f"{emoji} {category.replace('_', ' ').title():<18}: {len(pids):>3} processes")
        
        # Display common behavioral patterns
        if behavioral_patterns:
            pattern_counts = {}
            for pattern in behavioral_patterns:
                ptype = pattern['pattern_type']
                pattern_counts[ptype] = pattern_counts.get(ptype, 0) + 1
            
            print(f"\n🎭 BEHAVIORAL PATTERNS:")
            for pattern, count in list(pattern_counts.items())[:4]:  # Top 4 patterns
                print(f"   • {pattern.replace('_', ' ').title()}: {count} processes")

    def _display_bottlenecks_recommendations(self, analysis):
        """Display bottlenecks and recommendations"""
        bottlenecks = analysis.get('bottleneck_analysis', [])
        recommendations = analysis.get('scheduling_recommendations', [])
        resource_forecast = analysis.get('resource_forecasting', {})
        energy_efficiency = analysis.get('energy_efficiency', {})
        
        # Bottlenecks
        if bottlenecks:
            print(f"\n🚧 SYSTEM BOTTLENECKS")
            print("-" * 70)
            for bottleneck in bottlenecks[:2]:  # Show top 2
                severity_emoji = "🔴" if bottleneck['severity'] == 'high' else "🟡" if bottleneck['severity'] == 'medium' else "🟢"
                print(f"{severity_emoji} {bottleneck['description']}")
                if bottleneck['suggested_actions']:
                    print(f"   💡 {bottleneck['suggested_actions'][0]}")
        
        # Resource Forecasting
        if resource_forecast:
            forecast = resource_forecast.get('cpu_forecast', 'stable')
            trend_emoji = "📈" if forecast == 'increasing' else "📉" if forecast == 'decreasing' else "➡️"
            print(f"\n🔮 RESOURCE FORECAST: {trend_emoji} {forecast.upper()}")
        
        # Energy Efficiency
        if energy_efficiency:
            eff_score = energy_efficiency.get('energy_efficiency_score', 0)
            eff_emoji = "🟢" if eff_score > 80 else "🟡" if eff_score > 60 else "🔴"
            print(f"🔋 ENERGY EFFICIENCY: {eff_score:.1f}/100 {eff_emoji}")

    def _get_category_emoji(self, category):
        """Get emoji for process category"""
        emoji_map = {
            'interactive': '🚀',
            'cpu_intensive': '🔥', 
            'memory_intensive': '💾',
            'io_intensive': '📁',
            'background': '⚙️',
            'system': '🛡️',
            'browser': '🌐',
            'developer_tool': '💻',
            'parallel': '🔀'
        }
        return emoji_map.get(category, '📄')
    
    def stop(self):
        """Stop the application"""
        print("\n🛑 Stopping Process Monitor...")
        self.running = False
        self.monitor.stop_monitoring()
        
        # Export data
        self.monitor.export_to_csv()
        print("💾 Data exported to system_usage_complete.csv")
        
        # Show summary
        self._display_final_summary()
        print("✅ Process Monitor stopped successfully!")
    
    def _display_final_summary(self):
        """Display final monitoring summary"""
        if hasattr(self.monitor, 'system_history') and self.monitor.system_history:
            total_samples = len(self.monitor.system_history)
            total_processes = len(self.monitor.process_history)
            
            print(f"\n📈 MONITORING SUMMARY")
            print("-" * 40)
            print(f"Total samples collected: {total_samples}")
            print(f"Unique processes tracked: {total_processes}")
            print(f"Monitoring duration: {total_samples * 2} seconds")

def main():
    app = ProcessMonitorApp()
    app.start()

if __name__ == "__main__":
    main()