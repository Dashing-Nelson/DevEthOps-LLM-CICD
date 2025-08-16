"""
Pipeline Monitor - Performance and resource monitoring for pipeline stages
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict


class PipelineMonitor:
    """
    Monitor pipeline stage performance and resource utilization.
    Tracks execution time, memory usage, CPU usage, and other metrics.
    """
    
    def __init__(self):
        """Initialize the pipeline monitor"""
        self.logger = logging.getLogger(__name__)
        
        # Stage tracking
        self.stage_metrics = {}
        self.current_stages = {}
        
        # Resource monitoring
        self.resource_samples = defaultdict(list)
        self.monitoring_active = {}
        
        self.logger.info("Pipeline monitor initialized")
    
    def start_stage(self, stage_name: str) -> str:
        """
        Start monitoring a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            str: Stage execution ID
        """
        stage_id = f"{stage_name}_{int(time.time())}"
        
        start_time = time.time()
        start_datetime = datetime.now()
        
        # Initialize stage metrics
        self.stage_metrics[stage_id] = {
            'stage_name': stage_name,
            'start_time': start_time,
            'start_datetime': start_datetime.isoformat(),
            'end_time': None,
            'end_datetime': None,
            'duration': None,
            'resource_samples': [],
            'initial_memory': self._get_memory_usage(),
            'initial_cpu': self._get_cpu_usage(),
            'peak_memory': self._get_memory_usage(),
            'peak_cpu': self._get_cpu_usage(),
            'status': 'running'
        }
        
        # Store current stage reference
        self.current_stages[stage_name] = stage_id
        self.monitoring_active[stage_id] = True
        
        # Start resource monitoring for this stage
        self._start_resource_monitoring(stage_id)
        
        self.logger.info(f"Started monitoring stage: {stage_name} (ID: {stage_id})")
        return stage_id
    
    def end_stage(self, stage_name: str) -> Optional[float]:
        """
        End monitoring for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            Optional[float]: Duration in seconds, None if stage not found
        """
        stage_id = self.current_stages.get(stage_name)
        
        if not stage_id or stage_id not in self.stage_metrics:
            self.logger.warning(f"Stage {stage_name} not found for ending monitoring")
            return None
        
        # Stop resource monitoring
        self.monitoring_active[stage_id] = False
        
        # Record end time
        end_time = time.time()
        end_datetime = datetime.now()
        
        stage_metrics = self.stage_metrics[stage_id]
        stage_metrics['end_time'] = end_time
        stage_metrics['end_datetime'] = end_datetime.isoformat()
        stage_metrics['duration'] = end_time - stage_metrics['start_time']
        stage_metrics['status'] = 'completed'
        
        # Final resource snapshot
        final_memory = self._get_memory_usage()
        final_cpu = self._get_cpu_usage()
        
        stage_metrics['final_memory'] = final_memory
        stage_metrics['final_cpu'] = final_cpu
        stage_metrics['memory_delta'] = final_memory - stage_metrics['initial_memory']
        
        # Calculate resource statistics
        self._calculate_resource_statistics(stage_id)
        
        # Clean up
        if stage_name in self.current_stages:
            del self.current_stages[stage_name]
        
        duration = stage_metrics['duration']
        self.logger.info(f"Completed monitoring stage: {stage_name} (Duration: {duration:.2f}s)")
        
        return duration
    
    def _start_resource_monitoring(self, stage_id: str):
        """Start background resource monitoring for a stage"""
        import threading
        
        def monitor_resources():
            """Background thread function for resource monitoring"""
            while self.monitoring_active.get(stage_id, False):
                try:
                    timestamp = time.time()
                    memory_usage = self._get_memory_usage()
                    cpu_usage = self._get_cpu_usage()
                    
                    # Get additional system metrics
                    disk_usage = self._get_disk_usage()
                    network_io = self._get_network_io()
                    
                    sample = {
                        'timestamp': timestamp,
                        'memory_mb': memory_usage,
                        'cpu_percent': cpu_usage,
                        'disk_usage_percent': disk_usage,
                        'network_bytes_sent': network_io.get('bytes_sent', 0),
                        'network_bytes_recv': network_io.get('bytes_recv', 0)
                    }
                    
                    # Store sample
                    if stage_id in self.stage_metrics:
                        self.stage_metrics[stage_id]['resource_samples'].append(sample)
                        
                        # Update peak values
                        if memory_usage > self.stage_metrics[stage_id]['peak_memory']:
                            self.stage_metrics[stage_id]['peak_memory'] = memory_usage
                        
                        if cpu_usage > self.stage_metrics[stage_id]['peak_cpu']:
                            self.stage_metrics[stage_id]['peak_cpu'] = cpu_usage
                    
                    # Sample every 2 seconds
                    time.sleep(2)
                    
                except Exception as e:
                    self.logger.error(f"Error in resource monitoring: {str(e)}")
                    break
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def _get_disk_usage(self) -> float:
        """Get current disk usage percentage"""
        try:
            disk = psutil.disk_usage('/')
            return disk.percent
        except Exception:
            return 0.0
    
    def _get_network_io(self) -> Dict[str, int]:
        """Get network I/O statistics"""
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        except Exception:
            return {}
    
    def _calculate_resource_statistics(self, stage_id: str):
        """Calculate resource usage statistics for a completed stage"""
        try:
            stage_metrics = self.stage_metrics[stage_id]
            samples = stage_metrics['resource_samples']
            
            if not samples:
                return
            
            # Extract metrics
            memory_values = [sample['memory_mb'] for sample in samples]
            cpu_values = [sample['cpu_percent'] for sample in samples]
            
            # Calculate statistics
            resource_stats = {
                'memory': {
                    'min': min(memory_values),
                    'max': max(memory_values),
                    'mean': sum(memory_values) / len(memory_values),
                    'samples': len(memory_values)
                },
                'cpu': {
                    'min': min(cpu_values),
                    'max': max(cpu_values),
                    'mean': sum(cpu_values) / len(cpu_values),
                    'samples': len(cpu_values)
                }
            }
            
            stage_metrics['resource_statistics'] = resource_stats
            
        except Exception as e:
            self.logger.error(f"Error calculating resource statistics: {str(e)}")
    
    def get_stage_metrics(self, stage_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for a specific stage or all stages.
        
        Args:
            stage_name: Name of the stage (optional, returns all if None)
            
        Returns:
            Dict containing stage metrics
        """
        if stage_name:
            # Find the most recent stage with this name
            matching_stages = {
                stage_id: metrics 
                for stage_id, metrics in self.stage_metrics.items() 
                if metrics['stage_name'] == stage_name
            }
            
            if matching_stages:
                # Return the most recent one
                latest_stage_id = max(matching_stages.keys(), key=lambda x: self.stage_metrics[x]['start_time'])
                return {latest_stage_id: matching_stages[latest_stage_id]}
            else:
                return {}
        else:
            return self.stage_metrics.copy()
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("PIPELINE PERFORMANCE REPORT")
        report_lines.append("=" * 70)
        
        # Overall summary
        total_stages = len(self.stage_metrics)
        completed_stages = sum(1 for metrics in self.stage_metrics.values() if metrics['status'] == 'completed')
        
        report_lines.append(f"\nOverall Summary:")
        report_lines.append(f"  Total stages monitored: {total_stages}")
        report_lines.append(f"  Completed stages: {completed_stages}")
        
        if completed_stages > 0:
            total_duration = sum(
                metrics.get('duration', 0) 
                for metrics in self.stage_metrics.values() 
                if metrics['status'] == 'completed'
            )
            report_lines.append(f"  Total pipeline duration: {total_duration:.2f} seconds")
            report_lines.append(f"  Average stage duration: {total_duration/completed_stages:.2f} seconds")
        
        # Stage-by-stage breakdown
        report_lines.append(f"\nStage Breakdown:")
        report_lines.append("-" * 50)
        
        # Group by stage name and show the latest execution
        stage_names = {}
        for stage_id, metrics in self.stage_metrics.items():
            stage_name = metrics['stage_name']
            if stage_name not in stage_names or metrics['start_time'] > stage_names[stage_name]['start_time']:
                stage_names[stage_name] = metrics
        
        for stage_name, metrics in stage_names.items():
            report_lines.append(f"\n{stage_name.upper()}:")
            
            # Basic info
            status = metrics['status']
            duration = metrics.get('duration', 'N/A')
            if isinstance(duration, float):
                duration = f"{duration:.2f}s"
            
            report_lines.append(f"  Status: {status}")
            report_lines.append(f"  Duration: {duration}")
            report_lines.append(f"  Start: {metrics['start_datetime']}")
            
            if metrics.get('end_datetime'):
                report_lines.append(f"  End: {metrics['end_datetime']}")
            
            # Resource usage
            initial_memory = metrics.get('initial_memory', 0)
            peak_memory = metrics.get('peak_memory', 0)
            memory_delta = metrics.get('memory_delta', 0)
            
            report_lines.append(f"  Memory Usage:")
            report_lines.append(f"    Initial: {initial_memory:.1f} MB")
            report_lines.append(f"    Peak: {peak_memory:.1f} MB")
            report_lines.append(f"    Delta: {memory_delta:+.1f} MB")
            
            # Resource statistics
            resource_stats = metrics.get('resource_statistics', {})
            if resource_stats:
                memory_stats = resource_stats.get('memory', {})
                cpu_stats = resource_stats.get('cpu', {})
                
                if memory_stats:
                    report_lines.append(f"  Memory Statistics:")
                    report_lines.append(f"    Average: {memory_stats.get('mean', 0):.1f} MB")
                    report_lines.append(f"    Min/Max: {memory_stats.get('min', 0):.1f}/{memory_stats.get('max', 0):.1f} MB")
                
                if cpu_stats:
                    report_lines.append(f"  CPU Statistics:")
                    report_lines.append(f"    Average: {cpu_stats.get('mean', 0):.1f}%")
                    report_lines.append(f"    Min/Max: {cpu_stats.get('min', 0):.1f}/{cpu_stats.get('max', 0):.1f}%")
        
        # Performance insights
        report_lines.append(f"\n{'-' * 50}")
        report_lines.append("PERFORMANCE INSIGHTS")
        report_lines.append(f"{'-' * 50}")
        
        if completed_stages > 0:
            # Find bottleneck stage
            stage_durations = {
                metrics['stage_name']: metrics.get('duration', 0)
                for metrics in stage_names.values()
                if metrics['status'] == 'completed'
            }
            
            if stage_durations:
                slowest_stage = max(stage_durations.items(), key=lambda x: x[1])
                fastest_stage = min(stage_durations.items(), key=lambda x: x[1])
                
                report_lines.append(f"🐌 Slowest stage: {slowest_stage[0]} ({slowest_stage[1]:.2f}s)")
                report_lines.append(f"⚡ Fastest stage: {fastest_stage[0]} ({fastest_stage[1]:.2f}s)")
            
            # Memory usage insights
            memory_peaks = {
                metrics['stage_name']: metrics.get('peak_memory', 0)
                for metrics in stage_names.values()
                if metrics['status'] == 'completed'
            }
            
            if memory_peaks:
                highest_memory_stage = max(memory_peaks.items(), key=lambda x: x[1])
                report_lines.append(f"💾 Highest memory usage: {highest_memory_stage[0]} ({highest_memory_stage[1]:.1f} MB)")
        
        # Recommendations
        report_lines.append(f"\n{'-' * 50}")
        report_lines.append("RECOMMENDATIONS")
        report_lines.append(f"{'-' * 50}")
        
        recommendations = []
        
        # Check for long-running stages
        for stage_name, metrics in stage_names.items():
            duration = metrics.get('duration', 0)
            if duration > 300:  # 5 minutes
                recommendations.append(f"⏱️  {stage_name} stage is taking a long time ({duration:.1f}s). Consider optimization.")
        
        # Check for high memory usage
        for stage_name, metrics in stage_names.items():
            peak_memory = metrics.get('peak_memory', 0)
            if peak_memory > 1000:  # 1GB
                recommendations.append(f"🧠 {stage_name} stage uses high memory ({peak_memory:.1f} MB). Monitor for memory leaks.")
        
        # Check for resource-intensive stages
        for stage_name, metrics in stage_names.items():
            resource_stats = metrics.get('resource_statistics', {})
            cpu_stats = resource_stats.get('cpu', {})
            avg_cpu = cpu_stats.get('mean', 0)
            
            if avg_cpu > 80:
                recommendations.append(f"⚙️  {stage_name} stage is CPU intensive ({avg_cpu:.1f}%). Consider parallel processing.")
        
        if recommendations:
            for rec in recommendations:
                report_lines.append(rec)
        else:
            report_lines.append("✅ No performance issues detected. Pipeline is running efficiently!")
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def export_metrics(self, output_path: str):
        """Export metrics to JSON file"""
        import json
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.stage_metrics, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {str(e)}")
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        self.stage_metrics.clear()
        self.current_stages.clear()
        self.resource_samples.clear()
        self.monitoring_active.clear()
        
        self.logger.info("All metrics reset")


if __name__ == "__main__":
    # Example usage
    monitor = PipelineMonitor()
    
    # Simulate stage monitoring
    monitor.start_stage("test_stage")
    time.sleep(2)  # Simulate work
    duration = monitor.end_stage("test_stage")
    
    print(f"Stage completed in {duration:.2f} seconds")
    print("\nPerformance Report:")
    print(monitor.generate_performance_report())
