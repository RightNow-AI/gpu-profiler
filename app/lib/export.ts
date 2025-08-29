import { ParsedProfile } from '../types/profiling';

export interface ExportOptions {
  includeRawData?: boolean;
  includeVisualizations?: boolean;
  format?: 'json' | 'csv' | 'summary';
  compression?: boolean;
}

export class ExportService {
  static async exportAsJSON(profile: ParsedProfile, options: ExportOptions = {}): Promise<void> {
    const {
      includeRawData = true,
      format = 'json',
      compression = false
    } = options;

    const exportData: Record<string, unknown> = {
      metadata: {
        exportedAt: new Date().toISOString(),
        exportedBy: 'GPU_PROFILER_v1.0.0',
        originalSessionId: profile.session.id,
        exportFormat: format
      },
      summary: {
        sessionName: profile.session.name,
        device: profile.session.device,
        totalDuration: profile.session.totalDuration,
        kernelCount: profile.session.kernelLaunches.length,
        memoryOpCount: profile.session.memoryTransfers.length,
        issueCount: profile.hints.length
      }
    };

    if (includeRawData) {
      exportData.session = profile.session;
      exportData.hints = profile.hints;
    } else {
      // Export summary data only
      exportData.topKernels = profile.session.kernelLaunches
        .sort((a, b) => b.duration - a.duration)
        .slice(0, 20)
        .map(k => ({
          name: k.name,
          duration: k.duration,
          occupancy: k.occupancy,
          streamId: k.streamId
        }));
      
      exportData.criticalIssues = profile.hints
        .filter(h => h.severity === 'high')
        .map(h => ({
          type: h.type,
          title: h.title,
          description: h.description,
          suggestion: h.suggestion
        }));
    }

    // Generate filename
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `gpu_profile_${profile.session.id}_${timestamp}.json`;

    // Create and download file
    this.downloadFile(
      JSON.stringify(exportData, null, compression ? 0 : 2),
      filename,
      'application/json'
    );
  }

  static async exportAsCSV(profile: ParsedProfile): Promise<void> {
    // Export kernel data as CSV
    const csvData = this.convertKernelsToCSV(profile.session.kernelLaunches);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `gpu_kernels_${profile.session.id}_${timestamp}.csv`;
    
    this.downloadFile(csvData, filename, 'text/csv');
  }

  static async exportSummaryReport(profile: ParsedProfile): Promise<void> {
    const summary = this.generateTextSummary(profile);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `gpu_summary_${profile.session.id}_${timestamp}.txt`;
    
    this.downloadFile(summary, filename, 'text/plain');
  }

  static async exportVisualizationData(profile: ParsedProfile): Promise<void> {
    // Export data formatted for visualization tools (e.g., Chrome Tracing format)
    const traceData = this.convertToTraceFormat(profile);
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `gpu_trace_${profile.session.id}_${timestamp}.json`;
    
    this.downloadFile(
      JSON.stringify(traceData, null, 2),
      filename,
      'application/json'
    );
  }

  private static downloadFile(content: string, filename: string, mimeType: string): void {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Clean up the URL object
    setTimeout(() => URL.revokeObjectURL(url), 100);
  }

  private static convertKernelsToCSV(kernels: any[]): string {
    const headers = [
      'Name',
      'Duration_ns',
      'Start_Time_ns',
      'Stream_ID',
      'Block_Dim_X',
      'Block_Dim_Y',
      'Block_Dim_Z',
      'Grid_Dim_X',
      'Grid_Dim_Y',
      'Grid_Dim_Z',
      'Shared_Memory_B',
      'Registers',
      'Occupancy',
      'Memory_Throughput_GB_s',
      'Compute_Throughput_FLOPS'
    ];

    const rows = kernels.map(k => [
      `"${k.name}"`,
      k.duration,
      k.startTime,
      k.streamId,
      k.blockDim.x,
      k.blockDim.y,
      k.blockDim.z,
      k.gridDim.x,
      k.gridDim.y,
      k.gridDim.z,
      k.sharedMemory,
      k.registers,
      k.occupancy?.toFixed(4) || '',
      k.memoryThroughput?.toFixed(2) || '',
      k.computeThroughput?.toExponential(2) || ''
    ]);

    return [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
  }

  private static generateTextSummary(profile: ParsedProfile): string {
    const { session, hints } = profile;
    const totalKernels = session.kernelLaunches.length;
    const avgOccupancy = session.kernelLaunches.reduce((sum, k) => sum + (k.occupancy || 0), 0) / totalKernels;
    const totalDuration = session.totalDuration / 1000000000; // Convert to seconds
    
    const highIssues = hints.filter(h => h.severity === 'high');
    const mediumIssues = hints.filter(h => h.severity === 'medium');
    const lowIssues = hints.filter(h => h.severity === 'low');

    let summary = `GPU PROFILING SUMMARY REPORT
==========================================

SESSION INFORMATION
-------------------
Name: ${session.name}
Device: ${session.device}
Driver Version: ${session.driverVersion || 'Unknown'}
CUDA Version: ${session.cudaVersion || 'Unknown'}
Profile Date: ${session.createdAt.toISOString()}
Total Duration: ${totalDuration.toFixed(3)} seconds

KERNEL EXECUTION STATISTICS
---------------------------
Total Kernels: ${totalKernels}
Average Occupancy: ${(avgOccupancy * 100).toFixed(1)}%
Memory Operations: ${session.memoryTransfers.length}

TOP 10 KERNELS BY DURATION
--------------------------
`;

    const topKernels = session.kernelLaunches
      .sort((a, b) => b.duration - a.duration)
      .slice(0, 10);

    topKernels.forEach((kernel, i) => {
      const durationMs = (kernel.duration / 1000000).toFixed(3);
      const occupancy = ((kernel.occupancy || 0) * 100).toFixed(1);
      summary += `${i + 1}. ${kernel.name}\n   Duration: ${durationMs}ms, Occupancy: ${occupancy}%, Stream: ${kernel.streamId}\n\n`;
    });

    if (hints.length > 0) {
      summary += `PERFORMANCE ISSUES DETECTED
---------------------------
High Priority Issues: ${highIssues.length}
Medium Priority Issues: ${mediumIssues.length}
Low Priority Issues: ${lowIssues.length}

DETAILED ISSUES
--------------
`;

      hints.forEach((hint, i) => {
        summary += `${i + 1}. [${hint.severity.toUpperCase()}] ${hint.title}\n`;
        summary += `   Type: ${hint.type}\n`;
        summary += `   Description: ${hint.description}\n`;
        summary += `   Recommendation: ${hint.suggestion}\n`;
        if (hint.kernelIds && hint.kernelIds.length > 0) {
          summary += `   Affected Kernels: ${hint.kernelIds.length}\n`;
        }
        summary += `\n`;
      });
    } else {
      summary += `PERFORMANCE ANALYSIS
------------------
✓ No significant performance issues detected
✓ GPU utilization appears optimal
✓ Kernel configurations look good

`;
    }

    summary += `
RECOMMENDATIONS
--------------
`;

    if (avgOccupancy < 0.5) {
      summary += `• Consider optimizing kernel launch parameters to improve occupancy\n`;
    }
    if (hints.length === 0) {
      summary += `• Profile looks well-optimized!\n`;
      summary += `• Consider testing with different workload sizes\n`;
    }

    summary += `
Generated by GPU Profiler (RightNow AI)
Report Date: ${new Date().toISOString()}
`;

    return summary;
  }

  private static convertToTraceFormat(profile: ParsedProfile): any {
    // Convert to Chrome Tracing format for use in chrome://tracing or other tools
    const events: any[] = [];

    // Add kernel events
    profile.session.kernelLaunches.forEach(kernel => {
      events.push({
        name: kernel.name,
        cat: 'cuda_kernel',
        ph: 'X', // Complete event
        ts: kernel.startTime / 1000, // Convert ns to μs
        dur: kernel.duration / 1000, // Convert ns to μs
        pid: 1,
        tid: kernel.streamId,
        args: {
          occupancy: kernel.occupancy,
          blockDim: kernel.blockDim,
          gridDim: kernel.gridDim,
          sharedMemory: kernel.sharedMemory,
          registers: kernel.registers
        }
      });
    });

    // Add memory transfer events
    profile.session.memoryTransfers.forEach(transfer => {
      events.push({
        name: `${transfer.type}_Transfer`,
        cat: 'cuda_memory',
        ph: 'X',
        ts: transfer.startTime / 1000,
        dur: transfer.duration / 1000,
        pid: 1,
        tid: 999, // Special thread for memory operations
        args: {
          size: transfer.size,
          bandwidth: transfer.bandwidth,
          type: transfer.type
        }
      });
    });

    return {
      displayTimeUnit: 'ns',
      traceEvents: events,
      metadata: {
        'gpu-profiler-version': '1.0.0',
        'original-session-id': profile.session.id,
        'device': profile.session.device
      }
    };
  }
}