import { KernelLaunch, MemoryTransfer, GPUMetrics, ProfilingSession, ParsedProfile, BottleneckHint } from '../../types/profiling';

interface JsonProfileData {
  traceEvents?: any[];
  otherData?: any[];
  displayTimeUnit?: string;
  [key: string]: any;
}

export class JsonParser {
  static async parse(content: string): Promise<ParsedProfile> {
    try {
      const data: JsonProfileData = JSON.parse(content);
      
      // Handle different JSON formats (Chrome tracing, NVIDIA tools export, custom)
      if (data.traceEvents) {
        return this.parseChromeTracingFormat(data);
      } else if (data.otherData || Array.isArray(data)) {
        return this.parseCustomFormat(data);
      } else {
        // Fallback to mock data with JSON-specific characteristics
        return this.generateMockProfile(data);
      }
    } catch (error) {
      throw new Error(`Invalid JSON format: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private static parseChromeTracingFormat(data: JsonProfileData): ParsedProfile {
    const events = data.traceEvents || [];
    const kernels: KernelLaunch[] = [];
    const transfers: MemoryTransfer[] = [];
    const metrics: GPUMetrics[] = [];
    
    let minTime = Number.MAX_VALUE;
    let maxTime = 0;

    // Parse trace events
    events.forEach((event, index) => {
      const startTime = (event.ts || 0) * 1000; // Convert microseconds to nanoseconds
      const duration = (event.dur || 1000) * 1000; // Convert to nanoseconds
      
      minTime = Math.min(minTime, startTime);
      maxTime = Math.max(maxTime, startTime + duration);

      if (event.cat === 'cuda_kernel' || event.name?.includes('kernel')) {
        kernels.push({
          id: `json_kernel_${index}`,
          name: event.name || `unknown_kernel_${index}`,
          startTime,
          duration,
          streamId: event.args?.stream || 0,
          blockDim: event.args?.blockDim || { x: 256, y: 1, z: 1 },
          gridDim: event.args?.gridDim || { x: 64, y: 1, z: 1 },
          sharedMemory: event.args?.sharedMemory || 4096,
          registers: event.args?.registers || 32,
          occupancy: event.args?.occupancy || (0.6 + Math.random() * 0.3),
          memoryThroughput: event.args?.memoryThroughput || (100 + Math.random() * 200),
          computeThroughput: event.args?.computeThroughput || (5e12 + Math.random() * 5e12)
        });
      } else if (event.cat === 'cuda_memory' || event.name?.includes('memcpy')) {
        const size = event.args?.size || (1024 * 1024 * (1 + Math.random() * 99));
        transfers.push({
          id: `json_transfer_${index}`,
          type: this.inferTransferType(event.name || ''),
          startTime,
          duration,
          size,
          bandwidth: size / (duration / 1e9) / 1e9 // GB/s
        });
      }
    });

    // Generate mock metrics if not present in trace
    if (minTime !== Number.MAX_VALUE && maxTime > minTime) {
      const totalDuration = maxTime - minTime;
      const samplesCount = Math.min(1000, Math.floor(totalDuration / 10000000)); // Sample every 10ms
      
      for (let i = 0; i < samplesCount; i++) {
        const timestamp = minTime + (i / samplesCount) * totalDuration;
        metrics.push({
          timestamp,
          utilization: this.calculateUtilizationAtTime(timestamp, kernels),
          memoryUtilization: 0.4 + Math.random() * 0.4,
          temperature: 68 + Math.random() * 12,
          powerDraw: 250 + Math.random() * 150
        });
      }
    }

    const session: ProfilingSession = {
      id: `json_${Date.now()}`,
      name: 'JSON Trace Profile',
      device: data.device || 'Unknown GPU Device',
      driverVersion: data.driverVersion,
      cudaVersion: data.cudaVersion,
      totalDuration: maxTime - minTime,
      kernelLaunches: kernels,
      memoryTransfers: transfers,
      metrics,
      createdAt: new Date()
    };

    const hints = this.generateBottleneckHints(session);
    return { session, hints };
  }

  private static parseCustomFormat(data: any): ParsedProfile {
    // Handle custom JSON formats
    const session: ProfilingSession = {
      id: `json_custom_${Date.now()}`,
      name: 'Custom JSON Profile',
      device: data.device || 'Unknown GPU Device',
      driverVersion: data.driverVersion,
      cudaVersion: data.cudaVersion,
      totalDuration: data.totalDuration || 1000000000,
      kernelLaunches: this.parseCustomKernels(data.kernels || data.events || []),
      memoryTransfers: this.parseCustomTransfers(data.transfers || data.memcopy || []),
      metrics: this.parseCustomMetrics(data.metrics || data.utilization || []),
      createdAt: new Date()
    };

    const hints = this.generateBottleneckHints(session);
    return { session, hints };
  }

  private static generateMockProfile(data: any): ParsedProfile {
    const session: ProfilingSession = {
      id: `json_mock_${Date.now()}`,
      name: 'JSON Profile (Mock Data)',
      device: 'NVIDIA RTX 3080',
      driverVersion: '546.33',
      cudaVersion: '12.3',
      totalDuration: 1800000000, // 1.8 seconds
      kernelLaunches: this.generateMockKernels(),
      memoryTransfers: this.generateMockTransfers(),
      metrics: this.generateMockMetrics(),
      createdAt: new Date()
    };

    const hints = this.generateBottleneckHints(session);
    return { session, hints };
  }

  private static parseCustomKernels(kernelsData: any[]): KernelLaunch[] {
    return kernelsData.map((kernel, index) => ({
      id: kernel.id || `custom_kernel_${index}`,
      name: kernel.name || `kernel_${index}`,
      startTime: kernel.startTime || kernel.start || 0,
      duration: kernel.duration || kernel.dur || 1000000,
      streamId: kernel.streamId || kernel.stream || 0,
      blockDim: kernel.blockDim || { x: 256, y: 1, z: 1 },
      gridDim: kernel.gridDim || { x: 64, y: 1, z: 1 },
      sharedMemory: kernel.sharedMemory || kernel.smem || 4096,
      registers: kernel.registers || kernel.regs || 32,
      occupancy: kernel.occupancy || (0.5 + Math.random() * 0.4),
      memoryThroughput: kernel.memoryThroughput || (80 + Math.random() * 160),
      computeThroughput: kernel.computeThroughput || (4e12 + Math.random() * 6e12)
    }));
  }

  private static parseCustomTransfers(transfersData: any[]): MemoryTransfer[] {
    return transfersData.map((transfer, index) => ({
      id: transfer.id || `custom_transfer_${index}`,
      type: transfer.type || 'H2D',
      startTime: transfer.startTime || transfer.start || 0,
      duration: transfer.duration || transfer.dur || 1000000,
      size: transfer.size || (1024 * 1024),
      bandwidth: transfer.bandwidth || 10
    }));
  }

  private static parseCustomMetrics(metricsData: any[]): GPUMetrics[] {
    return metricsData.map(metric => ({
      timestamp: metric.timestamp || metric.time || 0,
      utilization: metric.utilization || metric.util || (0.3 + Math.random() * 0.5),
      memoryUtilization: metric.memoryUtilization || metric.memUtil || (0.2 + Math.random() * 0.4),
      temperature: metric.temperature || metric.temp || (60 + Math.random() * 20),
      powerDraw: metric.powerDraw || metric.power || (200 + Math.random() * 100)
    }));
  }

  private static inferTransferType(name: string): 'H2D' | 'D2H' | 'D2D' {
    if (name.includes('H2D') || name.includes('HtoD') || name.includes('upload')) return 'H2D';
    if (name.includes('D2H') || name.includes('DtoH') || name.includes('download')) return 'D2H';
    return 'D2D';
  }

  private static calculateUtilizationAtTime(timestamp: number, kernels: KernelLaunch[]): number {
    const activeKernels = kernels.filter(k => 
      timestamp >= k.startTime && timestamp < (k.startTime + k.duration)
    );
    
    if (activeKernels.length === 0) return 0;
    
    // Simplified utilization calculation
    const avgOccupancy = activeKernels.reduce((sum, k) => sum + (k.occupancy || 0.5), 0) / activeKernels.length;
    return Math.min(1, avgOccupancy * activeKernels.length / 8); // Assume 8 max concurrent kernels
  }

  private static generateMockKernels(): KernelLaunch[] {
    const kernels: KernelLaunch[] = [];
    let currentTime = 100000000; // Start after 100ms
    
    const jsonKernelTypes = [
      { name: 'json_exported_kernel_0', duration: 35000000, occupancy: 0.78 },
      { name: 'elementwise_add', duration: 8000000, occupancy: 0.92 },
      { name: 'batch_norm_fwd', duration: 22000000, occupancy: 0.84 },
      { name: 'softmax_kernel', duration: 18000000, occupancy: 0.66 },
      { name: 'lstm_cell_forward', duration: 95000000, occupancy: 0.57 }
    ];

    for (let i = 0; i < 20; i++) {
      const kernelType = jsonKernelTypes[i % jsonKernelTypes.length];
      const variance = 0.8 + Math.random() * 0.4;
      
      kernels.push({
        id: `json_kernel_${i}`,
        name: kernelType.name,
        startTime: currentTime,
        duration: Math.floor(kernelType.duration * variance),
        streamId: i % 6,
        blockDim: { x: 128 + (i % 3) * 64, y: 1, z: 1 },
        gridDim: { x: 32 + (i % 4) * 16, y: 1, z: 1 },
        sharedMemory: 2048 * (1 + i % 4),
        registers: 28 + (i % 12) * 2,
        occupancy: kernelType.occupancy * (0.9 + Math.random() * 0.2),
        memoryThroughput: 120 + Math.random() * 180,
        computeThroughput: 6e12 + Math.random() * 4e12
      });

      currentTime += Math.floor(kernelType.duration * variance) + Math.floor(Math.random() * 10000000);
    }

    return kernels;
  }

  private static generateMockTransfers(): MemoryTransfer[] {
    return [
      {
        id: 'json_h2d_0',
        type: 'H2D',
        startTime: 50000000,
        duration: 35000000,
        size: 128 * 1024 * 1024,
        bandwidth: 11.5
      },
      {
        id: 'json_d2h_0',
        type: 'D2H',
        startTime: 1650000000,
        duration: 22000000,
        size: 64 * 1024 * 1024,
        bandwidth: 9.8
      }
    ];
  }

  private static generateMockMetrics(): GPUMetrics[] {
    const metrics: GPUMetrics[] = [];
    const duration = 1800; // 1.8 seconds
    const samplesPerSecond = 5;

    for (let i = 0; i < duration * samplesPerSecond; i++) {
      const timestamp = (i / samplesPerSecond) * 1000000000;
      
      metrics.push({
        timestamp,
        utilization: 0.4 + Math.sin(i * 0.2) * 0.3 + Math.random() * 0.1,
        memoryUtilization: 0.35 + Math.random() * 0.25,
        temperature: 66 + Math.random() * 10,
        powerDraw: 260 + Math.random() * 80
      });
    }

    return metrics;
  }

  private static generateBottleneckHints(session: ProfilingSession): BottleneckHint[] {
    const hints: BottleneckHint[] = [];

    // JSON-specific analysis
    const totalKernelTime = session.kernelLaunches.reduce((sum, k) => sum + k.duration, 0);
    const avgOccupancy = session.kernelLaunches.reduce((sum, k) => sum + (k.occupancy || 0), 0) / session.kernelLaunches.length;

    if (avgOccupancy < 0.65) {
      hints.push({
        type: 'occupancy',
        severity: 'medium',
        title: 'Suboptimal GPU Occupancy',
        description: `Average occupancy is ${(avgOccupancy * 100).toFixed(1)}% across all kernels`,
        suggestion: 'Review kernel launch configurations and memory usage patterns'
      });
    }

    // Check for unbalanced streams
    const streamDistribution = new Map<number, number>();
    session.kernelLaunches.forEach(k => {
      streamDistribution.set(k.streamId, (streamDistribution.get(k.streamId) || 0) + k.duration);
    });

    const streamTimes = Array.from(streamDistribution.values());
    const maxStreamTime = Math.max(...streamTimes);
    const minStreamTime = Math.min(...streamTimes);
    const imbalanceRatio = maxStreamTime / minStreamTime;

    if (imbalanceRatio > 3 && streamTimes.length > 1) {
      hints.push({
        type: 'sync',
        severity: 'low',
        title: 'Stream Load Imbalance',
        description: `Stream workloads vary by ${imbalanceRatio.toFixed(1)}x - may indicate synchronization issues`,
        suggestion: 'Balance work distribution across streams or use dynamic scheduling'
      });
    }

    return hints;
  }
}