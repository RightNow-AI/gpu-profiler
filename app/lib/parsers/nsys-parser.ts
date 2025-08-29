import { KernelLaunch, MemoryTransfer, GPUMetrics, ProfilingSession, ParsedProfile, BottleneckHint } from '../../types/profiling';

export class NsysParser {
  static async parse(content: ArrayBuffer): Promise<ParsedProfile> {
    // In a real implementation, this would parse the binary .nsys-rep format
    // For demo purposes, we'll create mock data with nsys-specific characteristics
    
    const session: ProfilingSession = {
      id: `nsys_${Date.now()}`,
      name: 'Nsight Systems Profile',
      device: 'NVIDIA RTX A6000',
      driverVersion: '546.33',
      cudaVersion: '12.3',
      totalDuration: 2200000000, // 2.2 seconds
      kernelLaunches: this.generateMockKernels(),
      memoryTransfers: this.generateMockMemoryTransfers(),
      metrics: this.generateMockMetrics(),
      createdAt: new Date()
    };

    const hints = this.generateBottleneckHints(session);

    return { session, hints };
  }

  private static generateMockKernels(): KernelLaunch[] {
    const kernels: KernelLaunch[] = [];
    let currentTime = 0;

    // Nsight Systems typically shows more detailed kernel information
    const kernelPatterns = [
      { 
        name: 'ampere_sgemm_128x128_tn', 
        avgDuration: 125000000, 
        occupancy: 0.94,
        category: 'cublas'
      },
      { 
        name: 'volta_scudnn_128x128_relu_interior_nn', 
        avgDuration: 89000000, 
        occupancy: 0.87,
        category: 'cudnn'
      },
      { 
        name: 'maxwell_scudnn_winograd_128x128_ldg1_ldg4', 
        avgDuration: 156000000, 
        occupancy: 0.76,
        category: 'cudnn'
      },
      { 
        name: 'custom_fused_attention_kernel', 
        avgDuration: 67000000, 
        occupancy: 0.82,
        category: 'custom'
      },
      { 
        name: 'ampere_fp16_s884gemm_fp16_256x128_ldg8', 
        avgDuration: 198000000, 
        occupancy: 0.91,
        category: 'cublas'
      }
    ];

    // Generate more complex execution pattern
    for (let batch = 0; batch < 8; batch++) {
      for (let i = 0; i < kernelPatterns.length; i++) {
        const pattern = kernelPatterns[i];
        const variance = 0.85 + Math.random() * 0.3;
        const duration = Math.floor(pattern.avgDuration * variance);
        
        kernels.push({
          id: `nsys_kernel_${batch}_${i}`,
          name: pattern.name,
          startTime: currentTime,
          duration,
          streamId: (batch * kernelPatterns.length + i) % 8, // 8 streams
          blockDim: this.getBlockDimForKernel(pattern.name),
          gridDim: this.getGridDimForKernel(pattern.name),
          sharedMemory: this.getSharedMemoryUsage(pattern.category),
          registers: 24 + (i % 20) * 2,
          occupancy: pattern.occupancy * (0.92 + Math.random() * 0.16),
          memoryThroughput: 200 + Math.random() * 400,
          computeThroughput: 12e12 + Math.random() * 8e12
        });

        currentTime += duration + Math.floor(Math.random() * 8000000);
        
        // Add some overlap between streams
        if (i % 3 === 0) {
          currentTime -= Math.floor(duration * 0.3);
        }
      }
      
      // Batch gaps
      currentTime += Math.floor(Math.random() * 50000000);
    }

    return kernels.sort((a, b) => a.startTime - b.startTime);
  }

  private static getBlockDimForKernel(kernelName: string) {
    if (kernelName.includes('128x128')) return { x: 128, y: 128, z: 1 };
    if (kernelName.includes('256x128')) return { x: 256, y: 128, z: 1 };
    if (kernelName.includes('sgemm')) return { x: 16, y: 16, z: 1 };
    return { x: 256, y: 1, z: 1 };
  }

  private static getGridDimForKernel(kernelName: string) {
    const baseGrid = Math.ceil(4096 / 256);
    if (kernelName.includes('gemm')) return { x: baseGrid * 2, y: baseGrid * 2, z: 1 };
    if (kernelName.includes('conv')) return { x: baseGrid, y: baseGrid * 4, z: 1 };
    return { x: baseGrid, y: 1, z: 1 };
  }

  private static getSharedMemoryUsage(category: string): number {
    switch (category) {
      case 'cublas': return 32768 + Math.random() * 16384;
      case 'cudnn': return 16384 + Math.random() * 32768;
      case 'custom': return 8192 + Math.random() * 24576;
      default: return 4096 + Math.random() * 8192;
    }
  }

  private static generateMockMemoryTransfers(): MemoryTransfer[] {
    const transfers: MemoryTransfer[] = [];
    let currentTime = 0;

    // Initial setup transfers
    const setupTransfers = [
      { size: 512, duration: 45000000 },
      { size: 1024, duration: 89000000 },
      { size: 256, duration: 28000000 }
    ];

    setupTransfers.forEach((transfer, i) => {
      transfers.push({
        id: `nsys_h2d_setup_${i}`,
        type: 'H2D',
        startTime: currentTime,
        duration: transfer.duration,
        size: transfer.size * 1024 * 1024,
        bandwidth: (transfer.size * 1024 * 1024) / (transfer.duration / 1e9) / 1e9
      });
      currentTime += transfer.duration + 5000000;
    });

    // Intermediate transfers during computation
    const midTime = 1000000000;
    for (let i = 0; i < 4; i++) {
      transfers.push({
        id: `nsys_d2d_${i}`,
        type: 'D2D',
        startTime: midTime + i * 150000000,
        duration: 12000000 + Math.random() * 8000000,
        size: (64 + Math.random() * 128) * 1024 * 1024,
        bandwidth: 400 + Math.random() * 200 // Higher D2D bandwidth
      });
    }

    // Final result transfers
    const endTime = 2000000000;
    const resultTransfers = [
      { size: 128, duration: 18000000 },
      { size: 64, duration: 12000000 }
    ];

    resultTransfers.forEach((transfer, i) => {
      transfers.push({
        id: `nsys_d2h_result_${i}`,
        type: 'D2H',
        startTime: endTime + i * 25000000,
        duration: transfer.duration,
        size: transfer.size * 1024 * 1024,
        bandwidth: (transfer.size * 1024 * 1024) / (transfer.duration / 1e9) / 1e9
      });
    });

    return transfers;
  }

  private static generateMockMetrics(): GPUMetrics[] {
    const metrics: GPUMetrics[] = [];
    const duration = 2200; // 2.2 seconds
    const samplesPerSecond = 20; // Higher sampling rate for Nsight Systems

    for (let i = 0; i < duration * samplesPerSecond; i++) {
      const timestamp = (i / samplesPerSecond) * 1000000000;
      const time_s = i / samplesPerSecond;
      
      // More complex utilization pattern
      let utilization = 0.3; // Base utilization
      
      // High utilization during computation phases
      if (time_s > 0.5 && time_s < 2.0) {
        utilization = 0.85 + Math.sin(time_s * 2) * 0.1 + Math.random() * 0.05;
      }
      
      // Spikes during kernel launches
      if (Math.sin(time_s * 15) > 0.8) {
        utilization += 0.1;
      }
      
      utilization = Math.max(0, Math.min(1, utilization));
      
      metrics.push({
        timestamp,
        utilization,
        memoryUtilization: 0.6 + Math.sin(time_s * 0.5) * 0.2 + Math.random() * 0.1,
        temperature: 72 + Math.sin(time_s * 0.1) * 8 + Math.random() * 5,
        powerDraw: 320 + utilization * 80 + Math.random() * 30
      });
    }

    return metrics;
  }

  private static generateBottleneckHints(session: ProfilingSession): BottleneckHint[] {
    const hints: BottleneckHint[] = [];
    const { kernelLaunches, memoryTransfers, metrics } = session;

    // Analyze kernel execution patterns
    const totalKernelTime = kernelLaunches.reduce((sum, k) => sum + k.duration, 0);
    const totalSessionTime = session.totalDuration;
    const gpuUtilization = totalKernelTime / totalSessionTime;

    if (gpuUtilization < 0.6) {
      hints.push({
        type: 'occupancy',
        severity: 'high',
        title: 'Low Overall GPU Utilization',
        description: `GPU utilization is ${(gpuUtilization * 100).toFixed(1)}% - significant idle time detected`,
        suggestion: 'Consider increasing parallelism or overlapping computation with memory transfers'
      });
    }

    // Check for memory transfer bottlenecks
    const h2dTransfers = memoryTransfers.filter(t => t.type === 'H2D');
    const avgH2DBandwidth = h2dTransfers.reduce((sum, t) => sum + t.bandwidth, 0) / h2dTransfers.length;
    
    if (avgH2DBandwidth < 8) { // Below expected PCIe bandwidth
      hints.push({
        type: 'memory',
        severity: 'medium',
        title: 'Suboptimal Memory Transfer Performance',
        description: `Average H2D bandwidth is ${avgH2DBandwidth.toFixed(1)} GB/s - below expected PCIe performance`,
        suggestion: 'Use pinned memory and larger transfer sizes for better bandwidth utilization'
      });
    }

    // Analyze stream utilization
    const streamUsage = new Map<number, number>();
    kernelLaunches.forEach(k => {
      streamUsage.set(k.streamId, (streamUsage.get(k.streamId) || 0) + 1);
    });

    const streamsUsed = streamUsage.size;
    const maxStreamsRecommended = 8;
    
    if (streamsUsed < 3) {
      hints.push({
        type: 'sync',
        severity: 'low',
        title: 'Limited Stream Parallelism',
        description: `Only ${streamsUsed} CUDA streams used - potential for better overlap`,
        suggestion: 'Use multiple streams to overlap kernel execution and memory transfers'
      });
    }

    // Check for divergence patterns (simplified heuristic)
    const complexKernels = kernelLaunches.filter(k => 
      k.name.includes('winograd') || k.name.includes('attention') || k.occupancy! < 0.7
    );
    
    if (complexKernels.length > 0) {
      hints.push({
        type: 'divergence',
        severity: 'medium',
        title: 'Potential Warp Divergence',
        description: `${complexKernels.length} kernels show signs of complex control flow`,
        suggestion: 'Profile with compute sanitizer to identify warp divergence patterns',
        kernelIds: complexKernels.map(k => k.id)
      });
    }

    return hints;
  }
}