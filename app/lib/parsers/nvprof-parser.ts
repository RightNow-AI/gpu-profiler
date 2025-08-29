import { KernelLaunch, MemoryTransfer, GPUMetrics, ProfilingSession, ParsedProfile, BottleneckHint } from '../../types/profiling';

export class NvprofParser {
  static async parse(content: ArrayBuffer): Promise<ParsedProfile> {
    // In a real implementation, this would parse the binary .nvprof format
    // For demo purposes, we'll create mock data
    
    const session: ProfilingSession = {
      id: `nvprof_${Date.now()}`,
      name: 'NVPROF Profile Session',
      device: 'NVIDIA GeForce RTX 4090',
      driverVersion: '546.33',
      cudaVersion: '12.3',
      totalDuration: 1500000000, // 1.5 seconds in nanoseconds
      kernelLaunches: this.generateMockKernels(),
      memoryTransfers: this.generateMockMemoryTransfers(),
      metrics: this.generateMockMetrics(),
      createdAt: new Date()
    };

    const hints = this.generateBottleneckHints(session.kernelLaunches);

    return { session, hints };
  }

  private static generateMockKernels(): KernelLaunch[] {
    const kernels: KernelLaunch[] = [];
    let currentTime = 0;

    // Simulate different kernel patterns
    const kernelTypes = [
      { name: 'matrixMul', avgDuration: 45000000, occupancy: 0.85 },
      { name: 'vectorAdd', avgDuration: 12000000, occupancy: 0.92 },
      { name: 'reduction', avgDuration: 8500000, occupancy: 0.67 },
      { name: 'convolution2D', avgDuration: 78000000, occupancy: 0.43 },
      { name: 'transposeMatrix', avgDuration: 23000000, occupancy: 0.71 }
    ];

    for (let i = 0; i < 25; i++) {
      const kernelType = kernelTypes[i % kernelTypes.length];
      const variance = 0.8 + Math.random() * 0.4; // ±20% variance
      const duration = Math.floor(kernelType.avgDuration * variance);
      
      kernels.push({
        id: `kernel_${i}`,
        name: `${kernelType.name}_${Math.floor(i / kernelTypes.length)}`,
        startTime: currentTime,
        duration,
        streamId: i % 4, // 4 streams
        blockDim: { x: 256, y: 1, z: 1 },
        gridDim: { x: Math.ceil(1024 / 256), y: 1, z: 1 },
        sharedMemory: 1024 * (1 + i % 3),
        registers: 32 + (i % 16),
        occupancy: kernelType.occupancy * (0.9 + Math.random() * 0.2),
        memoryThroughput: 150 + Math.random() * 300, // GB/s
        computeThroughput: 8.5e12 + Math.random() * 4e12 // FLOPS
      });

      currentTime += duration + Math.floor(Math.random() * 5000000); // Small gaps
    }

    return kernels;
  }

  private static generateMockMemoryTransfers(): MemoryTransfer[] {
    const transfers: MemoryTransfer[] = [];
    let currentTime = 0;

    // Initial data uploads
    for (let i = 0; i < 3; i++) {
      transfers.push({
        id: `h2d_${i}`,
        type: 'H2D',
        startTime: currentTime,
        duration: 25000000 + Math.random() * 15000000,
        size: (64 + Math.random() * 192) * 1024 * 1024, // 64-256MB
        bandwidth: 12 + Math.random() * 8 // GB/s
      });
      currentTime += 30000000;
    }

    // Result downloads
    currentTime = 1400000000; // Near end
    for (let i = 0; i < 2; i++) {
      transfers.push({
        id: `d2h_${i}`,
        type: 'D2H',
        startTime: currentTime,
        duration: 18000000 + Math.random() * 10000000,
        size: (32 + Math.random() * 96) * 1024 * 1024, // 32-128MB
        bandwidth: 11 + Math.random() * 7 // GB/s
      });
      currentTime += 25000000;
    }

    return transfers;
  }

  private static generateMockMetrics(): GPUMetrics[] {
    const metrics: GPUMetrics[] = [];
    const duration = 1500; // 1.5 seconds
    const samplesPerSecond = 10;

    for (let i = 0; i < duration * samplesPerSecond; i++) {
      const timestamp = (i / samplesPerSecond) * 1000000000; // Convert to nanoseconds
      
      // Simulate varying utilization based on kernel activity
      const baseUtilization = 0.65;
      const variation = Math.sin(i * 0.1) * 0.2 + Math.random() * 0.1;
      const utilization = Math.max(0, Math.min(1, baseUtilization + variation));
      
      metrics.push({
        timestamp,
        utilization,
        memoryUtilization: 0.4 + Math.random() * 0.3,
        temperature: 65 + Math.random() * 15,
        powerDraw: 280 + Math.random() * 120
      });
    }

    return metrics;
  }

  private static generateBottleneckHints(kernels: KernelLaunch[]): BottleneckHint[] {
    const hints: BottleneckHint[] = [];

    // Find low occupancy kernels
    const lowOccupancyKernels = kernels.filter(k => k.occupancy && k.occupancy < 0.5);
    if (lowOccupancyKernels.length > 0) {
      hints.push({
        type: 'occupancy',
        severity: 'high',
        title: 'Low GPU Occupancy Detected',
        description: `${lowOccupancyKernels.length} kernels have occupancy below 50%`,
        suggestion: 'Consider increasing block size or reducing register usage',
        kernelIds: lowOccupancyKernels.map(k => k.id)
      });
    }

    // Find memory-bound kernels
    const memoryBoundKernels = kernels.filter(k => 
      k.memoryThroughput && k.computeThroughput && 
      k.memoryThroughput / (k.computeThroughput / 1e12) > 100
    );
    if (memoryBoundKernels.length > 0) {
      hints.push({
        type: 'memory',
        severity: 'medium',
        title: 'Memory-Bound Kernels Found',
        description: `${memoryBoundKernels.length} kernels appear to be memory bandwidth limited`,
        suggestion: 'Optimize memory access patterns or use shared memory',
        kernelIds: memoryBoundKernels.map(k => k.id)
      });
    }

    // Check for inefficient block sizes
    const inefficientBlocks = kernels.filter(k => 
      k.blockDim.x * k.blockDim.y * k.blockDim.z < 128
    );
    if (inefficientBlocks.length > 0) {
      hints.push({
        type: 'compute',
        severity: 'low',
        title: 'Small Block Sizes Detected',
        description: `${inefficientBlocks.length} kernels use block sizes < 128 threads`,
        suggestion: 'Consider using larger block sizes (256+ threads per block)',
        kernelIds: inefficientBlocks.map(k => k.id)
      });
    }

    return hints;
  }
}