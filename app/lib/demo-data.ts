import { ParsedProfile, KernelLaunch, MemoryTransfer, GPUMetrics, ProfilingSession, BottleneckHint } from '../types/profiling';

// Realistic demo data based on actual CUDA profiling patterns
export class DemoDataGenerator {
  static generateMatrixMultiplicationProfile(): ParsedProfile {
    const sessionId = `demo_matmul_${Date.now()}`;
    
    // Simulate matrix multiplication kernels with realistic patterns
    const kernels: KernelLaunch[] = [
      // cuBLAS SGEMM kernels (most optimized)
      {
        id: 'demo_k0',
        name: 'ampere_sgemm_128x128_tn',
        startTime: 150000000, // 150ms
        duration: 85000000, // 85ms - long duration for large matrix
        streamId: 0,
        blockDim: { x: 128, y: 128, z: 1 },
        gridDim: { x: 64, y: 64, z: 1 },
        sharedMemory: 32768,
        registers: 64,
        occupancy: 0.94, // Very high occupancy for cuBLAS
        memoryThroughput: 890.5, // GB/s
        computeThroughput: 15.2e12 // FLOPS
      },
      {
        id: 'demo_k1', 
        name: 'ampere_sgemm_256x128_tn',
        startTime: 240000000,
        duration: 75000000,
        streamId: 1,
        blockDim: { x: 256, y: 128, z: 1 },
        gridDim: { x: 32, y: 64, z: 1 },
        sharedMemory: 49152,
        registers: 72,
        occupancy: 0.91,
        memoryThroughput: 845.2,
        computeThroughput: 14.8e12
      },
      // Custom naive implementation (for comparison)
      {
        id: 'demo_k2',
        name: 'naive_matrix_multiply_kernel',
        startTime: 320000000,
        duration: 156000000, // Much slower than cuBLAS
        streamId: 0,
        blockDim: { x: 16, y: 16, z: 1 },
        gridDim: { x: 256, y: 256, z: 1 },
        sharedMemory: 4096,
        registers: 32,
        occupancy: 0.45, // Poor occupancy
        memoryThroughput: 234.1,
        computeThroughput: 3.2e12
      },
      // Preprocessing kernels
      {
        id: 'demo_k3',
        name: 'transpose_matrix_kernel',
        startTime: 50000000,
        duration: 12000000,
        streamId: 2,
        blockDim: { x: 32, y: 32, z: 1 },
        gridDim: { x: 128, y: 128, z: 1 },
        sharedMemory: 8192,
        registers: 24,
        occupancy: 0.78,
        memoryThroughput: 445.6,
        computeThroughput: 1.8e12
      },
      {
        id: 'demo_k4',
        name: 'elementwise_add_f32',
        startTime: 480000000,
        duration: 2400000,
        streamId: 1,
        blockDim: { x: 256, y: 1, z: 1 },
        gridDim: { x: 16384, y: 1, z: 1 },
        sharedMemory: 0,
        registers: 16,
        occupancy: 0.89,
        memoryThroughput: 456.7,
        computeThroughput: 0.8e12
      }
    ];

    // Memory transfers typical for matrix operations
    const memoryTransfers: MemoryTransfer[] = [
      // Input matrices upload
      {
        id: 'demo_h2d_0',
        type: 'H2D',
        startTime: 10000000,
        duration: 35000000, // 35ms for 256MB
        size: 256 * 1024 * 1024, // 256MB matrix A
        bandwidth: 7.3 // GB/s (typical PCIe)
      },
      {
        id: 'demo_h2d_1', 
        type: 'H2D',
        startTime: 20000000,
        duration: 35000000,
        size: 256 * 1024 * 1024, // 256MB matrix B
        bandwidth: 7.3
      },
      // Result download
      {
        id: 'demo_d2h_0',
        type: 'D2H', 
        startTime: 500000000,
        duration: 35000000,
        size: 256 * 1024 * 1024, // 256MB result matrix
        bandwidth: 7.3
      }
    ];

    // GPU metrics showing utilization spikes during computation
    const metrics: GPUMetrics[] = [];
    const totalDuration = 550; // 550ms total
    for (let i = 0; i < totalDuration * 2; i++) { // 2 samples per ms
      const timeMs = i / 2;
      const timestamp = timeMs * 1000000; // Convert to nanoseconds
      
      let utilization = 0.1; // Base utilization
      let memUtil = 0.3;
      let temperature = 65;
      let power = 180;

      // High utilization during SGEMM operations
      if (timeMs >= 150 && timeMs <= 235) { // First SGEMM
        utilization = 0.95 + Math.random() * 0.05;
        memUtil = 0.85 + Math.random() * 0.10;
        temperature = 78 + Math.random() * 5;
        power = 420 + Math.random() * 30;
      } else if (timeMs >= 240 && timeMs <= 315) { // Second SGEMM  
        utilization = 0.92 + Math.random() * 0.06;
        memUtil = 0.82 + Math.random() * 0.12;
        temperature = 76 + Math.random() * 6;
        power = 410 + Math.random() * 35;
      } else if (timeMs >= 320 && timeMs <= 476) { // Naive kernel
        utilization = 0.65 + Math.random() * 0.15; // Lower utilization
        memUtil = 0.45 + Math.random() * 0.15;
        temperature = 70 + Math.random() * 4;
        power = 280 + Math.random() * 40;
      } else if (timeMs >= 50 && timeMs <= 62) { // Transpose
        utilization = 0.78 + Math.random() * 0.10;
        memUtil = 0.65 + Math.random() * 0.15;
        temperature = 66 + Math.random() * 3;
        power = 250 + Math.random() * 20;
      }

      metrics.push({
        timestamp,
        utilization: Math.max(0, Math.min(1, utilization)),
        memoryUtilization: Math.max(0, Math.min(1, memUtil)),
        temperature,
        powerDraw: power
      });
    }

    const session: ProfilingSession = {
      id: sessionId,
      name: 'Matrix Multiplication Benchmark (4096x4096)',
      device: 'NVIDIA GeForce RTX 4090',
      driverVersion: '546.33',
      cudaVersion: '12.3',
      totalDuration: 550000000, // 550ms
      kernelLaunches: kernels,
      memoryTransfers,
      metrics,
      createdAt: new Date()
    };

    // Generate realistic bottleneck hints
    const hints: BottleneckHint[] = [
      {
        type: 'occupancy',
        severity: 'high',
        title: 'Low Occupancy in Custom Kernel',
        description: 'naive_matrix_multiply_kernel has 45% occupancy, significantly below cuBLAS performance',
        suggestion: 'Consider using cuBLAS for matrix operations or optimize block size and shared memory usage',
        kernelIds: ['demo_k2']
      },
      {
        type: 'compute', 
        severity: 'medium',
        title: 'Performance Gap vs Optimized Libraries',
        description: 'Custom implementation is 2.1x slower than cuBLAS equivalent operations',
        suggestion: 'Use highly optimized libraries (cuBLAS, cuDNN) for standard operations when possible',
        kernelIds: ['demo_k2']
      },
      {
        type: 'memory',
        severity: 'low',
        title: 'PCIe Bandwidth Underutilization', 
        description: 'Memory transfers at 7.3 GB/s - below theoretical PCIe 4.0 bandwidth',
        suggestion: 'Use pinned memory allocation and larger transfer sizes to improve bandwidth'
      }
    ];

    return { session, hints };
  }

  static generateDeepLearningProfile(): ParsedProfile {
    const sessionId = `demo_resnet_${Date.now()}`;
    
    const kernels: KernelLaunch[] = [
      // Convolution layers (cuDNN)
      {
        id: 'demo_dl_k0',
        name: 'volta_scudnn_128x128_relu_interior_nn',
        startTime: 100000000,
        duration: 145000000, // Conv2d is expensive
        streamId: 0,
        blockDim: { x: 128, y: 128, z: 1 },
        gridDim: { x: 32, y: 32, z: 1 },
        sharedMemory: 49152,
        registers: 84,
        occupancy: 0.87,
        memoryThroughput: 756.4,
        computeThroughput: 18.5e12
      },
      // Batch normalization
      {
        id: 'demo_dl_k1',
        name: 'batch_norm_forward_kernel',
        startTime: 250000000,
        duration: 8500000,
        streamId: 1,
        blockDim: { x: 512, y: 1, z: 1 },
        gridDim: { x: 2048, y: 1, z: 1 },
        sharedMemory: 16384,
        registers: 28,
        occupancy: 0.92,
        memoryThroughput: 445.8,
        computeThroughput: 2.1e12
      },
      // ReLU activation
      {
        id: 'demo_dl_k2',
        name: 'relu_activation_kernel',
        startTime: 260000000,
        duration: 1200000,
        streamId: 1,
        blockDim: { x: 256, y: 1, z: 1 },
        gridDim: { x: 8192, y: 1, z: 1 },
        sharedMemory: 0,
        registers: 12,
        occupancy: 0.95,
        memoryThroughput: 234.5,
        computeThroughput: 0.3e12
      },
      // Another conv layer with different pattern
      {
        id: 'demo_dl_k3',
        name: 'maxwell_scudnn_winograd_128x128_ldg1_ldg4',
        startTime: 270000000,
        duration: 189000000, // Winograd is complex
        streamId: 0,
        blockDim: { x: 128, y: 128, z: 1 },
        gridDim: { x: 16, y: 16, z: 1 },
        sharedMemory: 32768,
        registers: 96,
        occupancy: 0.76, // Lower due to high register usage
        memoryThroughput: 678.9,
        computeThroughput: 16.8e12
      },
      // Global average pooling
      {
        id: 'demo_dl_k4',
        name: 'adaptive_avg_pool2d_kernel',
        startTime: 465000000,
        duration: 3400000,
        streamId: 2,
        blockDim: { x: 128, y: 1, z: 1 },
        gridDim: { x: 512, y: 1, z: 1 },
        sharedMemory: 4096,
        registers: 24,
        occupancy: 0.84,
        memoryThroughput: 156.7,
        computeThroughput: 0.8e12
      },
      // Fully connected layer
      {
        id: 'demo_dl_k5',
        name: 'ampere_fp16_s884gemm_fp16_256x128_ldg8',
        startTime: 470000000,
        duration: 12000000,
        streamId: 0,
        blockDim: { x: 256, y: 128, z: 1 },
        gridDim: { x: 4, y: 4, z: 1 },
        sharedMemory: 65536,
        registers: 128,
        occupancy: 0.89,
        memoryThroughput: 567.8,
        computeThroughput: 24.6e12 // High FLOPS with FP16
      }
    ];

    const memoryTransfers: MemoryTransfer[] = [
      // Input batch upload
      {
        id: 'demo_dl_h2d_0',
        type: 'H2D',
        startTime: 50000000,
        duration: 45000000,
        size: 512 * 1024 * 1024, // 512MB batch
        bandwidth: 11.4
      },
      // Model weights (already loaded, just small transfers)
      {
        id: 'demo_dl_h2d_1',
        type: 'H2D', 
        startTime: 60000000,
        duration: 5000000,
        size: 64 * 1024 * 1024, // 64MB weights
        bandwidth: 12.8
      },
      // Output predictions download
      {
        id: 'demo_dl_d2h_0',
        type: 'D2H',
        startTime: 485000000,
        duration: 8000000,
        size: 32 * 1024 * 1024, // 32MB predictions
        bandwidth: 4.0
      }
    ];

    // Realistic metrics for deep learning workload
    const metrics: GPUMetrics[] = [];
    const totalDuration = 500; // 500ms
    for (let i = 0; i < totalDuration * 3; i++) { // 3 samples per ms
      const timeMs = i / 3;
      const timestamp = timeMs * 1000000;
      
      let utilization = 0.15;
      let memUtil = 0.75; // High memory usage is typical
      let temperature = 68;
      let power = 220;

      // High utilization during convolutions
      if ((timeMs >= 100 && timeMs <= 245) || (timeMs >= 270 && timeMs <= 459)) {
        utilization = 0.88 + Math.random() * 0.08;
        memUtil = 0.92 + Math.random() * 0.06;
        temperature = 82 + Math.random() * 6;
        power = 380 + Math.random() * 40;
      }

      metrics.push({
        timestamp,
        utilization: Math.max(0, Math.min(1, utilization)),
        memoryUtilization: Math.max(0, Math.min(1, memUtil)),
        temperature,
        powerDraw: power
      });
    }

    const session: ProfilingSession = {
      id: sessionId,
      name: 'ResNet-50 Forward Pass (Batch=32)',
      device: 'NVIDIA RTX A6000', 
      driverVersion: '546.33',
      cudaVersion: '12.3',
      totalDuration: 500000000,
      kernelLaunches: kernels,
      memoryTransfers,
      metrics,
      createdAt: new Date()
    };

    const hints: BottleneckHint[] = [
      {
        type: 'occupancy',
        severity: 'medium',
        title: 'Register Pressure in Winograd Convolution',
        description: 'Winograd kernel using 96 registers per thread, reducing occupancy to 76%',
        suggestion: 'Consider using different convolution algorithm or reduce register usage with compiler flags',
        kernelIds: ['demo_dl_k3']
      },
      {
        type: 'memory',
        severity: 'high',
        title: 'High Memory Utilization Throughout Session',
        description: 'Memory utilization consistently above 90% - risk of memory bandwidth bottlenecks',
        suggestion: 'Consider reducing batch size or using gradient checkpointing to reduce memory pressure'
      },
      {
        type: 'compute',
        severity: 'low',
        title: 'Mixed Precision Optimization Opportunity',
        description: 'Most kernels using FP32. FP16 operations show higher throughput',
        suggestion: 'Enable mixed precision training with automatic loss scaling for better performance'
      }
    ];

    return { session, hints };
  }

  static generateScienceComputingProfile(): ParsedProfile {
    const sessionId = `demo_nbody_${Date.now()}`;
    
    const kernels: KernelLaunch[] = [
      // N-body simulation kernel
      {
        id: 'demo_sci_k0',
        name: 'nbody_calculate_forces_kernel',
        startTime: 100000000,
        duration: 234000000, // Long-running simulation
        streamId: 0,
        blockDim: { x: 256, y: 1, z: 1 },
        gridDim: { x: 4096, y: 1, z: 1 },
        sharedMemory: 12288,
        registers: 48,
        occupancy: 0.67, // Moderate occupancy due to algorithm complexity
        memoryThroughput: 234.5,
        computeThroughput: 8.9e12
      },
      // Position update kernel
      {
        id: 'demo_sci_k1',
        name: 'update_positions_kernel', 
        startTime: 340000000,
        duration: 15000000,
        streamId: 1,
        blockDim: { x: 512, y: 1, z: 1 },
        gridDim: { x: 2048, y: 1, z: 1 },
        sharedMemory: 0,
        registers: 16,
        occupancy: 0.94,
        memoryThroughput: 345.6,
        computeThroughput: 1.2e12
      },
      // Velocity update kernel
      {
        id: 'demo_sci_k2',
        name: 'update_velocities_kernel',
        startTime: 356000000,
        duration: 18000000,
        streamId: 1, 
        blockDim: { x: 512, y: 1, z: 1 },
        gridDim: { x: 2048, y: 1, z: 1 },
        sharedMemory: 0,
        registers: 20,
        occupancy: 0.91,
        memoryThroughput: 298.4,
        computeThroughput: 1.4e12
      },
      // Reduction kernel for total energy
      {
        id: 'demo_sci_k3',
        name: 'reduce_total_energy_kernel',
        startTime: 375000000,
        duration: 8500000,
        streamId: 2,
        blockDim: { x: 256, y: 1, z: 1 },
        gridDim: { x: 256, y: 1, z: 1 },
        sharedMemory: 8192,
        registers: 32,
        occupancy: 0.78,
        memoryThroughput: 156.7,
        computeThroughput: 2.1e12
      },
      // Data visualization kernel
      {
        id: 'demo_sci_k4',
        name: 'render_particles_kernel',
        startTime: 390000000,
        duration: 28000000,
        streamId: 0,
        blockDim: { x: 64, y: 64, z: 1 },
        gridDim: { x: 64, y: 64, z: 1 },
        sharedMemory: 4096,
        registers: 24,
        occupancy: 0.85,
        memoryThroughput: 445.8,
        computeThroughput: 3.2e12
      }
    ];

    const memoryTransfers: MemoryTransfer[] = [
      // Initial data upload
      {
        id: 'demo_sci_h2d_0',
        type: 'H2D',
        startTime: 50000000,
        duration: 40000000,
        size: 128 * 1024 * 1024, // 128MB particle data
        bandwidth: 3.2 // Slower due to small random access patterns
      },
      // Results download
      {
        id: 'demo_sci_d2h_0',
        type: 'D2H',
        startTime: 420000000,
        duration: 32000000,
        size: 96 * 1024 * 1024, // 96MB updated positions
        bandwidth: 3.0
      },
      // Energy value download (small)
      {
        id: 'demo_sci_d2h_1',
        type: 'D2H',
        startTime: 384000000,
        duration: 100000, // Very small transfer
        size: 1024, // 1KB energy values
        bandwidth: 10.2
      }
    ];

    const metrics: GPUMetrics[] = [];
    const totalDuration = 460; // 460ms
    for (let i = 0; i < totalDuration * 2; i++) {
      const timeMs = i / 2;
      const timestamp = timeMs * 1000000;
      
      let utilization = 0.08;
      let memUtil = 0.35;
      let temperature = 62;
      let power = 180;

      // High utilization during force calculation
      if (timeMs >= 100 && timeMs <= 334) {
        utilization = 0.72 + Math.random() * 0.15; // Variable load
        memUtil = 0.45 + Math.random() * 0.20;
        temperature = 75 + Math.random() * 8;
        power = 320 + Math.random() * 50;
      }
      // Moderate utilization during updates and rendering
      else if (timeMs >= 340 && timeMs <= 418) {
        utilization = 0.55 + Math.random() * 0.25;
        memUtil = 0.60 + Math.random() * 0.15;
        temperature = 70 + Math.random() * 5;
        power = 280 + Math.random() * 30;
      }

      metrics.push({
        timestamp,
        utilization: Math.max(0, Math.min(1, utilization)),
        memoryUtilization: Math.max(0, Math.min(1, memUtil)),
        temperature,
        powerDraw: power
      });
    }

    const session: ProfilingSession = {
      id: sessionId,
      name: 'N-Body Simulation (1M particles)',
      device: 'NVIDIA Tesla V100',
      driverVersion: '535.86',
      cudaVersion: '11.8',
      totalDuration: 460000000,
      kernelLaunches: kernels,
      memoryTransfers,
      metrics,
      createdAt: new Date()
    };

    const hints: BottleneckHint[] = [
      {
        type: 'divergence',
        severity: 'medium',
        title: 'Potential Warp Divergence in N-Body Kernel',
        description: 'Force calculation kernel shows variable execution times suggesting branch divergence',
        suggestion: 'Consider restructuring conditionals or using warp-level primitives to reduce divergence',
        kernelIds: ['demo_sci_k0']
      },
      {
        type: 'memory',
        severity: 'high',
        title: 'Low Memory Transfer Bandwidth',
        description: 'Memory transfers achieving only 3.2 GB/s - well below PCIe capabilities',
        suggestion: 'Use coalesced memory access patterns and consider data structure reorganization',
      },
      {
        type: 'occupancy',
        severity: 'low',
        title: 'Moderate Occupancy in Main Compute Kernel',
        description: 'N-Body kernel running at 67% occupancy - room for improvement',
        suggestion: 'Experiment with different block sizes and shared memory usage patterns',
        kernelIds: ['demo_sci_k0']
      }
    ];

    return { session, hints };
  }

  static getAllDemoProfiles(): Array<{ name: string; description: string; generator: () => ParsedProfile }> {
    return [
      {
        name: 'Matrix Multiplication',
        description: 'cuBLAS vs naive implementation comparison',
        generator: this.generateMatrixMultiplicationProfile
      },
      {
        name: 'Deep Learning (ResNet-50)',
        description: 'CNN forward pass with cuDNN kernels',
        generator: this.generateDeepLearningProfile
      },
      {
        name: 'Scientific Computing',
        description: 'N-Body simulation with visualization',
        generator: this.generateScienceComputingProfile
      }
    ];
  }
}