// Core profiling data structures
export interface KernelLaunch {
  id: string;
  name: string;
  startTime: number; // nanoseconds
  duration: number; // nanoseconds  
  streamId: number;
  blockDim: {x: number, y: number, z: number};
  gridDim: {x: number, y: number, z: number};
  sharedMemory: number; // bytes
  registers: number;
  occupancy?: number; // 0-1
  memoryThroughput?: number; // GB/s
  computeThroughput?: number; // FLOPS/s
}

export interface MemoryTransfer {
  id: string;
  type: 'H2D' | 'D2H' | 'D2D'; // Host to Device, Device to Host, Device to Device
  startTime: number;
  duration: number;
  size: number; // bytes
  bandwidth: number; // GB/s
}

export interface GPUMetrics {
  timestamp: number;
  utilization: number; // 0-1
  memoryUtilization: number; // 0-1
  temperature?: number; // Celsius
  powerDraw?: number; // Watts
}

export interface ProfilingSession {
  id: string;
  name: string;
  device: string;
  driverVersion?: string;
  cudaVersion?: string;
  totalDuration: number;
  kernelLaunches: KernelLaunch[];
  memoryTransfers: MemoryTransfer[];
  metrics: GPUMetrics[];
  createdAt: Date;
}

export interface BottleneckHint {
  type: 'occupancy' | 'memory' | 'compute' | 'divergence' | 'sync';
  severity: 'low' | 'medium' | 'high';
  title: string;
  description: string;
  suggestion: string;
  kernelIds?: string[];
}

export interface ParsedProfile {
  session: ProfilingSession;
  hints: BottleneckHint[];
}

// File format types
export type SupportedFormat = 'nvprof' | 'nsys' | 'json';

export interface UploadedFile {
  file: File;
  format: SupportedFormat;
  content?: ArrayBuffer | string;
}