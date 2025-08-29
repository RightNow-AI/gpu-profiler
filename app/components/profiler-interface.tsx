"use client";

import { useState } from "react";
import { FileUpload } from "./file-upload";
import { Timeline } from "./visualizations/timeline";
import { FlameGraph } from "./visualizations/flame-graph";
import { Heatmap } from "./visualizations/heatmap";
import { BottleneckAnalysis } from "./bottleneck-analysis";
import { DemoSelectionModal } from "./demo-selection-modal";
import { TimelinePreview } from "./previews/timeline-preview";
import { FlamePreview } from "./previews/flame-preview";
import { HeatmapPreview } from "./previews/heatmap-preview";
import { useProfilingStore } from "../store/profiling-store";
import { ProfileParser } from "../lib/parsers";
import { DemoDataGenerator } from "../lib/demo-data";
import { UploadedFile, KernelLaunch, ParsedProfile } from "../types/profiling";
import { BarChart3, Flame, Activity, AlertTriangle, Share2, Download, FileText } from "lucide-react";

export function ProfilerInterface() {
  const [activeTab, setActiveTab] = useState<'timeline' | 'flame' | 'heatmap' | 'analysis'>('timeline');
  const [showDemoModal, setShowDemoModal] = useState(false);
  const { currentProfile, selectedKernel, setCurrentProfile, setSelectedKernel, setError } = useProfilingStore();

  const handleFileUploaded = async (uploadedFile: UploadedFile) => {
    try {
      setError(null);
      const parsedProfile = await ProfileParser.parseFile(uploadedFile);
      setCurrentProfile(parsedProfile);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to parse file';
      setError(errorMessage);
    }
  };

  const handleKernelSelect = (kernel: KernelLaunch) => {
    setSelectedKernel(kernel);
  };

  const handleDemoSelected = (profile: ParsedProfile) => {
    setCurrentProfile(profile);
    setError(null);
  };

  const handleQuickDemo = () => {
    // Load the most popular demo (Matrix Multiplication) directly
    const profile = DemoDataGenerator.generateMatrixMultiplicationProfile();
    handleDemoSelected(profile);
  };

  const handleShareResults = async () => {
    if (!currentProfile) return;
    
    try {
      const shareData = {
        sessionId: currentProfile.session.id,
        timestamp: Date.now(),
        // In a real app, this would upload to a backend
        profile: currentProfile
      };
      
      const shareUrl = `${window.location.origin}${window.location.pathname}?share=${btoa(JSON.stringify(shareData))}`;
      await navigator.clipboard.writeText(shareUrl);
      
      // Show success feedback (you could add a toast notification here)
      console.log('Share URL copied to clipboard');
    } catch (error) {
      console.error('Failed to share results:', error);
    }
  };

  const handleExportData = () => {
    if (!currentProfile) return;
    
    const exportData = {
      ...currentProfile,
      exportedAt: new Date().toISOString(),
      exportedBy: 'GPU_PROFILER'
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `gpu_profile_${currentProfile.session.id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const tabs = [
    { id: 'timeline' as const, label: 'TIMELINE', icon: BarChart3 },
    { id: 'flame' as const, label: 'FLAME_GRAPH', icon: Flame },
    { id: 'heatmap' as const, label: 'HEATMAP', icon: Activity },
    { id: 'analysis' as const, label: 'ANALYSIS', icon: AlertTriangle }
  ];

  const getVisualizationDimensions = () => {
    // Responsive dimensions based on screen size
    const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;
    return {
      width: isMobile ? 350 : 800,
      height: isMobile ? 250 : 400
    };
  };

  if (!currentProfile) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="font-jetbrains text-3xl font-bold text-neutral-800 dark:text-white mb-4">
            GPU_PROFILER
          </h1>
          <p className="font-jetbrains text-sm text-neutral-600 dark:text-neutral-400 mb-8 flex items-center justify-center space-x-2">
            <FileText className="h-3 w-3" />
            <span>Drop .nvprof/.nsys files</span>
            <span>→</span>
            <BarChart3 className="h-3 w-3" />
            <span>Get insights</span>
          </p>
          <FileUpload onFileUploaded={handleFileUploaded} />
        </div>

        {/* Interactive Demo Section */}
        <div className="mb-16">
          <div className="text-center mb-12">
            <h2 className="font-jetbrains text-xl font-bold text-neutral-800 dark:text-white mb-3">
              LIVE_GPU_PROFILING_EXAMPLES
            </h2>
            <p className="font-jetbrains text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl mx-auto leading-relaxed">
              Experience real GPU profiling data from production CUDA workloads. These interactive visualizations show 
              actual kernel executions, performance bottlenecks, and optimization opportunities.
            </p>
          </div>

          {/* Matrix Multiplication Demo */}
          <div className="mb-12 p-6 border border-neutral-200 dark:border-neutral-700 bg-gradient-to-br from-blue-500/5 to-cyan-500/5" style={{borderRadius: '2px'}}>
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h3 className="font-jetbrains text-sm font-bold text-neutral-800 dark:text-white mb-1">
                    MATRIX_MULTIPLICATION: cuBLAS vs Naive Implementation
                  </h3>
                  <p className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                    Comparing optimized cuBLAS library against naive CUDA kernel • 4096×4096 matrices • 5 kernels • 550ms total
                  </p>
                </div>
                <button 
                  onClick={() => handleDemoSelected(DemoDataGenerator.generateMatrixMultiplicationProfile())}
                  className="rn-button px-4 py-2"
                >
                  EXPLORE_FULL_ANALYSIS
                </button>
              </div>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <TimelinePreview 
                kernels={DemoDataGenerator.generateMatrixMultiplicationProfile().session.kernelLaunches} 
                memoryTransfers={DemoDataGenerator.generateMatrixMultiplicationProfile().session.memoryTransfers}
              />
              <FlamePreview 
                kernels={DemoDataGenerator.generateMatrixMultiplicationProfile().session.kernelLaunches}
              />
              <HeatmapPreview 
                metrics={DemoDataGenerator.generateMatrixMultiplicationProfile().session.metrics}
              />
            </div>
            
            <div className="mt-4 grid grid-cols-3 gap-4 font-jetbrains text-xs">
              <div className="text-center p-3 bg-white/50 dark:bg-neutral-800/50" style={{borderRadius: '2px'}}>
                <div className="text-neutral-500 dark:text-neutral-400">BOTTLENECK</div>
                <div className="text-neutral-800 dark:text-white font-medium">Memory Bandwidth</div>
              </div>
              <div className="text-center p-3 bg-white/50 dark:bg-neutral-800/50" style={{borderRadius: '2px'}}>
                <div className="text-neutral-500 dark:text-neutral-400">SPEEDUP</div>
                <div className="text-neutral-800 dark:text-white font-medium">2.1x with cuBLAS</div>
              </div>
              <div className="text-center p-3 bg-white/50 dark:bg-neutral-800/50" style={{borderRadius: '2px'}}>
                <div className="text-neutral-500 dark:text-neutral-400">OCCUPANCY</div>
                <div className="text-neutral-800 dark:text-white font-medium">94% vs 45%</div>
              </div>
            </div>
          </div>

          {/* Deep Learning Demo */}
          <div className="mb-12 p-6 border border-neutral-200 dark:border-neutral-700 bg-gradient-to-br from-green-500/5 to-emerald-500/5" style={{borderRadius: '2px'}}>
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h3 className="font-jetbrains text-sm font-bold text-neutral-800 dark:text-white mb-1">
                    DEEP_LEARNING: ResNet-50 Forward Pass
                  </h3>
                  <p className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                    Neural network inference with cuDNN optimized convolutions • Mixed precision FP16/FP32 • 6 kernels • 500ms total
                  </p>
                </div>
                <button 
                  onClick={() => handleDemoSelected(DemoDataGenerator.generateDeepLearningProfile())}
                  className="rn-button px-4 py-2"
                >
                  EXPLORE_FULL_ANALYSIS
                </button>
              </div>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <TimelinePreview 
                kernels={DemoDataGenerator.generateDeepLearningProfile().session.kernelLaunches} 
                memoryTransfers={DemoDataGenerator.generateDeepLearningProfile().session.memoryTransfers}
              />
              <FlamePreview 
                kernels={DemoDataGenerator.generateDeepLearningProfile().session.kernelLaunches}
              />
              <HeatmapPreview 
                metrics={DemoDataGenerator.generateDeepLearningProfile().session.metrics}
              />
            </div>
            
            <div className="mt-4 grid grid-cols-3 gap-4 font-jetbrains text-xs">
              <div className="text-center p-3 bg-white/50 dark:bg-neutral-800/50" style={{borderRadius: '2px'}}>
                <div className="text-neutral-500 dark:text-neutral-400">BOTTLENECK</div>
                <div className="text-neutral-800 dark:text-white font-medium">Tensor Cores</div>
              </div>
              <div className="text-center p-3 bg-white/50 dark:bg-neutral-800/50" style={{borderRadius: '2px'}}>
                <div className="text-neutral-500 dark:text-neutral-400">PRECISION</div>
                <div className="text-neutral-800 dark:text-white font-medium">Mixed FP16/32</div>
              </div>
              <div className="text-center p-3 bg-white/50 dark:bg-neutral-800/50" style={{borderRadius: '2px'}}>
                <div className="text-neutral-500 dark:text-neutral-400">EFFICIENCY</div>
                <div className="text-neutral-800 dark:text-white font-medium">87% GPU Util</div>
              </div>
            </div>
          </div>

          {/* Scientific Computing Demo */}
          <div className="mb-8 p-6 border border-neutral-200 dark:border-neutral-700 bg-gradient-to-br from-purple-500/5 to-pink-500/5" style={{borderRadius: '2px'}}>
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h3 className="font-jetbrains text-sm font-bold text-neutral-800 dark:text-white mb-1">
                    SCIENTIFIC_COMPUTING: N-Body Particle Simulation
                  </h3>
                  <p className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                    High-performance computing workload with 1M particles • Memory-bound optimization • 5 kernels • 460ms total
                  </p>
                </div>
                <button 
                  onClick={() => handleDemoSelected(DemoDataGenerator.generateScienceComputingProfile())}
                  className="rn-button px-4 py-2"
                >
                  EXPLORE_FULL_ANALYSIS
                </button>
              </div>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <TimelinePreview 
                kernels={DemoDataGenerator.generateScienceComputingProfile().session.kernelLaunches} 
                memoryTransfers={DemoDataGenerator.generateScienceComputingProfile().session.memoryTransfers}
              />
              <FlamePreview 
                kernels={DemoDataGenerator.generateScienceComputingProfile().session.kernelLaunches}
              />
              <HeatmapPreview 
                metrics={DemoDataGenerator.generateScienceComputingProfile().session.metrics}
              />
            </div>
            
            <div className="mt-4 grid grid-cols-3 gap-4 font-jetbrains text-xs">
              <div className="text-center p-3 bg-white/50 dark:bg-neutral-800/50" style={{borderRadius: '2px'}}>
                <div className="text-neutral-500 dark:text-neutral-400">BOTTLENECK</div>
                <div className="text-neutral-800 dark:text-white font-medium">Memory Access</div>
              </div>
              <div className="text-center p-3 bg-white/50 dark:bg-neutral-800/50" style={{borderRadius: '2px'}}>
                <div className="text-neutral-500 dark:text-neutral-400">PARTICLES</div>
                <div className="text-neutral-800 dark:text-white font-medium">1M Simulated</div>
              </div>
              <div className="text-center p-3 bg-white/50 dark:bg-neutral-800/50" style={{borderRadius: '2px'}}>
                <div className="text-neutral-500 dark:text-neutral-400">THROUGHPUT</div>
                <div className="text-neutral-800 dark:text-white font-medium">2.2M ops/sec</div>
              </div>
            </div>
          </div>

          {/* Call to Action */}
          <div className="text-center">
            <div className="flex flex-wrap gap-3 justify-center">
              <button 
                onClick={() => setShowDemoModal(true)}
                className="rn-chip flex items-center space-x-2"
              >
                <FileText className="h-4 w-4" />
                <span>VIEW_MORE_SAMPLES</span>
              </button>
              <a href="https://github.com/rightnow-ai/gpu-profiler" className="rn-chip flex items-center space-x-2">
                <Share2 className="h-4 w-4" />
                <span>STAR_ON_GITHUB</span>
              </a>
              <a href="https://www.rightnowai.co/" className="rn-chip active flex items-center space-x-2">
                <Share2 className="h-4 w-4" />
                <span>RIGHTNOW_AI</span>
              </a>
            </div>
          </div>
        </div>


        {/* Demo Selection Modal */}
        <DemoSelectionModal 
          isOpen={showDemoModal}
          onClose={() => setShowDemoModal(false)}
          onDemoSelected={handleDemoSelected}
        />
      </div>
    );
  }

  const { session, hints } = currentProfile;
  const dimensions = getVisualizationDimensions();

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      {/* Session Header */}
      <div className="mb-6 p-4 border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="font-jetbrains text-lg font-bold text-neutral-800 dark:text-white mb-1">
              {session.name}
            </h1>
            <div className="flex items-center space-x-4 font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
              <span>{session.device}</span>
              <span>•</span>
              <span>{session.kernelLaunches.length} KERNELS</span>
              <span>•</span>
              <span>{hints.length} HINTS</span>
              <span>•</span>
              <span>{session.createdAt.toLocaleDateString()}</span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setCurrentProfile(null)}
              className="rn-chip flex items-center space-x-1"
            >
              <FileText className="h-3 w-3" />
              <span>NEW</span>
            </button>
            <button
              onClick={handleShareResults}
              className="rn-chip flex items-center space-x-1"
            >
              <Share2 className="h-3 w-3" />
              <span>SHARE</span>
            </button>
            <button
              onClick={handleExportData}
              className="rn-chip flex items-center space-x-1"
            >
              <Download className="h-3 w-3" />
              <span>EXPORT</span>
            </button>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="mb-6">
        <div className="flex items-center space-x-1 border-b border-neutral-200 dark:border-neutral-700">
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-4 py-3 font-jetbrains text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'text-neutral-800 dark:text-white border-b-2 border-neutral-800 dark:border-white'
                    : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-white'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Visualization Content */}
      <div className="space-y-6">
        {activeTab === 'timeline' && (
          <Timeline
            kernels={session.kernelLaunches}
            memoryTransfers={session.memoryTransfers}
            width={dimensions.width}
            height={dimensions.height}
            onKernelSelect={handleKernelSelect}
          />
        )}

        {activeTab === 'flame' && (
          <FlameGraph
            kernels={session.kernelLaunches}
            width={dimensions.width}
            height={dimensions.height}
            onKernelSelect={handleKernelSelect}
          />
        )}

        {activeTab === 'heatmap' && (
          <Heatmap
            metrics={session.metrics}
            width={dimensions.width}
            height={dimensions.height}
          />
        )}

        {activeTab === 'analysis' && (
          <BottleneckAnalysis
            hints={hints}
            session={session}
          />
        )}

        {/* Selected Kernel Details - Show on all tabs except analysis */}
        {selectedKernel && activeTab !== 'analysis' && (
          <div className="p-4 border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
            <div className="flex items-center justify-between mb-3">
              <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
                KERNEL_DETAILS
              </div>
              <button
                onClick={() => setSelectedKernel(null)}
                className="font-jetbrains text-xs text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
              >
                CLOSE
              </button>
            </div>
            <div className="space-y-3">
              <div>
                <div className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500 mb-1">
                  KERNEL_NAME
                </div>
                <div className="font-jetbrains text-sm text-neutral-800 dark:text-white font-medium">
                  {selectedKernel.name}
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">DURATION</div>
                  <div className="font-jetbrains text-sm text-neutral-800 dark:text-white">
                    {(selectedKernel.duration / 1000000).toFixed(2)}ms
                  </div>
                </div>
                <div>
                  <div className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">OCCUPANCY</div>
                  <div className="font-jetbrains text-sm text-neutral-800 dark:text-white">
                    {((selectedKernel.occupancy || 0) * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">STREAM</div>
                  <div className="font-jetbrains text-sm text-neutral-800 dark:text-white">
                    {selectedKernel.streamId}
                  </div>
                </div>
                <div>
                  <div className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">REGISTERS</div>
                  <div className="font-jetbrains text-sm text-neutral-800 dark:text-white">
                    {selectedKernel.registers}
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">BLOCK_DIM</div>
                  <div className="font-jetbrains text-sm text-neutral-800 dark:text-white">
                    {selectedKernel.blockDim.x}×{selectedKernel.blockDim.y}×{selectedKernel.blockDim.z}
                  </div>
                </div>
                <div>
                  <div className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">GRID_DIM</div>
                  <div className="font-jetbrains text-sm text-neutral-800 dark:text-white">
                    {selectedKernel.gridDim.x}×{selectedKernel.gridDim.y}×{selectedKernel.gridDim.z}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Quick Stats Footer */}
      <div className="mt-8 pt-6 border-t border-neutral-200 dark:border-neutral-700">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-jetbrains text-xs">
          <div className="text-center">
            <div className="text-neutral-500 dark:text-neutral-500">TOTAL_DURATION</div>
            <div className="text-neutral-800 dark:text-white font-medium">
              {(session.totalDuration / 1000000000).toFixed(2)}s
            </div>
          </div>
          <div className="text-center">
            <div className="text-neutral-500 dark:text-neutral-500">KERNEL_COUNT</div>
            <div className="text-neutral-800 dark:text-white font-medium">
              {session.kernelLaunches.length}
            </div>
          </div>
          <div className="text-center">
            <div className="text-neutral-500 dark:text-neutral-500">MEMORY_OPS</div>
            <div className="text-neutral-800 dark:text-white font-medium">
              {session.memoryTransfers.length}
            </div>
          </div>
          <div className="text-center">
            <div className="text-neutral-500 dark:text-neutral-500">ISSUE_COUNT</div>
            <div className="text-neutral-800 dark:text-white font-medium">
              {hints.length}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}