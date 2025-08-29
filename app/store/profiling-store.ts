import { create } from 'zustand';
import { ParsedProfile, ProfilingSession, KernelLaunch, BottleneckHint } from '../types/profiling';

interface ProfilingState {
  // Current session data
  currentProfile: ParsedProfile | null;
  selectedKernel: KernelLaunch | null;
  
  // UI state
  isUploading: boolean;
  uploadProgress: number;
  error: string | null;
  
  // View settings
  timelineZoom: { start: number; end: number } | null;
  showOnlyBottlenecks: boolean;
  sortBy: 'duration' | 'startTime' | 'name';
  
  // Actions
  setCurrentProfile: (profile: ParsedProfile | null) => void;
  setSelectedKernel: (kernel: KernelLaunch | null) => void;
  setUploadState: (uploading: boolean, progress?: number) => void;
  setError: (error: string | null) => void;
  setTimelineZoom: (zoom: { start: number; end: number } | null) => void;
  toggleBottlenecksFilter: () => void;
  setSortBy: (sortBy: 'duration' | 'startTime' | 'name') => void;
  reset: () => void;
}

export const useProfilingStore = create<ProfilingState>((set, get) => ({
  // Initial state
  currentProfile: null,
  selectedKernel: null,
  isUploading: false,
  uploadProgress: 0,
  error: null,
  timelineZoom: null,
  showOnlyBottlenecks: false,
  sortBy: 'duration',
  
  // Actions
  setCurrentProfile: (profile) => set({ currentProfile: profile, error: null }),
  
  setSelectedKernel: (kernel) => set({ selectedKernel: kernel }),
  
  setUploadState: (uploading, progress = 0) => 
    set({ isUploading: uploading, uploadProgress: progress }),
  
  setError: (error) => set({ error }),
  
  setTimelineZoom: (zoom) => set({ timelineZoom: zoom }),
  
  toggleBottlenecksFilter: () => 
    set((state) => ({ showOnlyBottlenecks: !state.showOnlyBottlenecks })),
  
  setSortBy: (sortBy) => set({ sortBy }),
  
  reset: () => set({
    currentProfile: null,
    selectedKernel: null,
    isUploading: false,
    uploadProgress: 0,
    error: null,
    timelineZoom: null,
    showOnlyBottlenecks: false,
    sortBy: 'duration'
  })
}));