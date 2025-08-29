"use client";

import { useState } from "react";
import { BottleneckHint, ProfilingSession } from "../types/profiling";
import { AlertTriangle, Info, AlertCircle, CheckCircle, ChevronDown, ChevronRight, Zap, Activity, Cpu, MemoryStick, GitBranch } from "lucide-react";
import { formatDuration, formatPercentage } from "../lib/utils";

interface BottleneckAnalysisProps {
  hints: BottleneckHint[];
  session: ProfilingSession;
}

export function BottleneckAnalysis({ hints, session }: BottleneckAnalysisProps) {
  const [expandedHints, setExpandedHints] = useState<Set<string>>(new Set());
  const [selectedSeverity, setSelectedSeverity] = useState<'all' | 'high' | 'medium' | 'low'>('all');

  const toggleHint = (index: number) => {
    const key = `hint-${index}`;
    setExpandedHints(prev => {
      const newSet = new Set(prev);
      if (newSet.has(key)) {
        newSet.delete(key);
      } else {
        newSet.add(key);
      }
      return newSet;
    });
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'high': return AlertTriangle;
      case 'medium': return AlertCircle;
      case 'low': return Info;
      default: return Info;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'text-red-600 dark:text-red-400';
      case 'medium': return 'text-yellow-600 dark:text-yellow-400';
      case 'low': return 'text-blue-600 dark:text-blue-400';
      default: return 'text-neutral-600 dark:text-neutral-400';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'occupancy': return Activity;
      case 'memory': return MemoryStick;
      case 'compute': return Cpu;
      case 'divergence': return GitBranch;
      case 'sync': return Zap;
      default: return AlertCircle;
    }
  };

  const filteredHints = selectedSeverity === 'all' 
    ? hints 
    : hints.filter(hint => hint.severity === selectedSeverity);

  // Calculate session statistics
  const totalKernelTime = session.kernelLaunches.reduce((sum, k) => sum + k.duration, 0);
  const avgOccupancy = session.kernelLaunches.reduce((sum, k) => sum + (k.occupancy || 0), 0) / session.kernelLaunches.length;
  const gpuUtilization = totalKernelTime / session.totalDuration;

  const highSeverityCount = hints.filter(h => h.severity === 'high').length;
  const mediumSeverityCount = hints.filter(h => h.severity === 'medium').length;
  const lowSeverityCount = hints.filter(h => h.severity === 'low').length;

  return (
    <div className="border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
      <div className="flex items-center justify-between p-3 border-b border-neutral-200 dark:border-neutral-700">
        <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
          BOTTLENECK_ANALYSIS
        </div>
        <div className="flex items-center space-x-1">
          {(['all', 'high', 'medium', 'low'] as const).map(severity => (
            <button
              key={severity}
              onClick={() => setSelectedSeverity(severity)}
              className={`px-2 py-1 text-xs font-jetbrains transition-colors ${
                selectedSeverity === severity
                  ? 'bg-neutral-800 dark:bg-white text-white dark:text-neutral-800'
                  : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-800'
              }`}
              style={{borderRadius: '2px'}}
            >
              {severity.toUpperCase()}
              {severity !== 'all' && (
                <span className={`ml-1 ${getSeverityColor(severity)}`}>
                  ({severity === 'high' ? highSeverityCount : 
                    severity === 'medium' ? mediumSeverityCount : lowSeverityCount})
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      <div className="p-4">
        {/* Session Overview */}
        <div className="mb-6 p-3 bg-neutral-50 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
          <div className="font-jetbrains text-xs font-medium text-neutral-800 dark:text-white mb-3">
            SESSION_OVERVIEW
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-1">
              <div className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">
                GPU_UTILIZATION
              </div>
              <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
                {formatPercentage(gpuUtilization)}
              </div>
            </div>
            <div className="space-y-1">
              <div className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">
                AVG_OCCUPANCY
              </div>
              <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
                {formatPercentage(avgOccupancy)}
              </div>
            </div>
            <div className="space-y-1">
              <div className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">
                TOTAL_KERNELS
              </div>
              <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
                {session.kernelLaunches.length}
              </div>
            </div>
            <div className="space-y-1">
              <div className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">
                SESSION_TIME
              </div>
              <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
                {formatDuration(session.totalDuration)}
              </div>
            </div>
          </div>
        </div>

        {/* Issues Summary */}
        <div className="mb-6">
          <div className="font-jetbrains text-xs font-medium text-neutral-800 dark:text-white mb-3">
            ISSUES_SUMMARY
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-red-600 dark:text-red-400" />
              <span className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                {highSeverityCount} HIGH
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
              <span className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                {mediumSeverityCount} MEDIUM
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <Info className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              <span className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                {lowSeverityCount} LOW
              </span>
            </div>
            {hints.length === 0 && (
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
                <span className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                  NO_ISSUES_DETECTED
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Hints List */}
        {filteredHints.length > 0 ? (
          <div className="space-y-3">
            {filteredHints.map((hint, index) => {
              const isExpanded = expandedHints.has(`hint-${index}`);
              const SeverityIcon = getSeverityIcon(hint.severity);
              const TypeIcon = getTypeIcon(hint.type);

              return (
                <div 
                  key={index}
                  className="border border-neutral-200 dark:border-neutral-700"
                  style={{borderRadius: '2px'}}
                >
                  <button
                    onClick={() => toggleHint(index)}
                    className="w-full flex items-center justify-between p-3 hover:bg-neutral-50 dark:hover:bg-neutral-900 transition-colors"
                  >
                    <div className="flex items-center space-x-3">
                      <SeverityIcon className={`h-4 w-4 ${getSeverityColor(hint.severity)}`} />
                      <TypeIcon className="h-4 w-4 text-neutral-500 dark:text-neutral-400" />
                      <div className="text-left">
                        <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
                          {hint.title}
                        </div>
                        <div className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                          {hint.type.toUpperCase()} • {hint.severity.toUpperCase()}
                        </div>
                      </div>
                    </div>
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4 text-neutral-500 dark:text-neutral-400" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-neutral-500 dark:text-neutral-400" />
                    )}
                  </button>
                  
                  {isExpanded && (
                    <div className="px-3 pb-3 border-t border-neutral-200 dark:border-neutral-700">
                      <div className="pt-3 space-y-3">
                        <div>
                          <div className="font-jetbrains text-xs font-medium text-neutral-600 dark:text-neutral-400 mb-1">
                            DESCRIPTION
                          </div>
                          <div className="font-jetbrains text-xs text-neutral-800 dark:text-white">
                            {hint.description}
                          </div>
                        </div>
                        
                        <div>
                          <div className="font-jetbrains text-xs font-medium text-neutral-600 dark:text-neutral-400 mb-1">
                            RECOMMENDATION
                          </div>
                          <div className="font-jetbrains text-xs text-neutral-800 dark:text-white">
                            {hint.suggestion}
                          </div>
                        </div>

                        {hint.kernelIds && hint.kernelIds.length > 0 && (
                          <div>
                            <div className="font-jetbrains text-xs font-medium text-neutral-600 dark:text-neutral-400 mb-1">
                              AFFECTED_KERNELS ({hint.kernelIds.length})
                            </div>
                            <div className="flex flex-wrap gap-1">
                              {hint.kernelIds.slice(0, 5).map((kernelId, kidx) => {
                                const kernel = session.kernelLaunches.find(k => k.id === kernelId);
                                return (
                                  <span
                                    key={kidx}
                                    className="inline-block px-2 py-1 bg-neutral-100 dark:bg-neutral-800 font-jetbrains text-xs text-neutral-600 dark:text-neutral-400"
                                    style={{borderRadius: '2px'}}
                                  >
                                    {kernel?.name || kernelId}
                                  </span>
                                );
                              })}
                              {hint.kernelIds.length > 5 && (
                                <span className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">
                                  +{hint.kernelIds.length - 5} more
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-8">
            {selectedSeverity === 'all' ? (
              <div className="flex flex-col items-center space-y-2">
                <CheckCircle className="h-8 w-8 text-green-500" />
                <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
                  NO_BOTTLENECKS_DETECTED
                </div>
                <div className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                  Your GPU profile looks optimized!
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center space-y-2">
                <Info className="h-8 w-8 text-neutral-400" />
                <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
                  NO_{selectedSeverity.toUpperCase()}_ISSUES
                </div>
                <div className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                  Try selecting &quot;ALL&quot; to see all issues
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}