"use client";

import { useState } from "react";
import { X, BarChart3, Flame, Activity, AlertTriangle } from "lucide-react";
import { DemoDataGenerator } from "../lib/demo-data";
import { ParsedProfile } from "../types/profiling";

interface DemoSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onDemoSelected: (profile: ParsedProfile) => void;
}

export function DemoSelectionModal({ isOpen, onClose, onDemoSelected }: DemoSelectionModalProps) {
  const [selectedDemo, setSelectedDemo] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const demos = DemoDataGenerator.getAllDemoProfiles();

  const handleSelectDemo = async (demoName: string) => {
    setIsLoading(true);
    try {
      const demo = demos.find(d => d.name === demoName);
      if (demo) {
        const profile = demo.generator();
        onDemoSelected(profile);
        onClose();
      }
    } catch (error) {
      console.error('Failed to load demo:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700 max-w-2xl w-full mx-4" style={{borderRadius: '2px'}}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-neutral-200 dark:border-neutral-700">
          <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
            SELECT_DEMO_PROFILE
          </div>
          <button
            onClick={onClose}
            className="text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          <div className="mb-6">
            <div className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400 mb-4">
              Choose from realistic GPU profiling examples based on real CUDA workloads:
            </div>
          </div>

          <div className="space-y-3">
            {demos.map((demo, index) => {
              const isSelected = selectedDemo === demo.name;
              const getIcon = () => {
                if (demo.name.includes('Matrix')) return BarChart3;
                if (demo.name.includes('Deep Learning')) return Flame;
                if (demo.name.includes('Scientific')) return Activity;
                return AlertTriangle;
              };
              const Icon = getIcon();

              return (
                <button
                  key={index}
                  onClick={() => setSelectedDemo(demo.name)}
                  onDoubleClick={() => handleSelectDemo(demo.name)}
                  disabled={isLoading}
                  className={`w-full p-4 border text-left transition-all ${
                    isSelected
                      ? 'border-neutral-800 dark:border-white bg-neutral-50 dark:bg-neutral-800'
                      : 'border-neutral-200 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-800'
                  } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                  style={{borderRadius: '2px'}}
                >
                  <div className="flex items-start space-x-3">
                    <Icon className="h-5 w-5 text-neutral-500 dark:text-neutral-400 mt-0.5" />
                    <div className="flex-1">
                      <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white mb-1">
                        {demo.name.toUpperCase().replace(/ /g, '_')}
                      </div>
                      <div className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                        {demo.description}
                      </div>
                      
                      {/* Preview stats */}
                      <div className="mt-3 flex items-center space-x-4 font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">
                        {demo.name.includes('Matrix') && (
                          <>
                            <span>5 KERNELS</span>
                            <span>•</span>
                            <span>550MS</span>
                            <span>•</span>
                            <span>cuBLAS + NAIVE</span>
                          </>
                        )}
                        {demo.name.includes('Deep Learning') && (
                          <>
                            <span>6 KERNELS</span>
                            <span>•</span>
                            <span>500MS</span>
                            <span>•</span>
                            <span>RESNET-50</span>
                          </>
                        )}
                        {demo.name.includes('Scientific') && (
                          <>
                            <span>5 KERNELS</span>
                            <span>•</span>
                            <span>460MS</span>
                            <span>•</span>
                            <span>N-BODY_1M</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Actions */}
          <div className="mt-6 flex items-center justify-between">
            <div className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">
              {selectedDemo ? 'Double-click or use LOAD button' : 'Select a demo profile'}
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={onClose}
                className="px-4 py-2 font-jetbrains text-xs text-neutral-600 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-white"
                style={{borderRadius: '2px'}}
              >
                CANCEL
              </button>
              <button
                onClick={() => selectedDemo && handleSelectDemo(selectedDemo)}
                disabled={!selectedDemo || isLoading}
                className={`rn-button ${!selectedDemo || isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {isLoading ? 'LOADING...' : 'LOAD_DEMO'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}