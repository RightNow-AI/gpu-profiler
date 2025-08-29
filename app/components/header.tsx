"use client";

import { Github, ExternalLink } from "lucide-react";

export function Header() {
  return (
    <header className="border-b border-neutral-200 dark:border-neutral-700 bg-background">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex h-14 items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <a 
                href="/"
                className="font-jetbrains text-base font-bold text-neutral-800 dark:text-white hover:text-neutral-600 dark:hover:text-neutral-300 transition-colors cursor-pointer"
              >
                GPU_PROFILER
              </a>
            </div>
            <div className="hidden md:flex items-center space-x-1 text-neutral-500 dark:text-neutral-500">
              <span className="font-jetbrains text-xs text-neutral-800 dark:text-neutral-300">━</span>
              <span className="font-jetbrains text-xs">made by RightNow AI</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-1">
            <a 
              href="https://github.com/rightnow-ai/gpu-profiler" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center space-x-1.5 px-3 py-1.5 text-neutral-600 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-white hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors duration-100"
              style={{borderRadius: '2px'}}
            >
              <Github className="h-3.5 w-3.5" />
              <span className="font-jetbrains text-xs font-medium">GITHUB</span>
            </a>
            <a 
              href="https://www.rightnowai.co/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center space-x-1.5 px-3 py-1.5 font-jetbrains text-xs font-medium bg-neutral-800 dark:bg-white text-white dark:text-neutral-800 hover:bg-neutral-700 dark:hover:bg-neutral-100 transition-colors duration-100"
              style={{borderRadius: '2px'}}
            >
              <ExternalLink className="h-3.5 w-3.5" />
              <span>RIGHTNOW_AI</span>
            </a>
          </div>
        </div>
      </div>
    </header>
  );
}