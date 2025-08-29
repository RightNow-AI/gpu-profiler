"use client";

import { ArrowRight, Sparkles } from "lucide-react";

export function NotificationBanner() {
  return (
    <div className="relative z-50 w-full bg-gradient-to-r from-neutral-800 to-neutral-700 dark:from-neutral-900 dark:to-neutral-800 py-2 text-xs md:text-sm">
      <div className="max-w-6xl mx-auto px-4">
        <a 
          href="https://www.rightnowai.co/" 
          target="_blank"
          rel="noopener noreferrer"
          className="group flex w-fit items-center gap-2 text-white hover:text-neutral-200 transition-colors"
        >
          <div className="flex items-center">
            <Sparkles className="h-4 w-4 shrink-0 animate-pulse text-yellow-400" />
          </div>
          <span className="font-jetbrains text-pretty md:hidden">
            Join GPU AI Editor Waitlist
          </span>
          <span className="font-jetbrains hidden text-pretty md:block">
            NEW: Join the waitlist for our GPU AI Code Editor
          </span>
          <ArrowRight className="h-4 w-4 shrink-0 group-hover:translate-x-1 transition-transform" />
        </a>
      </div>
    </div>
  );
}