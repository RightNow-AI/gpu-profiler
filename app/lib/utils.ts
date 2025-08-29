import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(nanoseconds: number): string {
  if (nanoseconds < 1000) {
    return `${nanoseconds}ns`;
  } else if (nanoseconds < 1000000) {
    return `${(nanoseconds / 1000).toFixed(2)}μs`;
  } else if (nanoseconds < 1000000000) {
    return `${(nanoseconds / 1000000).toFixed(2)}ms`;
  } else {
    return `${(nanoseconds / 1000000000).toFixed(2)}s`;
  }
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)}GB`;
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function generateShareableId(): string {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}