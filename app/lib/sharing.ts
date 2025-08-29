import { ParsedProfile } from '../types/profiling';

export interface ShareableData {
  sessionId: string;
  timestamp: number;
  profile: ParsedProfile;
  version: string;
}

export class SharingService {
  private static readonly VERSION = '1.0.0';
  private static readonly MAX_URL_LENGTH = 2000; // Browser URL length limit

  static async createShareableLink(profile: ParsedProfile): Promise<string> {
    try {
      const shareData: ShareableData = {
        sessionId: profile.session.id,
        timestamp: Date.now(),
        profile,
        version: this.VERSION
      };

      // Compress data for URL sharing
      const compressedData = this.compressProfileData(shareData);
      const encodedData = btoa(JSON.stringify(compressedData));
      
      // Check if URL would be too long
      const baseUrl = `${window.location.origin}${window.location.pathname}`;
      const shareUrl = `${baseUrl}?share=${encodedData}`;
      
      if (shareUrl.length > this.MAX_URL_LENGTH) {
        // For large profiles, we'd typically upload to a backend service
        // For now, we'll create a simplified version
        const simplifiedData = this.createSimplifiedProfile(profile);
        const simplifiedEncoded = btoa(JSON.stringify({
          sessionId: profile.session.id,
          timestamp: Date.now(),
          profile: simplifiedData,
          version: this.VERSION,
          simplified: true
        }));
        
        return `${baseUrl}?share=${simplifiedEncoded}`;
      }
      
      return shareUrl;
    } catch (error) {
      throw new Error(`Failed to create shareable link: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  static parseSharedData(shareParam: string): ShareableData | null {
    try {
      const decoded = atob(shareParam);
      const data = JSON.parse(decoded) as ShareableData;
      
      // Validate data structure
      if (!data.sessionId || !data.profile || !data.version) {
        throw new Error('Invalid share data format');
      }
      
      return data;
    } catch (error) {
      console.error('Failed to parse shared data:', error);
      return null;
    }
  }

  static async copyToClipboard(text: string): Promise<boolean> {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'absolute';
        textArea.style.left = '-999999px';
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        return true;
      }
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      return false;
    }
  }

  private static compressProfileData(data: ShareableData): Partial<ShareableData> {
    // Remove unnecessary data to reduce URL size
    const compressed = {
      sessionId: data.sessionId,
      timestamp: data.timestamp,
      version: data.version,
      profile: {
        session: {
          id: data.profile.session.id,
          name: data.profile.session.name,
          device: data.profile.session.device,
          totalDuration: data.profile.session.totalDuration,
          // Limit to top 50 kernels by duration
          kernelLaunches: data.profile.session.kernelLaunches
            .sort((a, b) => b.duration - a.duration)
            .slice(0, 50)
            .map(k => ({
              id: k.id,
              name: k.name.length > 30 ? k.name.substring(0, 30) + '...' : k.name,
              startTime: k.startTime,
              duration: k.duration,
              streamId: k.streamId,
              occupancy: k.occupancy ? Math.round(k.occupancy * 1000) / 1000 : undefined
            })),
          // Limit memory transfers
          memoryTransfers: data.profile.session.memoryTransfers.slice(0, 20),
          // Sample metrics (every 10th sample)
          metrics: data.profile.session.metrics.filter((_, i) => i % 10 === 0),
          createdAt: data.profile.session.createdAt
        },
        hints: data.profile.hints.slice(0, 10) // Limit hints
      }
    };

    return compressed;
  }

  private static createSimplifiedProfile(profile: ParsedProfile): Partial<ParsedProfile> {
    // Create a very simplified version for extremely large profiles
    const topKernels = profile.session.kernelLaunches
      .sort((a, b) => b.duration - a.duration)
      .slice(0, 20);

    return {
      session: {
        id: profile.session.id,
        name: profile.session.name,
        device: profile.session.device,
        totalDuration: profile.session.totalDuration,
        kernelLaunches: topKernels,
        memoryTransfers: profile.session.memoryTransfers.slice(0, 5),
        metrics: profile.session.metrics.filter((_, i) => i % 50 === 0),
        createdAt: profile.session.createdAt
      } as any,
      hints: profile.hints.slice(0, 5)
    };
  }

  static generateSessionSummary(profile: ParsedProfile): string {
    const { session, hints } = profile;
    const totalKernels = session.kernelLaunches.length;
    const avgOccupancy = session.kernelLaunches.reduce((sum, k) => sum + (k.occupancy || 0), 0) / totalKernels;
    const totalDuration = (session.totalDuration / 1000000000).toFixed(2);
    
    const highSeverityIssues = hints.filter(h => h.severity === 'high').length;
    const mediumSeverityIssues = hints.filter(h => h.severity === 'medium').length;
    
    return `GPU Profile Summary:
Device: ${session.device}
Duration: ${totalDuration}s
Kernels: ${totalKernels}
Avg Occupancy: ${(avgOccupancy * 100).toFixed(1)}%
Issues: ${highSeverityIssues} high, ${mediumSeverityIssues} medium

Generated by GPU Profiler - https://profiler.rightnowai.co`;
  }
}

// URL parameter utilities
export const UrlUtils = {
  getShareParam(): string | null {
    if (typeof window === 'undefined') return null;
    
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('share');
  },

  updateUrl(params: Record<string, string>) {
    if (typeof window === 'undefined') return;
    
    const url = new URL(window.location.href);
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.set(key, value);
    });
    
    window.history.pushState({}, '', url.toString());
  },

  clearShareParam() {
    if (typeof window === 'undefined') return;
    
    const url = new URL(window.location.href);
    url.searchParams.delete('share');
    window.history.replaceState({}, '', url.toString());
  }
};