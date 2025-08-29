import { NvprofParser } from './nvprof-parser';
import { NsysParser } from './nsys-parser';
import { JsonParser } from './json-parser';
import { SupportedFormat, UploadedFile, ParsedProfile } from '../../types/profiling';

export class ProfileParser {
  static async parseFile(uploadedFile: UploadedFile): Promise<ParsedProfile> {
    const { format, content } = uploadedFile;

    if (!content) {
      throw new Error('File content is empty');
    }

    try {
      switch (format) {
        case 'nvprof':
          return await NvprofParser.parse(content as ArrayBuffer);
        
        case 'nsys':
          return await NsysParser.parse(content as ArrayBuffer);
        
        case 'json':
          return await JsonParser.parse(content as string);
        
        default:
          throw new Error(`Unsupported format: ${format}`);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown parsing error';
      throw new Error(`Failed to parse ${format} file: ${errorMessage}`);
    }
  }

  static getSupportedFormats(): SupportedFormat[] {
    return ['nvprof', 'nsys', 'json'];
  }

  static getFormatDescription(format: SupportedFormat): string {
    switch (format) {
      case 'nvprof':
        return 'NVIDIA nvprof profiler output (.nvprof)';
      case 'nsys':
        return 'NVIDIA Nsight Systems report (.nsys-rep)';
      case 'json':
        return 'JSON trace export from CUDA tools';
      default:
        return 'Unknown format';
    }
  }

  static validateFileSize(file: File): void {
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
      throw new Error(`File size ${(file.size / 1024 / 1024).toFixed(1)}MB exceeds maximum allowed size of 100MB`);
    }
  }

  static detectFormat(filename: string): SupportedFormat | null {
    const name = filename.toLowerCase();
    if (name.endsWith('.nvprof')) return 'nvprof';
    if (name.endsWith('.nsys-rep') || name.endsWith('.nsys')) return 'nsys';
    if (name.endsWith('.json')) return 'json';
    return null;
  }
}

// Export parsers for individual use if needed
export { NvprofParser, NsysParser, JsonParser };