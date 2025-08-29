"use client";

import { useState, useRef } from "react";
import { Upload, Loader2, FileText, AlertCircle, CheckCircle, X } from "lucide-react";
import { useProfilingStore } from "../store/profiling-store";
import { SupportedFormat, UploadedFile } from "../types/profiling";

interface FileUploadProps {
  onFileUploaded?: (file: UploadedFile) => void;
}

export function FileUpload({ onFileUploaded }: FileUploadProps) {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { isUploading, setUploadState, setError: setGlobalError } = useProfilingStore();

  const getSupportedFormat = (filename: string): SupportedFormat | null => {
    const ext = filename.toLowerCase();
    if (ext.endsWith('.nvprof')) return 'nvprof';
    if (ext.endsWith('.nsys-rep') || ext.endsWith('.nsys')) return 'nsys';
    if (ext.endsWith('.json')) return 'json';
    return null;
  };

  const validateFile = (file: File): string | null => {
    // Size check (max 100MB)
    if (file.size > 100 * 1024 * 1024) {
      return `File too large: ${(file.size / 1024 / 1024).toFixed(1)}MB (max 100MB)`;
    }

    // Format check
    const format = getSupportedFormat(file.name);
    if (!format) {
      return `Unsupported format. Use .nvprof, .nsys-rep, or .json files`;
    }

    return null;
  };

  const handleFiles = async (files: FileList) => {
    const fileArray = Array.from(files);
    const validFiles: UploadedFile[] = [];
    let hasErrors = false;

    setError(null);
    setGlobalError(null);

    // Validate all files first
    for (const file of fileArray) {
      const error = validateFile(file);
      if (error) {
        setError(error);
        hasErrors = true;
        break;
      }

      const format = getSupportedFormat(file.name)!;
      validFiles.push({ file, format });
    }

    if (hasErrors) return;

    // Process files
    setUploadState(true, 0);
    
    try {
      for (let i = 0; i < validFiles.length; i++) {
        const uploadFile = validFiles[i];
        setUploadState(true, (i / validFiles.length) * 100);

        // Read file content
        const content = await readFileContent(uploadFile.file);
        uploadFile.content = content;

        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 500));
        
        setUploadedFiles(prev => [...prev, uploadFile]);
        onFileUploaded?.(uploadFile);
      }
      
      setUploadState(false, 100);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to process file';
      setError(errorMsg);
      setGlobalError(errorMsg);
      setUploadState(false, 0);
    }
  };

  const readFileContent = (file: File): Promise<ArrayBuffer | string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as ArrayBuffer | string);
      reader.onerror = () => reject(new Error('Failed to read file'));
      
      if (file.name.endsWith('.json')) {
        reader.readAsText(file);
      } else {
        reader.readAsArrayBuffer(file);
      }
    });
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleFileSelect = () => {
    fileInputRef.current?.click();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  };

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div 
        className={`upload-area ${dragOver ? 'dragover' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center space-y-4">
          {isUploading ? (
            <Loader2 className="h-8 w-8 text-neutral-400 animate-spin" />
          ) : error ? (
            <AlertCircle className="h-8 w-8 text-red-500" />
          ) : uploadedFiles.length > 0 ? (
            <CheckCircle className="h-8 w-8 text-green-500" />
          ) : (
            <Upload className="h-8 w-8 text-neutral-400" />
          )}
          
          <div className="text-center">
            <h3 className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
              {isUploading ? "PROCESSING..." : 
               error ? "ERROR" :
               uploadedFiles.length > 0 ? "FILES_READY" : "DROP_FILES"}
            </h3>
            <p className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500 mt-1">
              .nvprof • .nsys-rep • .json
            </p>
          </div>

          {error && (
            <div className="text-center p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800" style={{borderRadius: '2px'}}>
              <p className="font-jetbrains text-xs text-red-600 dark:text-red-400">
                {error}
              </p>
            </div>
          )}

          {!isUploading && !error && (
            <button 
              onClick={handleFileSelect}
              className="rn-button max-w-xs flex items-center justify-center space-x-2"
            >
              <Upload className="h-3 w-3" />
              <span>SELECT</span>
            </button>
          )}
        </div>
      </div>

      {/* File List */}
      {uploadedFiles.length > 0 && (
        <div className="mt-6 space-y-2">
          <h4 className="font-jetbrains text-xs font-medium text-neutral-600 dark:text-neutral-400">
            UPLOADED_FILES ({uploadedFiles.length})
          </h4>
          {uploadedFiles.map((file, index) => (
            <div 
              key={index}
              className="flex items-center justify-between p-3 bg-neutral-50 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700"
              style={{borderRadius: '2px'}}
            >
              <div className="flex items-center space-x-3">
                <FileText className="h-4 w-4 text-neutral-500 dark:text-neutral-400" />
                <div>
                  <p className="font-jetbrains text-xs font-medium text-neutral-800 dark:text-white">
                    {file.file.name}
                  </p>
                  <p className="font-jetbrains text-xs text-neutral-500 dark:text-neutral-500">
                    {file.format.toUpperCase()} • {(file.file.size / 1024 / 1024).toFixed(1)}MB
                  </p>
                </div>
              </div>
              <button
                onClick={() => removeFile(index)}
                className="p-1 hover:bg-neutral-200 dark:hover:bg-neutral-700 transition-colors duration-100"
                style={{borderRadius: '2px'}}
              >
                <X className="h-3 w-3 text-neutral-500 dark:text-neutral-400" />
              </button>
            </div>
          ))}
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".nvprof,.nsys-rep,.nsys,.json"
        onChange={handleInputChange}
        className="hidden"
      />
    </div>
  );
}