"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { KernelLaunch } from "../../types/profiling";
import { formatDuration } from "../../lib/utils";
import { Search, RotateCcw } from "lucide-react";

interface FlameNode {
  name: string;
  value: number;
  children?: FlameNode[];
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  depth?: number;
  kernelId?: string;
  kernel?: KernelLaunch;
}

interface FlameGraphProps {
  kernels: KernelLaunch[];
  width?: number;
  height?: number;
  onKernelSelect?: (kernel: KernelLaunch) => void;
}

export function FlameGraph({ 
  kernels, 
  width = 800, 
  height = 400,
  onKernelSelect 
}: FlameGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedKernel, setSelectedKernel] = useState<KernelLaunch | null>(null);
  const [flameData, setFlameData] = useState<FlameNode | null>(null);

  // Process kernels into flame graph hierarchy
  useEffect(() => {
    if (kernels.length === 0) return;

    const hierarchyData = buildFlameHierarchy(kernels);
    setFlameData(hierarchyData);
  }, [kernels]);

  const buildFlameHierarchy = (kernels: KernelLaunch[]): FlameNode => {
    // Group kernels by function name patterns
    const groups = new Map<string, KernelLaunch[]>();
    
    kernels.forEach(kernel => {
      // Extract base function name (remove template parameters, indices, etc.)
      let baseName = kernel.name
        .replace(/_\d+$/, '') // Remove trailing numbers
        .replace(/<.*?>/, '') // Remove template parameters
        .replace(/\[.*?\]/, '') // Remove array indices
        .split('(')[0]; // Remove function parameters
      
      // Further simplify common CUDA kernel patterns
      if (baseName.includes('gemm')) baseName = 'GEMM_OPERATIONS';
      else if (baseName.includes('conv')) baseName = 'CONVOLUTION_OPS';
      else if (baseName.includes('reduce')) baseName = 'REDUCTION_OPS';
      else if (baseName.includes('add') || baseName.includes('mul') || baseName.includes('div')) baseName = 'ELEMENTWISE_OPS';
      else if (baseName.includes('cudnn')) baseName = 'CUDNN_OPERATIONS';
      else if (baseName.includes('cublas')) baseName = 'CUBLAS_OPERATIONS';
      else if (baseName.includes('cufft')) baseName = 'CUFFT_OPERATIONS';
      
      if (!groups.has(baseName)) {
        groups.set(baseName, []);
      }
      groups.get(baseName)!.push(kernel);
    });

    // Build hierarchy
    const children: FlameNode[] = [];
    groups.forEach((groupKernels, groupName) => {
      const totalDuration = groupKernels.reduce((sum, k) => sum + k.duration, 0);
      
      // For groups with many similar kernels, create sub-groups
      if (groupKernels.length > 5) {
        const subGroups = new Map<string, KernelLaunch[]>();
        groupKernels.forEach(kernel => {
          const subName = kernel.name.length > 20 
            ? kernel.name.substring(0, 20) + '...'
            : kernel.name;
          if (!subGroups.has(subName)) {
            subGroups.set(subName, []);
          }
          subGroups.get(subName)!.push(kernel);
        });

        const subChildren: FlameNode[] = [];
        subGroups.forEach((subKernels, subName) => {
          const subDuration = subKernels.reduce((sum, k) => sum + k.duration, 0);
          
          // Create individual kernel nodes
          const kernelChildren = subKernels.map(kernel => ({
            name: `${kernel.name}_${kernel.streamId}`,
            value: kernel.duration,
            kernelId: kernel.id,
            kernel
          }));

          subChildren.push({
            name: subName,
            value: subDuration,
            children: kernelChildren
          });
        });

        children.push({
          name: groupName,
          value: totalDuration,
          children: subChildren
        });
      } else {
        // Small groups - directly add kernels
        const kernelChildren = groupKernels.map(kernel => ({
          name: kernel.name,
          value: kernel.duration,
          kernelId: kernel.id,
          kernel
        }));

        children.push({
          name: groupName,
          value: totalDuration,
          children: kernelChildren
        });
      }
    });

    const totalTime = kernels.reduce((sum, k) => sum + k.duration, 0);
    return {
      name: "GPU_EXECUTION",
      value: totalTime,
      children: children.sort((a, b) => b.value - a.value)
    };
  };

  useEffect(() => {
    if (!svgRef.current || !flameData) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 10, right: 10, bottom: 10, left: 10 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create hierarchy
    const root = d3.hierarchy(flameData)
      .sum(d => d.value)
      .sort((a, b) => (b.value || 0) - (a.value || 0));

    // Calculate layout
    const maxDepth = root.height;
    const boxHeight = Math.max(18, innerHeight / (maxDepth + 1));
    
    // Position nodes
    const positionNodes = (node: d3.HierarchyNode<FlameNode>, x: number, width: number, depth: number) => {
      if (node.data) {
        node.data.x = x;
        node.data.y = depth * boxHeight;
        node.data.width = width;
        node.data.height = boxHeight;
        node.data.depth = depth;
      }

      if (node.children) {
        let currentX = x;
        const totalValue = node.value || 1;
        
        node.children.forEach(child => {
          const childWidth = width * ((child.value || 0) / totalValue);
          positionNodes(child, currentX, childWidth, depth + 1);
          currentX += childWidth;
        });
      }
    };

    positionNodes(root, 0, innerWidth, 0);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Get all nodes
    const allNodes = root.descendants();

    // Filter by search term
    const filteredNodes = searchTerm 
      ? allNodes.filter(d => d.data.name.toLowerCase().includes(searchTerm.toLowerCase()))
      : allNodes;

    // Draw flame graph rectangles
    const rects = g.selectAll(".flame-rect")
      .data(filteredNodes)
      .enter()
      .append("rect")
      .attr("class", "flame-rect")
      .attr("x", d => d.data.x || 0)
      .attr("y", d => d.data.y || 0)
      .attr("width", d => Math.max(0, (d.data.width || 0) - 1))
      .attr("height", d => (d.data.height || boxHeight) - 1)
      .attr("fill", d => {
        const depth = d.data.depth || 0;
        if (d.data.kernel) {
          // Color by occupancy for individual kernels
          const occupancy = d.data.kernel.occupancy || 0.5;
          if (occupancy > 0.8) return "#10b981";
          if (occupancy > 0.5) return "#f59e0b";
          return "#ef4444";
        } else {
          // Color by depth for groups
          const colors = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#c084fc"];
          return colors[depth % colors.length];
        }
      })
      .attr("stroke", d => selectedKernel?.id === d.data.kernelId ? "#000" : "none")
      .attr("stroke-width", 2)
      .style("cursor", d => d.data.kernel ? "pointer" : "default")
      .on("mouseover", function(event, d) {
        if ((d.data.width || 0) < 2) return; // Skip tiny rectangles

        const tooltip = d3.select("body")
          .append("div")
          .attr("class", "flame-tooltip")
          .style("position", "absolute")
          .style("background", "rgba(0,0,0,0.9)")
          .style("color", "white")
          .style("padding", "8px")
          .style("border-radius", "2px")
          .style("font-family", "JetBrains Mono, monospace")
          .style("font-size", "12px")
          .style("pointer-events", "none")
          .style("z-index", "1000");

        let tooltipContent = `<div><strong>${d.data.name}</strong></div>`;
        tooltipContent += `<div>Duration: ${formatDuration(d.data.value)}</div>`;
        tooltipContent += `<div>Percentage: ${((d.data.value / (flameData?.value || 1)) * 100).toFixed(1)}%</div>`;
        
        if (d.data.kernel) {
          tooltipContent += `<div>Occupancy: ${((d.data.kernel.occupancy || 0) * 100).toFixed(1)}%</div>`;
          tooltipContent += `<div>Stream: ${d.data.kernel.streamId}</div>`;
        }

        tooltip.html(tooltipContent)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 10) + "px");

        d3.select(this).attr("opacity", 0.8);
      })
      .on("mouseout", function() {
        d3.selectAll(".flame-tooltip").remove();
        d3.select(this).attr("opacity", 1);
      })
      .on("click", function(event, d) {
        if (d.data.kernel) {
          setSelectedKernel(d.data.kernel);
          onKernelSelect?.(d.data.kernel);
          
          // Update stroke for all rectangles
          g.selectAll(".flame-rect")
            .attr("stroke", node => node.data.kernelId === d.data.kernelId ? "#000" : "none");
        }
      });

    // Add text labels
    g.selectAll(".flame-text")
      .data(filteredNodes)
      .enter()
      .append("text")
      .attr("class", "flame-text")
      .attr("x", d => (d.data.x || 0) + 4)
      .attr("y", d => (d.data.y || 0) + (boxHeight / 2))
      .attr("dy", "0.35em")
      .attr("fill", "white")
      .attr("font-family", "JetBrains Mono, monospace")
      .attr("font-size", "10px")
      .attr("pointer-events", "none")
      .text(d => {
        const width = d.data.width || 0;
        if (width < 30) return ""; // Don't show text for very small boxes
        
        const maxChars = Math.floor(width / 6);
        const name = d.data.name;
        return name.length > maxChars ? name.slice(0, maxChars - 2) + ".." : name;
      })
      .each(function(d) {
        // Hide text if it doesn't fit
        const textWidth = (this as SVGTextElement).getComputedTextLength();
        if (textWidth > ((d.data.width || 0) - 8)) {
          d3.select(this).style("display", "none");
        }
      });

  }, [flameData, width, height, searchTerm, selectedKernel]);

  const handleReset = () => {
    setSearchTerm("");
    setSelectedKernel(null);
  };

  return (
    <div className="border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
      <div className="flex items-center justify-between p-3 border-b border-neutral-200 dark:border-neutral-700">
        <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
          FLAME_GRAPH
        </div>
        <div className="flex items-center space-x-2">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-3 w-3 text-neutral-400" />
            <input
              type="text"
              placeholder="Search kernels..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-7 pr-3 py-1 text-xs font-jetbrains bg-neutral-100 dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 text-neutral-800 dark:text-white placeholder:text-neutral-400"
              style={{borderRadius: '2px', width: '150px'}}
            />
          </div>
          <button
            onClick={handleReset}
            className="p-1 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
            style={{borderRadius: '2px'}}
          >
            <RotateCcw className="h-4 w-4 text-neutral-600 dark:text-neutral-400" />
          </button>
        </div>
      </div>
      <div className="p-4">
        <svg
          ref={svgRef}
          width={width}
          height={height}
          style={{ overflow: "hidden" }}
        />
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center space-x-4 font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-green-500" style={{borderRadius: '1px'}}></div>
              <span>HIGH_OCC</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-yellow-500" style={{borderRadius: '1px'}}></div>
              <span>MED_OCC</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-3 h-3 bg-red-500" style={{borderRadius: '1px'}}></div>
              <span>LOW_OCC</span>
            </div>
          </div>
          {flameData && (
            <div className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
              TOTAL: {formatDuration(flameData.value)} • {kernels.length} KERNELS
            </div>
          )}
        </div>
        {selectedKernel && (
          <div className="mt-4 p-3 bg-neutral-50 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
            <div className="font-jetbrains text-xs font-medium text-neutral-800 dark:text-white mb-2">
              SELECTED: {selectedKernel.name}
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
              <div>Duration: {formatDuration(selectedKernel.duration)}</div>
              <div>Stream: {selectedKernel.streamId}</div>
              <div>Occupancy: {((selectedKernel.occupancy || 0) * 100).toFixed(1)}%</div>
              <div>SMem: {(selectedKernel.sharedMemory / 1024).toFixed(1)}KB</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}