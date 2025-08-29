"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { KernelLaunch, MemoryTransfer } from "../../types/profiling";
import { formatDuration } from "../../lib/utils";
import { ZoomIn, ZoomOut, RotateCcw } from "lucide-react";

interface TimelineProps {
  kernels: KernelLaunch[];
  memoryTransfers: MemoryTransfer[];
  width?: number;
  height?: number;
  onKernelSelect?: (kernel: KernelLaunch) => void;
}

export function Timeline({ 
  kernels, 
  memoryTransfers, 
  width = 800, 
  height = 400,
  onKernelSelect 
}: TimelineProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedKernel, setSelectedKernel] = useState<KernelLaunch | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState(0);

  const margin = { top: 20, right: 20, bottom: 40, left: 80 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  useEffect(() => {
    if (!svgRef.current || kernels.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Calculate time domain
    const allEvents = [
      ...kernels.map(k => ({ start: k.startTime, end: k.startTime + k.duration })),
      ...memoryTransfers.map(m => ({ start: m.startTime, end: m.startTime + m.duration }))
    ];

    const minTime = d3.min(allEvents, d => d.start) || 0;
    const maxTime = d3.max(allEvents, d => d.end) || 1000000000;

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([minTime, maxTime])
      .range([0, innerWidth * zoomLevel]);

    const streamScale = d3.scaleBand()
      .domain([...new Set(kernels.map(k => k.streamId))].map(String))
      .range([0, innerHeight * 0.6])
      .padding(0.1);

    const transferY = innerHeight * 0.7;
    const transferHeight = innerHeight * 0.25;

    // Create main group with transform
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left + panOffset},${margin.top})`);

    // Add background grid
    const xTicks = xScale.ticks(10);
    g.selectAll(".grid-line")
      .data(xTicks)
      .enter()
      .append("line")
      .attr("class", "grid-line")
      .attr("x1", d => xScale(d))
      .attr("x2", d => xScale(d))
      .attr("y1", 0)
      .attr("y2", innerHeight)
      .attr("stroke", "#e5e5e5")
      .attr("stroke-width", 0.5)
      .attr("opacity", 0.3);

    // Draw kernel launches
    const kernelGroups = g.selectAll(".kernel")
      .data(kernels)
      .enter()
      .append("g")
      .attr("class", "kernel")
      .style("cursor", "pointer");

    kernelGroups
      .append("rect")
      .attr("x", d => xScale(d.startTime))
      .attr("y", d => streamScale(String(d.streamId)) || 0)
      .attr("width", d => Math.max(2, xScale(d.startTime + d.duration) - xScale(d.startTime)))
      .attr("height", streamScale.bandwidth())
      .attr("fill", d => {
        if (!d.occupancy) return "#3b82f6";
        if (d.occupancy > 0.8) return "#10b981"; // High occupancy - green
        if (d.occupancy > 0.5) return "#f59e0b"; // Medium occupancy - yellow  
        return "#ef4444"; // Low occupancy - red
      })
      .attr("stroke", d => selectedKernel?.id === d.id ? "#000" : "none")
      .attr("stroke-width", 2)
      .attr("opacity", 0.8)
      .on("mouseover", function(event, d) {
        // Create tooltip
        const tooltip = d3.select("body")
          .append("div")
          .attr("class", "tooltip")
          .style("position", "absolute")
          .style("background", "rgba(0,0,0,0.9)")
          .style("color", "white")
          .style("padding", "8px")
          .style("border-radius", "2px")
          .style("font-family", "JetBrains Mono, monospace")
          .style("font-size", "12px")
          .style("pointer-events", "none")
          .style("z-index", "1000");

        tooltip.html(`
          <div><strong>${d.name}</strong></div>
          <div>Duration: ${formatDuration(d.duration)}</div>
          <div>Stream: ${d.streamId}</div>
          <div>Occupancy: ${((d.occupancy || 0) * 100).toFixed(1)}%</div>
          <div>Block: ${d.blockDim.x}×${d.blockDim.y}×${d.blockDim.z}</div>
          <div>Grid: ${d.gridDim.x}×${d.gridDim.y}×${d.gridDim.z}</div>
        `)
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 10) + "px");

        d3.select(this).attr("opacity", 1);
      })
      .on("mouseout", function() {
        d3.selectAll(".tooltip").remove();
        d3.select(this).attr("opacity", 0.8);
      })
      .on("click", function(event, d) {
        setSelectedKernel(d);
        onKernelSelect?.(d);
        
        // Update stroke for all rectangles
        g.selectAll(".kernel rect")
          .attr("stroke", k => k.id === d.id ? "#000" : "none");
      });

    // Add kernel labels for larger kernels
    kernelGroups
      .filter(d => xScale(d.startTime + d.duration) - xScale(d.startTime) > 60)
      .append("text")
      .attr("x", d => xScale(d.startTime) + 4)
      .attr("y", d => (streamScale(String(d.streamId)) || 0) + streamScale.bandwidth() / 2)
      .attr("dy", "0.35em")
      .attr("fill", "white")
      .attr("font-family", "JetBrains Mono, monospace")
      .attr("font-size", "10px")
      .text(d => {
        const width = xScale(d.startTime + d.duration) - xScale(d.startTime);
        const maxChars = Math.floor(width / 6);
        return d.name.length > maxChars ? d.name.slice(0, maxChars - 2) + ".." : d.name;
      });

    // Draw memory transfers
    const transferGroups = g.selectAll(".transfer")
      .data(memoryTransfers)
      .enter()
      .append("g")
      .attr("class", "transfer");

    transferGroups
      .append("rect")
      .attr("x", d => xScale(d.startTime))
      .attr("y", transferY)
      .attr("width", d => Math.max(1, xScale(d.startTime + d.duration) - xScale(d.startTime)))
      .attr("height", transferHeight)
      .attr("fill", d => {
        switch (d.type) {
          case 'H2D': return "#8b5cf6"; // Purple
          case 'D2H': return "#06b6d4"; // Cyan
          case 'D2D': return "#84cc16"; // Lime
          default: return "#6b7280"; // Gray
        }
      })
      .attr("opacity", 0.7);

    // Add stream labels
    const streamIds = [...new Set(kernels.map(k => k.streamId))];
    g.selectAll(".stream-label")
      .data(streamIds)
      .enter()
      .append("text")
      .attr("class", "stream-label")
      .attr("x", -10)
      .attr("y", d => (streamScale(String(d)) || 0) + streamScale.bandwidth() / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", "end")
      .attr("font-family", "JetBrains Mono, monospace")
      .attr("font-size", "11px")
      .attr("fill", "#666")
      .text(d => `STREAM_${d}`);

    // Add memory transfer label
    g.append("text")
      .attr("x", -10)
      .attr("y", transferY + transferHeight / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", "end")
      .attr("font-family", "JetBrains Mono, monospace")
      .attr("font-size", "11px")
      .attr("fill", "#666")
      .text("MEMORY");

    // Add time axis
    const timeAxis = d3.axisBottom(xScale)
      .tickFormat(d => {
        const ms = d / 1000000;
        return ms < 1000 ? `${ms.toFixed(1)}ms` : `${(ms / 1000).toFixed(2)}s`;
      })
      .ticks(8);

    g.append("g")
      .attr("transform", `translate(0, ${innerHeight})`)
      .call(timeAxis)
      .selectAll("text")
      .attr("font-family", "JetBrains Mono, monospace")
      .attr("font-size", "10px");


  }, [kernels, memoryTransfers, width, height, selectedKernel, zoomLevel, panOffset]);

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.5, 10));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.5, 1));
    setPanOffset(0); // Reset pan when zooming out
  };

  const handleReset = () => {
    setZoomLevel(1);
    setPanOffset(0);
    setSelectedKernel(null);
  };

  return (
    <div className="border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
      <div className="flex items-center justify-between p-3 border-b border-neutral-200 dark:border-neutral-700">
        <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
          TIMELINE_VIEW
        </div>
        <div className="flex items-center space-x-1">
          <button
            onClick={handleZoomIn}
            className="p-1 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
            style={{borderRadius: '2px'}}
            disabled={zoomLevel >= 10}
          >
            <ZoomIn className="h-4 w-4 text-neutral-600 dark:text-neutral-400" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-1 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
            style={{borderRadius: '2px'}}
            disabled={zoomLevel <= 1}
          >
            <ZoomOut className="h-4 w-4 text-neutral-600 dark:text-neutral-400" />
          </button>
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
        <div className="flex items-start space-x-4">
          <svg
            ref={svgRef}
            width={width}
            height={height}
            style={{ overflow: "hidden" }}
          />
          
          {/* External Legend */}
          <div className="flex-shrink-0 min-w-[120px]">
            <div className="font-jetbrains text-xs font-medium text-neutral-800 dark:text-white mb-2">
              LEGEND
            </div>
            <div className="space-y-1.5">
              {[
                { color: "#10b981", label: "HIGH_OCC" },
                { color: "#f59e0b", label: "MED_OCC" },
                { color: "#ef4444", label: "LOW_OCC" },
                { color: "#8b5cf6", label: "H2D" },
                { color: "#06b6d4", label: "D2H" },
                { color: "#84cc16", label: "D2D" }
              ].map((item, i) => (
                <div key={i} className="flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 flex-shrink-0"
                    style={{
                      backgroundColor: item.color,
                      borderRadius: '1px'
                    }}
                  />
                  <span className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                    {item.label}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
        
        {selectedKernel && (
          <div className="mt-4 p-3 bg-neutral-50 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
            <div className="font-jetbrains text-xs font-medium text-neutral-800 dark:text-white mb-2">
              SELECTED_KERNEL: {selectedKernel.name}
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
              <div>Duration: {formatDuration(selectedKernel.duration)}</div>
              <div>Stream: {selectedKernel.streamId}</div>
              <div>Occupancy: {((selectedKernel.occupancy || 0) * 100).toFixed(1)}%</div>
              <div>Registers: {selectedKernel.registers}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}