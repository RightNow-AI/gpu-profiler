"use client";

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { KernelLaunch, MemoryTransfer } from '../../types/profiling';

interface TimelinePreviewProps {
  kernels: KernelLaunch[];
  memoryTransfers: MemoryTransfer[];
  width?: number;
  height?: number;
}

export function TimelinePreview({ 
  kernels, 
  memoryTransfers, 
  width = 350, 
  height = 120 
}: TimelinePreviewProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 20, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Get time extent
    const startTime = Math.min(...kernels.map(k => k.startTime));
    const endTime = Math.max(...kernels.map(k => k.startTime + k.duration));
    const timeScale = d3.scaleLinear()
      .domain([startTime, endTime])
      .range([0, innerWidth]);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Timeline axis
    const xAxis = d3.axisBottom(timeScale)
      .tickFormat(d => `${(d as number / 1000000).toFixed(0)}ms`)
      .ticks(4);

    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .selectAll('text')
      .style('font-family', 'JetBrains Mono, monospace')
      .style('font-size', '9px');

    // Stream lanes
    const streamIds = [...new Set(kernels.map(k => k.streamId))].slice(0, 3); // Max 3 streams for preview
    const streamScale = d3.scaleBand()
      .domain(streamIds.map(String))
      .range([0, innerHeight - 20])
      .padding(0.1);

    // Draw kernels
    kernels.forEach((kernel, i) => {
      if (!streamIds.includes(kernel.streamId)) return;
      
      const x = timeScale(kernel.startTime);
      const width = timeScale(kernel.startTime + kernel.duration) - x;
      const y = streamScale(String(kernel.streamId)) || 0;
      const height = streamScale.bandwidth();

      // Color based on duration
      const maxDuration = Math.max(...kernels.map(k => k.duration));
      const intensity = kernel.duration / maxDuration;
      const color = d3.interpolateBlues(0.3 + intensity * 0.7);

      g.append('rect')
        .attr('x', x)
        .attr('y', y)
        .attr('width', Math.max(width, 1))
        .attr('height', height)
        .attr('fill', color)
        .attr('stroke', '#374151')
        .attr('stroke-width', 0.5)
        .style('cursor', 'pointer');

      // Kernel name (only for wider ones)
      if (width > 30) {
        g.append('text')
          .attr('x', x + width / 2)
          .attr('y', y + height / 2)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .style('font-family', 'JetBrains Mono, monospace')
          .style('font-size', '8px')
          .style('fill', '#1f2937')
          .text(kernel.name.split('_')[0].substring(0, 8));
      }
    });

    // Stream labels
    streamIds.forEach(streamId => {
      const y = (streamScale(String(streamId)) || 0) + streamScale.bandwidth() / 2;
      g.append('text')
        .attr('x', -10)
        .attr('y', y)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .style('font-family', 'JetBrains Mono, monospace')
        .style('font-size', '9px')
        .style('fill', '#6b7280')
        .text(`S${streamId}`);
    });

  }, [kernels, memoryTransfers, width, height]);

  return (
    <div className="border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
      <div className="p-2 border-b border-neutral-200 dark:border-neutral-700">
        <div className="font-jetbrains text-xs font-medium text-neutral-800 dark:text-white">
          TIMELINE_PREVIEW
        </div>
      </div>
      <div className="p-2">
        <svg
          ref={svgRef}
          width={width}
          height={height}
          className="overflow-visible"
        />
      </div>
    </div>
  );
}