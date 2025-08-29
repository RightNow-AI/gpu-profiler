"use client";

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { KernelLaunch } from '../../types/profiling';

interface FlamePreviewProps {
  kernels: KernelLaunch[];
  width?: number;
  height?: number;
}

export function FlamePreview({ kernels, width = 350, height = 120 }: FlamePreviewProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 10, right: 10, bottom: 20, left: 10 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create hierarchical data from kernels
    const sortedKernels = [...kernels].sort((a, b) => b.duration - a.duration);
    const totalDuration = sortedKernels.reduce((sum, k) => sum + k.duration, 0);
    
    // Create flame layers
    const layers = 3;
    const layerHeight = innerHeight / layers;
    
    let currentX = 0;
    let currentLayer = 0;
    
    sortedKernels.slice(0, 8).forEach((kernel, index) => {
      const kernelWidth = (kernel.duration / totalDuration) * innerWidth;
      
      if (currentX + kernelWidth > innerWidth || index % 3 === 0) {
        currentLayer = (currentLayer + 1) % layers;
        currentX = (currentLayer * 20) % (innerWidth / 3); // Stagger start positions
      }
      
      const x = currentX;
      const y = currentLayer * (layerHeight - 5);
      const w = Math.max(kernelWidth, 15);
      const h = layerHeight - 8;
      
      // Color based on performance
      const occupancy = kernel.occupancy || 0.5;
      const color = d3.interpolateRdYlGn(occupancy);
      
      g.append('rect')
        .attr('x', x)
        .attr('y', y)
        .attr('width', w)
        .attr('height', h)
        .attr('fill', color)
        .attr('stroke', '#374151')
        .attr('stroke-width', 0.5)
        .style('cursor', 'pointer');
      
      // Add text if wide enough
      if (w > 25) {
        g.append('text')
          .attr('x', x + w / 2)
          .attr('y', y + h / 2)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .style('font-family', 'JetBrains Mono, monospace')
          .style('font-size', '7px')
          .style('fill', occupancy > 0.5 ? '#1f2937' : '#f9fafb')
          .text(kernel.name.split('_')[0].substring(0, 6));
      }
      
      currentX += w + 2;
    });

    // Add legend
    g.append('text')
      .attr('x', 0)
      .attr('y', innerHeight + 15)
      .style('font-family', 'JetBrains Mono, monospace')
      .style('font-size', '8px')
      .style('fill', '#6b7280')
      .text('Width = Duration | Color = Occupancy');

  }, [kernels, width, height]);

  return (
    <div className="border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
      <div className="p-2 border-b border-neutral-200 dark:border-neutral-700">
        <div className="font-jetbrains text-xs font-medium text-neutral-800 dark:text-white">
          FLAME_GRAPH_PREVIEW
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