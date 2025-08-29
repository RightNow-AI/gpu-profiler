"use client";

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { GPUMetrics } from '../../types/profiling';

interface HeatmapPreviewProps {
  metrics: GPUMetrics[];
  width?: number;
  height?: number;
}

export function HeatmapPreview({ metrics, width = 350, height = 120 }: HeatmapPreviewProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !metrics.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 20, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create a simplified heatmap with sample data points
    const timePoints = 12; // Number of time segments
    const metricTypes = ['GPU_UTIL', 'MEMORY', 'TEMP'];
    
    const heatmapData: Array<{x: number, y: number, value: number, metric: string}> = [];
    
    metricTypes.forEach((metric, metricIndex) => {
      for (let timeIndex = 0; timeIndex < timePoints; timeIndex++) {
        // Generate realistic values based on metric type
        let value;
        if (metric === 'GPU_UTIL') {
          value = 0.3 + Math.sin(timeIndex * 0.8) * 0.4 + Math.random() * 0.2;
        } else if (metric === 'MEMORY') {
          value = 0.6 + Math.cos(timeIndex * 0.5) * 0.3 + Math.random() * 0.15;
        } else { // TEMP
          value = 0.4 + timeIndex * 0.03 + Math.random() * 0.1;
        }
        
        heatmapData.push({
          x: timeIndex,
          y: metricIndex,
          value: Math.max(0, Math.min(1, value)),
          metric
        });
      }
    });

    // Scales
    const xScale = d3.scaleBand()
      .domain(d3.range(timePoints).map(String))
      .range([0, innerWidth])
      .padding(0.02);

    const yScale = d3.scaleBand()
      .domain(metricTypes)
      .range([0, innerHeight])
      .padding(0.1);

    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, 1]);

    // Draw heatmap cells
    g.selectAll('.heatmap-cell')
      .data(heatmapData)
      .enter()
      .append('rect')
      .attr('class', 'heatmap-cell')
      .attr('x', d => xScale(String(d.x)) || 0)
      .attr('y', d => yScale(d.metric) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', '#1f2937')
      .attr('stroke-width', 0.5)
      .style('cursor', 'pointer');

    // Y-axis labels
    metricTypes.forEach(metric => {
      const y = (yScale(metric) || 0) + yScale.bandwidth() / 2;
      g.append('text')
        .attr('x', -5)
        .attr('y', y)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .style('font-family', 'JetBrains Mono, monospace')
        .style('font-size', '9px')
        .style('fill', '#6b7280')
        .text(metric);
    });

    // X-axis (time segments)
    const timeLabels = ['0ms', '50ms', '100ms', '150ms'];
    [0, 3, 6, 9].forEach((index, i) => {
      const x = (xScale(String(index)) || 0) + xScale.bandwidth() / 2;
      g.append('text')
        .attr('x', x)
        .attr('y', innerHeight + 15)
        .attr('text-anchor', 'middle')
        .style('font-family', 'JetBrains Mono, monospace')
        .style('font-size', '8px')
        .style('fill', '#6b7280')
        .text(timeLabels[i]);
    });

  }, [metrics, width, height]);

  return (
    <div className="border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
      <div className="p-2 border-b border-neutral-200 dark:border-neutral-700">
        <div className="font-jetbrains text-xs font-medium text-neutral-800 dark:text-white">
          HEATMAP_PREVIEW
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