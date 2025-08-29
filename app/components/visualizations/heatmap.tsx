"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { GPUMetrics } from "../../types/profiling";
import { formatPercentage } from "../../lib/utils";
import { RotateCcw, TrendingUp, Thermometer, Zap } from "lucide-react";

interface HeatmapProps {
  metrics: GPUMetrics[];
  width?: number;
  height?: number;
  metricType?: 'utilization' | 'memory' | 'temperature' | 'power';
}

export function Heatmap({ 
  metrics, 
  width = 800, 
  height = 300,
  metricType = 'utilization'
}: HeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedMetric, setSelectedMetric] = useState<'utilization' | 'memory' | 'temperature' | 'power'>(metricType);
  const [hoveredPoint, setHoveredPoint] = useState<GPUMetrics | null>(null);

  const margin = { top: 20, right: 60, bottom: 40, left: 60 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  useEffect(() => {
    if (!svgRef.current || metrics.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Calculate time bins (group metrics into time buckets for better visualization)
    const timeRange = d3.extent(metrics, d => d.timestamp) as [number, number];
    const timeBins = 60; // Number of time bins
    const binSize = (timeRange[1] - timeRange[0]) / timeBins;
    
    const binnedData: Array<{
      timeStart: number;
      timeEnd: number;
      value: number;
      count: number;
      originalMetrics: GPUMetrics[]
    }> = [];

    for (let i = 0; i < timeBins; i++) {
      const binStart = timeRange[0] + i * binSize;
      const binEnd = binStart + binSize;
      
      const binMetrics = metrics.filter(m => 
        m.timestamp >= binStart && m.timestamp < binEnd
      );

      if (binMetrics.length > 0) {
        const avgValue = d3.mean(binMetrics, d => {
          switch (selectedMetric) {
            case 'utilization': return d.utilization;
            case 'memory': return d.memoryUtilization;
            case 'temperature': return (d.temperature || 70) / 100; // Normalize to 0-1
            case 'power': return (d.powerDraw || 300) / 500; // Normalize to 0-1 (assume max 500W)
            default: return d.utilization;
          }
        }) || 0;

        binnedData.push({
          timeStart: binStart,
          timeEnd: binEnd,
          value: avgValue,
          count: binMetrics.length,
          originalMetrics: binMetrics
        });
      }
    }

    // Create scales
    const xScale = d3.scaleLinear()
      .domain(timeRange)
      .range([0, innerWidth]);

    const yScale = d3.scaleBand()
      .domain(['GPU_METRICS']) // Single row for now, could expand to multiple metrics
      .range([0, innerHeight])
      .padding(0.1);

    const colorScale = d3.scaleSequential()
      .domain([0, 1])
      .interpolator(getColorInterpolator(selectedMetric));

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Draw heatmap cells
    g.selectAll(".heatmap-cell")
      .data(binnedData)
      .enter()
      .append("rect")
      .attr("class", "heatmap-cell")
      .attr("x", d => xScale(d.timeStart))
      .attr("y", yScale('GPU_METRICS') || 0)
      .attr("width", d => Math.max(1, xScale(d.timeEnd) - xScale(d.timeStart)))
      .attr("height", yScale.bandwidth())
      .attr("fill", d => colorScale(d.value))
      .attr("stroke", "white")
      .attr("stroke-width", 0.5)
      .style("cursor", "pointer")
      .on("mouseover", function(event, d) {
        const avgMetric = d.originalMetrics[0]; // Use first metric as representative
        setHoveredPoint(avgMetric);

        const tooltip = d3.select("body")
          .append("div")
          .attr("class", "heatmap-tooltip")
          .style("position", "absolute")
          .style("background", "rgba(0,0,0,0.9)")
          .style("color", "white")
          .style("padding", "8px")
          .style("border-radius", "2px")
          .style("font-family", "JetBrains Mono, monospace")
          .style("font-size", "12px")
          .style("pointer-events", "none")
          .style("z-index", "1000");

        const timeMs = (d.timeStart / 1000000);
        const timeStr = timeMs < 1000 ? `${timeMs.toFixed(1)}ms` : `${(timeMs / 1000).toFixed(2)}s`;
        
        let valueStr = "";
        switch (selectedMetric) {
          case 'utilization':
            valueStr = formatPercentage(d.value);
            break;
          case 'memory':
            valueStr = formatPercentage(d.value);
            break;
          case 'temperature':
            valueStr = `${(d.value * 100).toFixed(1)}°C`;
            break;
          case 'power':
            valueStr = `${(d.value * 500).toFixed(0)}W`;
            break;
        }

        tooltip.html(`
          <div><strong>Time: ${timeStr}</strong></div>
          <div>${getMetricLabel(selectedMetric)}: ${valueStr}</div>
          <div>Samples: ${d.count}</div>
        `)
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 10) + "px");

        d3.select(this).attr("opacity", 0.8);
      })
      .on("mouseout", function() {
        d3.selectAll(".heatmap-tooltip").remove();
        setHoveredPoint(null);
        d3.select(this).attr("opacity", 1);
      });

    // Add time axis
    const timeAxis = d3.axisBottom(xScale)
      .tickFormat(d => {
        const ms = d / 1000000;
        return ms < 1000 ? `${ms.toFixed(0)}ms` : `${(ms / 1000).toFixed(1)}s`;
      })
      .ticks(8);

    g.append("g")
      .attr("transform", `translate(0, ${innerHeight})`)
      .call(timeAxis)
      .selectAll("text")
      .attr("font-family", "JetBrains Mono, monospace")
      .attr("font-size", "10px");

    // Add metric label
    g.append("text")
      .attr("x", -10)
      .attr("y", (yScale('GPU_METRICS') || 0) + yScale.bandwidth() / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", "end")
      .attr("font-family", "JetBrains Mono, monospace")
      .attr("font-size", "11px")
      .attr("fill", "#666")
      .text(getMetricLabel(selectedMetric));


    // Calculate and display statistics
    const values = binnedData.map(d => d.value);
    const stats = {
      min: d3.min(values) || 0,
      max: d3.max(values) || 0,
      avg: d3.mean(values) || 0,
      med: d3.median(values) || 0
    };

    // Stats text
    const statsGroup = svg.append("g")
      .attr("transform", `translate(20, 20)`);

    const statsText = [
      `MIN: ${formatStatValue(stats.min, selectedMetric)}`,
      `MAX: ${formatStatValue(stats.max, selectedMetric)}`,
      `AVG: ${formatStatValue(stats.avg, selectedMetric)}`,
      `MED: ${formatStatValue(stats.med, selectedMetric)}`
    ];

    statsText.forEach((text, i) => {
      statsGroup.append("text")
        .attr("x", i * 80)
        .attr("y", 0)
        .attr("font-family", "JetBrains Mono, monospace")
        .attr("font-size", "10px")
        .attr("fill", "#666")
        .text(text);
    });

  }, [metrics, selectedMetric, width, height]);

  const getColorInterpolator = (metric: string) => {
    switch (metric) {
      case 'utilization':
        return d3.interpolateViridis;
      case 'memory':
        return d3.interpolateBlues;
      case 'temperature':
        return d3.interpolateYlOrRd;
      case 'power':
        return d3.interpolatePlasma;
      default:
        return d3.interpolateViridis;
    }
  };

  const getMetricLabel = (metric: string): string => {
    switch (metric) {
      case 'utilization': return 'GPU_UTIL';
      case 'memory': return 'MEM_UTIL';
      case 'temperature': return 'TEMP';
      case 'power': return 'POWER';
      default: return 'METRIC';
    }
  };

  const formatStatValue = (value: number, metric: string): string => {
    switch (metric) {
      case 'utilization':
      case 'memory':
        return formatPercentage(value);
      case 'temperature':
        return `${(value * 100).toFixed(0)}°C`;
      case 'power':
        return `${(value * 500).toFixed(0)}W`;
      default:
        return formatPercentage(value);
    }
  };

  const getMetricIcon = (metric: string) => {
    switch (metric) {
      case 'utilization': return TrendingUp;
      case 'memory': return TrendingUp;
      case 'temperature': return Thermometer;
      case 'power': return Zap;
      default: return TrendingUp;
    }
  };

  const handleReset = () => {
    setSelectedMetric('utilization');
    setHoveredPoint(null);
  };

  return (
    <div className="border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
      <div className="flex items-center justify-between p-3 border-b border-neutral-200 dark:border-neutral-700">
        <div className="font-jetbrains text-sm font-medium text-neutral-800 dark:text-white">
          HEATMAP_VIEW
        </div>
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1">
            {(['utilization', 'memory', 'temperature', 'power'] as const).map(metric => {
              const Icon = getMetricIcon(metric);
              return (
                <button
                  key={metric}
                  onClick={() => setSelectedMetric(metric)}
                  className={`flex items-center space-x-1 px-2 py-1 text-xs font-jetbrains transition-colors ${
                    selectedMetric === metric 
                      ? 'bg-neutral-800 dark:bg-white text-white dark:text-neutral-800'
                      : 'text-neutral-600 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-neutral-800'
                  }`}
                  style={{borderRadius: '2px'}}
                >
                  <Icon className="h-3 w-3" />
                  <span>{getMetricLabel(metric)}</span>
                </button>
              );
            })}
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
        <div className="flex items-start space-x-4">
          <svg
            ref={svgRef}
            width={width}
            height={height}
            style={{ overflow: "hidden" }}
          />
          
          {/* External Color Legend */}
          <div className="flex-shrink-0 min-w-[140px]">
            <div className="font-jetbrains text-xs font-medium text-neutral-800 dark:text-white mb-2">
              COLOR_SCALE
            </div>
            <div className="space-y-2">
              {/* Gradient bar */}
              <div className="relative">
                <div 
                  className="w-20 h-3"
                  style={{
                    background: selectedMetric === 'utilization' || selectedMetric === 'memory' 
                      ? 'linear-gradient(to right, #440154, #31688e, #35b779, #fde725)'
                      : selectedMetric === 'temperature' 
                      ? 'linear-gradient(to right, #0d1a26, #ffeda0, #feb24c, #f03b20)'
                      : 'linear-gradient(to right, #0f0f23, #6a1b9a, #ad42c7, #ff6ec7)',
                    borderRadius: '1px',
                    border: '1px solid #e5e7eb'
                  }}
                />
                <div className="flex justify-between mt-1 font-jetbrains text-xs text-neutral-500 dark:text-neutral-400">
                  <span>Low</span>
                  <span>High</span>
                </div>
              </div>
              
              {/* Value range */}
              <div className="font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
                {selectedMetric === 'utilization' || selectedMetric === 'memory' ? '0% - 100%'
                : selectedMetric === 'temperature' ? '60°C - 100°C'
                : selectedMetric === 'power' ? '150W - 500W'
                : '0% - 100%'}
              </div>
            </div>
          </div>
        </div>
        
        {hoveredPoint && (
          <div className="mt-4 p-3 bg-neutral-50 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700" style={{borderRadius: '2px'}}>
            <div className="font-jetbrains text-xs font-medium text-neutral-800 dark:text-white mb-2">
              POINT_DETAILS
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 font-jetbrains text-xs text-neutral-600 dark:text-neutral-400">
              <div>GPU Util: {formatPercentage(hoveredPoint.utilization)}</div>
              <div>Mem Util: {formatPercentage(hoveredPoint.memoryUtilization)}</div>
              {hoveredPoint.temperature && <div>Temp: {hoveredPoint.temperature.toFixed(0)}°C</div>}
              {hoveredPoint.powerDraw && <div>Power: {hoveredPoint.powerDraw.toFixed(0)}W</div>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}