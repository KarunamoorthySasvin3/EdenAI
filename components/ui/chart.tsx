import type * as React from "react"

interface ChartContainerProps {
  children: React.ReactNode
}

export function ChartContainer({ children }: ChartContainerProps) {
  return <div className="relative w-full h-full">{children}</div>
}

type ChartTooltipProps = {}

export function ChartTooltip({}: ChartTooltipProps) {
  return <div className="absolute z-10 p-2 bg-white border rounded shadow-md">Tooltip</div>
}

type ChartLegendProps = {}

export function ChartLegend({}: ChartLegendProps) {
  return <div className="absolute z-10 p-2 bg-white border rounded shadow-md">Legend</div>
}

interface ChartPieProps {
  data: { name: string; value: number }[]
  index: string
  valueFormatter: (value: number) => string
  category: string
  className?: string
  colors: string[]
  children?: React.ReactNode
}

export function ChartPie({ data, index, valueFormatter, category, className, colors, children }: ChartPieProps) {
  return <div className={className}>ChartPie</div>
}

interface ChartBarProps {
  data: { month: string; sequestered: number }[]
  index: string
  categories: string[]
  colors: string[]
  valueFormatter: (value: number) => string
  showLegend: boolean
  showXAxis: boolean
  showYAxis: boolean
  showGridLines: boolean
  children?: React.ReactNode
}

export function ChartBar({
  data,
  index,
  categories,
  colors,
  valueFormatter,
  showLegend,
  showXAxis,
  showYAxis,
  showGridLines,
  children,
}: ChartBarProps) {
  return <div>ChartBar</div>
}

