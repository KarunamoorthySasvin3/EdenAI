import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartLegend,
  ChartPie,
} from "@/components/ui/chart";
import { Thermometer, Droplets, Wind, Cloud } from "lucide-react";

export function EnvironmentalImpact() {
  return (
    <section id="impact" className="w-full py-12 md:py-24 lg:py-32 bg-muted">
      <div className="container px-4 md:px-6">
        <div className="flex flex-col items-center justify-center space-y-4 text-center">
          <div className="inline-block rounded-lg bg-primary/10 px-3 py-1 text-sm text-primary mb-2">
            AI-Powered Climate Analytics
          </div>
          <div className="space-y-2">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">
              Climate Impact Dashboard
            </h2>
            <p className="max-w-[900px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
              Track your garden's contribution to climate change mitigation with
              real-time AI predictions and analytics.
            </p>
          </div>
        </div>
        <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-2 mt-12">
          <Card>
            <CardHeader>
              <CardTitle>Carbon Sequestration</CardTitle>
              <CardDescription>
                PyTorch-modeled CO₂ capture projections by garden type
              </CardDescription>
            </CardHeader>
            <CardContent className="flex justify-center">
              <div className="h-80 w-80">
                <ChartContainer>
                  <ChartPie data={carbonData} index="name" category="value" valueFormatter={(value) => `${value}`} colors={["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0"]} />
                  <ChartTooltip />
                  <ChartLegend />
                </ChartContainer>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Global Warming Contribution</CardTitle>
              <CardDescription>
                Your garden's impact on reducing greenhouse gas emissions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span>Carbon Offset Potential:</span>
                <span className="font-bold text-primary">342 kg CO₂/year</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Equivalent to:</span>
                <span className="font-bold">14 trees</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Cooling Effect:</span>
                <span className="font-bold">-3.2°C local temp</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Climate Resilience Score</CardTitle>
              <CardDescription>
                AI-generated assessment of your garden's ability to withstand
                climate extremes
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="flex flex-col items-center p-2 border rounded-md">
                  <Thermometer className="h-8 w-8 text-orange-500" />
                  <span className="mt-1">Heat Waves</span>
                  <span className="font-bold">82%</span>
                </div>
                <div className="flex flex-col items-center p-2 border rounded-md">
                  <Droplets className="h-8 w-8 text-blue-500" />
                  <span className="mt-1">Drought</span>
                  <span className="font-bold">76%</span>
                </div>
                <div className="flex flex-col items-center p-2 border rounded-md">
                  <Cloud className="h-8 w-8 text-gray-500" />
                  <span className="mt-1">Flooding</span>
                  <span className="font-bold">69%</span>
                </div>
                <div className="flex flex-col items-center p-2 border rounded-md">
                  <Wind className="h-8 w-8 text-teal-500" />
                  <span className="mt-1">Storms</span>
                  <span className="font-bold">85%</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Climate Projection Model</CardTitle>
              <CardDescription>
                PyTorch-powered predictions of your garden's climate adaptation
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                Our neural network analyzes weather patterns, soil conditions,
                and plant selections to project:
              </p>
              <ul className="list-disc pl-5 space-y-1">
                <li>Long-term carbon sequestration potential</li>
                <li>Microclimate stabilization effects</li>
                <li>Biodiversity support metrics</li>
                <li>Water conservation projections</li>
              </ul>
              <p className="text-sm text-muted-foreground mt-2">
                Learn how your garden can help mitigate climate change using the information given
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}

const carbonData = [
  { name: "Native Garden", value: 450 },
  { name: "Vegetable Garden", value: 280 },
  { name: "Water Garden", value: 320 },
  { name: "Forest Garden", value: 580 },
];
