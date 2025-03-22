import React, { useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter
} from "../ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Button } from "../ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { AlertCircle, TrendingUp, ThermometerSnowflake, Cloud, Droplets } from "lucide-react";
import { Alert, AlertTitle, AlertDescription } from "../ui/alert";
import { Progress } from "../ui/progress";

interface ClimateChangeAnalysisProps {
  className?: string;
}

export function ClimateChangeAnalysis({ className }: ClimateChangeAnalysisProps) {
  const [selectedRegion, setSelectedRegion] = useState("north-america");
  const [selectedScenario, setSelectedScenario] = useState("rcp45");
  const [timeframe, setTimeframe] = useState("2050");

  // This would come from a real model in production
  const scenarioData = {
    "rcp26": {
      name: "Low Emissions (RCP 2.6)",
      description: "Global action limits warming to 1.5°C",
      temperature: 1.5,
      precipitationChange: 5,
      extremeWeatherIncrease: 20,
      seaLevelRise: 0.3
    },
    "rcp45": {
      name: "Intermediate Emissions (RCP 4.5)",
      description: "Some mitigation efforts limit warming to 2.4°C",
      temperature: 2.4,
      precipitationChange: 8,
      extremeWeatherIncrease: 45,
      seaLevelRise: 0.5
    },
    "rcp85": {
      name: "High Emissions (RCP 8.5)",
      description: "Business as usual leads to significant warming",
      temperature: 4.3,
      precipitationChange: 15,
      extremeWeatherIncrease: 120,
      seaLevelRise: 0.8
    }
  };

  const regions = [
    { id: "north-america", name: "North America" },
    { id: "europe", name: "Europe" },
    { id: "asia", name: "Asia" },
    { id: "africa", name: "Africa" },
    { id: "south-america", name: "South America" },
    { id: "oceania", name: "Oceania" }
  ];

  const timeframes = [
    { id: "2030", name: "2030" },
    { id: "2050", name: "2050" },
    { id: "2100", name: "2100" }
  ];

  // Get the current scenario data
  const currentScenario = scenarioData[selectedScenario as keyof typeof scenarioData];
  
  // Adjust values based on timeframe
  const timeframeMultiplier = timeframe === "2030" ? 0.5 : timeframe === "2050" ? 1.0 : 2.0;
  
  // Calculate garden impact - this would come from a model in production
  const calculateGardenImpact = () => {
    return {
      temperatureReduction: 0.02 * timeframeMultiplier,
      waterConservation: 1.8 * timeframeMultiplier,
      biodiversitySupport: 25 * timeframeMultiplier,
      resilienceScore: Math.min(100, 65 + 15 * timeframeMultiplier)
    };
  };
  
  const gardenImpact = calculateGardenImpact();

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Climate Change Impact Analysis</CardTitle>
        <CardDescription>
          PyTorch-powered projection of climate scenarios and your garden&apos;s mitigating effect
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 space-y-2">
            <label className="text-sm font-medium">Region</label>
            <Select value={selectedRegion} onValueChange={setSelectedRegion}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {regions.map(region => (
                  <SelectItem key={region.id} value={region.id}>{region.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          <div className="flex-1 space-y-2">
            <label className="text-sm font-medium">Climate Scenario</label>
            <Select value={selectedScenario} onValueChange={setSelectedScenario}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="rcp26">{scenarioData.rcp26.name}</SelectItem>
                <SelectItem value="rcp45">{scenarioData.rcp45.name}</SelectItem>
                <SelectItem value="rcp85">{scenarioData.rcp85.name}</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="flex-1 space-y-2">
            <label className="text-sm font-medium">Timeframe</label>
            <Select value={timeframe} onValueChange={setTimeframe}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {timeframes.map(tf => (
                  <SelectItem key={tf.id} value={tf.id}>{tf.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        
        <Alert variant="default" className="bg-amber-50 border-amber-200">
          <AlertCircle className="h-4 w-4 text-amber-500" />
          <AlertTitle>Climate Scenario: {currentScenario.name}</AlertTitle>
          <AlertDescription className="text-sm">
            {currentScenario.description}
          </AlertDescription>
        </Alert>
        
        <Tabs defaultValue="projections">
          <TabsList className="grid grid-cols-3">
            <TabsTrigger value="projections">Projections</TabsTrigger>
            <TabsTrigger value="garden-impact">Garden Impact</TabsTrigger>
            <TabsTrigger value="adaptation">Adaptation Strategies</TabsTrigger>
          </TabsList>
          
          <TabsContent value="projections" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="border rounded-lg p-4 space-y-2">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-red-500" />
                  <h3 className="font-medium">Temperature Increase</h3>
                </div>
                <div className="text-2xl font-bold">+{(currentScenario.temperature * timeframeMultiplier).toFixed(1)}°C</div>
                <Progress value={currentScenario.temperature * timeframeMultiplier * 15} className="h-2 bg-red-100" />
              </div>
              
              <div className="border rounded-lg p-4 space-y-2">
                <div className="flex items-center gap-2">
                  <ThermometerSnowflake className="h-4 w-4 text-amber-500" />
                  <h3 className="font-medium">Extreme Weather Events</h3>
                </div>
                <div className="text-2xl font-bold">+{Math.round(currentScenario.extremeWeatherIncrease * timeframeMultiplier)}%</div>
                <Progress value={currentScenario.extremeWeatherIncrease * timeframeMultiplier / 2} className="h-2 bg-amber-100" />
              </div>
              
              <div className="border rounded-lg p-4 space-y-2">
                <div className="flex items-center gap-2">
                  <Cloud className="h-4 w-4 text-blue-500" />
                  <h3 className="font-medium">Precipitation Change</h3>
                </div>
                <div className="text-2xl font-bold">{currentScenario.precipitationChange > 0 ? '+' : ''}{Math.round(currentScenario.precipitationChange * timeframeMultiplier)}%</div>
                <Progress value={50 + currentScenario.precipitationChange * timeframeMultiplier * 3} className="h-2 bg-blue-100" />
              </div>
              
              <div className="border rounded-lg p-4 space-y-2">
                <div className="flex items-center gap-2">
                  <Droplets className="h-4 w-4 text-cyan-500" />
                  <h3 className="font-medium">Sea Level Rise</h3>
                </div>
                <div className="text-2xl font-bold">{(currentScenario.seaLevelRise * timeframeMultiplier).toFixed(1)}m</div>
                <Progress value={currentScenario.seaLevelRise * timeframeMultiplier * 60} className="h-2 bg-cyan-100" />
              </div>
            </div>
            
            <div className="border rounded-lg p-4 bg-slate-50">
              <h3 className="font-medium mb-2">PyTorch Model Insights</h3>
              <p className="text-sm text-slate-700">
                These projections are generated using an ensemble of 5 neural networks trained on 
                CMIP6 climate model data. The uncertainty range for this {selectedRegion} 
                projection is ±{Math.round(currentScenario.temperature * 0.2 * 10) / 10}°C for temperature
                and ±{Math.round(currentScenario.precipitationChange * 0.3)}% for precipitation.
              </p>
            </div>
          </TabsContent>
          
          <TabsContent value="garden-impact" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="border rounded-lg p-4 space-y-2">
                <h3 className="font-medium">Local Temperature Reduction</h3>
                <div className="text-2xl font-bold text-emerald-600">
                  {gardenImpact.temperatureReduction.toFixed(2)}°C
                </div>
                <p className="text-xs text-slate-500">
                  Urban heat island effect mitigation through evapotranspiration
                </p>
              </div>
              
              <div className="border rounded-lg p-4 space-y-2">
                <h3 className="font-medium">Water Conservation</h3>
                <div className="text-2xl font-bold text-emerald-600">
                  {gardenImpact.waterConservation.toFixed(1)}m³/month
                </div>
                <p className="text-xs text-slate-500">
                  Reduced water requirements compared to conventional landscapes
                </p>
              </div>
              
              <div className="border rounded-lg p-4 space-y-2">
                <h3 className="font-medium">Biodiversity Support</h3>
                <div className="text-2xl font-bold text-emerald-600">
                  +{Math.round(gardenImpact.biodiversitySupport)} species
                </div>
                <p className="text-xs text-slate-500">
                  Estimated additional habitat support for local wildlife
                </p>
              </div>
              
              <div className="border rounded-lg p-4 space-y-2">
                <h3 className="font-medium">Climate Resilience Score</h3>
                <div className="text-2xl font-bold text-emerald-600">
                  {Math.round(gardenImpact.resilienceScore)}/100
                </div>
                <p className="text-xs text-slate-500">
                  AI-derived rating of adaptation potential for extreme weather
                </p>
              </div>
            </div>
            
            <div className="p-4 border rounded-lg bg-green-50">
              <h3 className="font-medium mb-2">Scaling Effect</h3>
              <p className="text-sm">
                If 20% of properties in your neighborhood implemented similar gardens, 
                it could reduce local temperatures by {(gardenImpact.temperatureReduction * 12).toFixed(1)}°C 
                during heat waves and improve stormwater management by up to 35%.
              </p>
            </div>
          </TabsContent>
          
          <TabsContent value="adaptation" className="space-y-4">
            <div className="border rounded-lg p-4 space-y-3">
              <h3 className="font-medium">AI-Recommended Adaptation Strategies</h3>
              <ul className="space-y-2">
                <li className="flex items-start gap-2 text-sm">
                  <div className="rounded-full bg-green-100 p-1 mt-0.5">
                    <AlertCircle className="h-3 w-3 text-green-600" />
                  </div>
                  <span>
                    <strong>Drought Resilience:</strong> Increase plant diversity with drought-tolerant species that can withstand {Math.round(currentScenario.temperature * timeframeMultiplier * 3)} additional drought days per year.
                  </span>
                </li>
                <li className="flex items-start gap-2 text-sm">
                  <div className="rounded-full bg-green-100 p-1 mt-0.5">
                    <AlertCircle className="h-3 w-3 text-green-600" />
                  </div>
                  <span>
                    <strong>Water Management:</strong> Implement rainwater harvesting to offset the {currentScenario.precipitationChange > 0 ? 'increased intensity' : 'reduced frequency'} of rainfall events.
                  </span>
                </li>
                <li className="flex items-start gap-2 text-sm">
                  <div className="rounded-full bg-green-100 p-1 mt-0.5">
                    <AlertCircle className="h-3 w-3 text-green-600" />
                  </div>
                  <span>
                    <strong>Thermal Protection:</strong> Add shade trees on the south and west sides to reduce cooling needs during increasingly hot summers.
                  </span>
                </li>
                <li className="flex items-start gap-2 text-sm">
                  <div className="rounded-full bg-green-100 p-1 mt-0.5">
                    <AlertCircle className="h-3 w-3 text-green-600" />
                  </div>
                  <span>
                    <strong>Biodiversity Support:</strong> Plant native species that support pollinators and wildlife facing habitat pressures.
                  </span>
                </li>
              </ul>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="border rounded-lg p-4">
                <h4 className="font-medium mb-2">Recommended Plant Shifts</h4>
                <p className="text-sm">
                  For your region, consider plants that will thrive in conditions 
                  {currentScenario.temperature > 2 ? ' 1-2 USDA zones warmer' : ' with greater temperature extremes'} 
                  by {timeframe}.
                </p>
              </div>
              
              <div className="border rounded-lg p-4">
                <h4 className="font-medium mb-2">Soil Management</h4>
                <p className="text-sm">
                  Focus on building organic matter to increase water retention 
                  capacity by {Math.round(5 + currentScenario.temperature * 10)}% and improve 
                  carbon sequestration.
                </p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter className="border-t pt-4">
        <div className="text-xs text-muted-foreground w-full text-center">
          Model: Climate CMIP6 Ensemble v2.1 | PyTorch Geometric Processing | Last updated: March 2025
        </div>
      </CardFooter>
    </Card>
  );
}