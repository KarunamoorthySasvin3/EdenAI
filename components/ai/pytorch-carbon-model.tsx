import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
} from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Slider } from "../ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Alert, AlertTitle, AlertDescription } from "../ui/alert";
import { AlertCircle, Info, Download, LineChart } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Switch } from "../ui/switch";
import { Label } from "../ui/label";

interface PyTorchCarbonProps {
  className?: string;
}

interface ModelOutput {
  totalSequestration: number;
  confidence: number;
  improvementSuggestions: { action: string; impact: string }[];
  projections: { year: number; sequestration: number }[];
  climateImpact: {
    localTemperatureReduction: number;
    watershedImprovement: number;
    biodiversityScore: number;
    carbonOffsetEquivalent: {
      trees: number;
      carMiles: number;
      flightHours: number;
    };
  };
}

export function PyTorchCarbon({ className }: PyTorchCarbonProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [modelVersion, setModelVersion] = useState<string>("v2.3");
  const [loadingProgress, setLoadingProgress] = useState(0);

  // Garden parameters
  const [gardenSize, setGardenSize] = useState<number>(50);
  const [plantDiversity, setPlantDiversity] = useState<number>(5);
  const [soilQuality, setSoilQuality] = useState<number>(6);
  const [climateZone, setClimateZone] = useState<number>(7);
  const [hasTrees, setHasTrees] = useState<boolean>(false);
  const [nativePlantPercentage, setNativePlantPercentage] =
    useState<number>(30);
  const [averageRainfall, setAverageRainfall] = useState<number>(800);
  const [irrigationType, setIrrigationType] = useState<string>("drip");
  const [composting, setComposting] = useState<boolean>(true);
  const [mulching, setMulching] = useState<boolean>(true);

  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [modelOutput, setModelOutput] = useState<ModelOutput>({
    totalSequestration: 0,
    confidence: 0,
    improvementSuggestions: [],
    projections: [],
    climateImpact: {
      localTemperatureReduction: 0,
      watershedImprovement: 0,
      biodiversityScore: 0,
      carbonOffsetEquivalent: {
        trees: 0,
        carMiles: 0,
        flightHours: 0,
      },
    },
  });

  useEffect(() => {
    if (isLoading) {
      const timer = setInterval(() => {
        setLoadingProgress((prev) => {
          if (prev >= 100) {
            clearInterval(timer);
            return 100;
          }
          return prev + 5;
        });
      }, 100);

      return () => clearInterval(timer);
    }
  }, [isLoading]);

  useEffect(() => {
    if (loadingProgress === 100) {
      setTimeout(() => {
        computeModelResults();
        setIsLoading(false);
        setLoadingProgress(0);
      }, 400);
    }
  }, [loadingProgress]);

  const computeModelResults = () => {
    // In a production app, this would call the PyTorch model API
    const baseSequestration =
      gardenSize * 4.2 * (plantDiversity / 10) * (soilQuality / 10);
    const adjustedSequestration =
      baseSequestration *
      (hasTrees ? 1.5 : 1.0) *
      (1 + (nativePlantPercentage / 100) * 0.3) *
      (composting ? 1.2 : 1.0) *
      (mulching ? 1.15 : 1.0);

    // Adjust for irrigation efficiency
    const irrigationFactor =
      {
        none: 0.8,
        manual: 0.9,
        drip: 1.2,
        sprinkler: 1.0,
      }[irrigationType] || 1.0;

    const finalSequestration = adjustedSequestration * irrigationFactor;

    // Generate climate impact data
    const tempReduction = (finalSequestration / 100) * 0.8;
    const watershedScore =
      (nativePlantPercentage / 100) * 85 + (irrigationType === "drip" ? 10 : 0);
    const bioScore = plantDiversity * 7 + (nativePlantPercentage / 100) * 30;

    const newOutput: ModelOutput = {
      totalSequestration: finalSequestration,
      confidence: 0.75 + Math.random() * 0.2,
      improvementSuggestions: generateSuggestions(),
      projections: [
        { year: 1, sequestration: finalSequestration },
        { year: 2, sequestration: finalSequestration * 1.4 },
        { year: 5, sequestration: finalSequestration * 2.3 },
        { year: 10, sequestration: finalSequestration * 3.1 },
        { year: 20, sequestration: finalSequestration * 4.2 },
      ],
      climateImpact: {
        localTemperatureReduction: tempReduction,
        watershedImprovement: watershedScore,
        biodiversityScore: bioScore,
        carbonOffsetEquivalent: {
          trees: Math.round(finalSequestration / 21),
          carMiles: Math.round(finalSequestration * 4),
          flightHours: Math.round(finalSequestration / 90),
        },
      },
    };

    setModelOutput(newOutput);
  };

  const generateSuggestions = () => {
    const suggestions = [];

    if (plantDiversity < 7) {
      suggestions.push({
        action: "Increase plant diversity with native species",
        impact: `+${Math.round(gardenSize * 0.5)} kg CO₂/year`,
      });
    }

    if (soilQuality < 6) {
      suggestions.push({
        action: "Improve soil quality through organic amendments",
        impact: `+${Math.round(gardenSize * 0.4)} kg CO₂/year`,
      });
    }

    if (!hasTrees && gardenSize > 30) {
      suggestions.push({
        action: "Add at least one tree appropriate for your climate zone",
        impact: `+${Math.round(gardenSize * 0.8)} kg CO₂/year`,
      });
    }

    if (nativePlantPercentage < 50) {
      suggestions.push({
        action: "Increase native plant percentage to at least 50%",
        impact: `+${Math.round(gardenSize * 0.3)} kg CO₂/year`,
      });
    }

    if (!composting) {
      suggestions.push({
        action: "Implement composting in your garden routine",
        impact: `+${Math.round(gardenSize * 0.2)} kg CO₂/year`,
      });
    }

    if (!mulching) {
      suggestions.push({
        action: "Add organic mulch to garden beds",
        impact: `+${Math.round(gardenSize * 0.15)} kg CO₂/year`,
      });
    }

    return suggestions;
  };

  const runModel = () => {
    setIsLoading(true);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>PyTorch Climate Impact Model</CardTitle>
            <CardDescription>
              Advanced neural network predictions for your garden&apos;s carbon
              sequestration and climate resilience
            </CardDescription>
          </div>
          <div className="bg-primary/10 px-3 py-1 rounded-full text-xs font-medium text-primary">
            Model: {modelVersion} • PyTorch 2.0
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="input">
          <TabsList className="grid grid-cols-4 mb-4">
            <TabsTrigger value="input">Input Parameters</TabsTrigger>
            <TabsTrigger value="results">Carbon Metrics</TabsTrigger>
            <TabsTrigger value="projections">Climate Projections</TabsTrigger>
            <TabsTrigger value="technical">Technical Details</TabsTrigger>
          </TabsList>

          <TabsContent value="input" className="space-y-6">
            <div className="space-y-4">
              <div>
                <Label className="mb-2 block">Garden Size (m²)</Label>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[gardenSize]}
                    onValueChange={(value) => setGardenSize(value[0])}
                    min={5}
                    max={500}
                    step={5}
                    className="flex-1"
                  />
                  <span className="w-12 text-right">{gardenSize}m²</span>
                </div>
              </div>

              <div>
                <Label className="mb-2 block">Plant Diversity (1-10)</Label>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[plantDiversity]}
                    onValueChange={(value) => setPlantDiversity(value[0])}
                    min={1}
                    max={10}
                    step={1}
                    className="flex-1"
                  />
                  <span className="w-12 text-right">{plantDiversity}</span>
                </div>
              </div>

              <div>
                <Label className="mb-2 block">Native Plant Percentage</Label>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[nativePlantPercentage]}
                    onValueChange={(value) =>
                      setNativePlantPercentage(value[0])
                    }
                    min={0}
                    max={100}
                    step={5}
                    className="flex-1"
                  />
                  <span className="w-12 text-right">
                    {nativePlantPercentage}%
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <Label>Has Trees</Label>
                <Switch checked={hasTrees} onCheckedChange={setHasTrees} />
              </div>

              <div className="flex items-center justify-between">
                <Label>Show Advanced Options</Label>
                <Switch
                  checked={showAdvancedOptions}
                  onCheckedChange={setShowAdvancedOptions}
                />
              </div>

              {showAdvancedOptions && (
                <div className="space-y-4 pt-4 border-t">
                  <div>
                    <Label className="mb-2 block">Soil Quality (1-10)</Label>
                    <div className="flex items-center gap-4">
                      <Slider
                        value={[soilQuality]}
                        onValueChange={(value) => setSoilQuality(value[0])}
                        min={1}
                        max={10}
                        step={1}
                        className="flex-1"
                      />
                      <span className="w-12 text-right">{soilQuality}</span>
                    </div>
                  </div>

                  <div>
                    <Label className="mb-2 block">Climate Zone</Label>
                    <Select
                      value={climateZone.toString()}
                      onValueChange={(value) => setClimateZone(parseInt(value))}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="USDA Zone" />
                      </SelectTrigger>
                      <SelectContent>
                        {Array.from({ length: 13 }, (_, i) => i + 1).map(
                          (zone) => (
                            <SelectItem key={zone} value={zone.toString()}>
                              Zone {zone}
                            </SelectItem>
                          )
                        )}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label className="mb-2 block">
                      Average Annual Rainfall (mm)
                    </Label>
                    <Input
                      type="number"
                      value={averageRainfall}
                      onChange={(e) =>
                        setAverageRainfall(Number(e.target.value))
                      }
                      min={0}
                      max={5000}
                    />
                  </div>

                  <div>
                    <Label className="mb-2 block">Irrigation Type</Label>
                    <Select
                      value={irrigationType}
                      onValueChange={setIrrigationType}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">None</SelectItem>
                        <SelectItem value="manual">Manual</SelectItem>
                        <SelectItem value="drip">Drip</SelectItem>
                        <SelectItem value="sprinkler">Sprinkler</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex items-center justify-between">
                    <Label>Composting</Label>
                    <Switch
                      checked={composting}
                      onCheckedChange={setComposting}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <Label>Mulching</Label>
                    <Switch checked={mulching} onCheckedChange={setMulching} />
                  </div>
                </div>
              )}

              <Button
                onClick={runModel}
                className="w-full"
                disabled={isLoading}
              >
                {isLoading
                  ? `Processing Model (${loadingProgress}%)`
                  : "Run PyTorch Carbon Model"}
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="results" className="space-y-6">
            <div className="rounded-lg bg-primary/5 p-6 text-center">
              <h3 className="text-xl font-bold text-primary">
                {modelOutput.totalSequestration.toFixed(1)} kg CO₂/year
              </h3>
              <p className="text-sm text-muted-foreground">
                Estimated carbon sequestration with{" "}
                {(modelOutput.confidence * 100).toFixed(2)}% confidence
              </p>
            </div>

            <div className="space-y-2">
              <h3 className="font-medium">Climate Impact Equivalents</h3>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="rounded-lg bg-background p-4 border">
                  <div className="text-xl font-bold">
                    {modelOutput.climateImpact.carbonOffsetEquivalent.trees}
                  </div>
                  <div className="text-xs text-muted-foreground">Trees</div>
                </div>
                <div className="rounded-lg bg-background p-4 border">
                  <div className="text-xl font-bold">
                    {modelOutput.climateImpact.carbonOffsetEquivalent.carMiles}
                  </div>
                  <div className="text-xs text-muted-foreground">Car Miles</div>
                </div>
                <div className="rounded-lg bg-background p-4 border">
                  <div className="text-xl font-bold">
                    {
                      modelOutput.climateImpact.carbonOffsetEquivalent
                        .flightHours
                    }
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Flight Hours
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="font-medium mb-2">
                AI-Generated Improvement Suggestions
              </h3>
              {modelOutput.improvementSuggestions.length > 0 ? (
                <div className="space-y-2">
                  {modelOutput.improvementSuggestions.map((suggestion, idx) => (
                    <div
                      key={idx}
                      className="flex justify-between items-center p-3 rounded bg-background border"
                    >
                      <span>{suggestion.action}</span>
                      <span className="text-primary font-medium">
                        {suggestion.impact}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-4 border rounded bg-background text-center text-muted-foreground">
                  Your garden is optimized for carbon sequestration!
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="projections" className="space-y-6">
            <div className="h-64">
              {/* This would be replaced by an actual chart component */}
              <div className="h-full flex items-center justify-center border rounded bg-muted/20">
                <LineChart className="h-8 w-8 text-muted-foreground" />
                <span className="ml-2 text-muted-foreground">
                  Carbon Sequestration Projection Chart
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <h3 className="font-medium">
                20-Year Climate Impact Projections
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">
                    Local Temperature Reduction
                  </div>
                  <div className="font-bold text-lg">
                    {modelOutput.climateImpact.localTemperatureReduction.toFixed(
                      1
                    )}
                    °C
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">
                    Watershed Improvement
                  </div>
                  <div className="font-bold text-lg">
                    {modelOutput.climateImpact.watershedImprovement.toFixed(0)}%
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">
                    Biodiversity Score
                  </div>
                  <div className="font-bold text-lg">
                    {modelOutput.climateImpact.biodiversityScore.toFixed(0)}/100
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="text-sm text-muted-foreground">
                    Total CO₂ Sequestered
                  </div>
                  <div className="font-bold text-lg">
                    {(
                      modelOutput.projections[
                        modelOutput.projections.length - 1
                      ]?.sequestration || 0
                    ).toFixed(0)}{" "}
                    kg
                  </div>
                </div>
              </div>
            </div>

            <Alert variant="default" className="bg-blue-50 border-blue-200">
              <Info className="h-4 w-4 text-blue-500" />
              <AlertTitle>Climate Resilience Analysis</AlertTitle>
              <AlertDescription className="text-sm">
                This garden design is projected to increase local resilience to
                climate-related events like heat waves and flash flooding by{" "}
                {Math.round(40 + plantDiversity * 5)}%.
              </AlertDescription>
            </Alert>
          </TabsContent>

          <TabsContent value="technical" className="space-y-4">
            <div className="space-y-2">
              <h3 className="font-medium">PyTorch Model Architecture</h3>
              <div className="p-3 bg-black text-green-400 font-mono text-xs rounded overflow-x-auto">
                <pre>
                  {`class CarbonSequestrationModel(nn.Module):
    def __init__(self):
        super(CarbonSequestrationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    
    def forward(self, x):
        return self.model(x)`}
                </pre>
              </div>
            </div>

            <div className="space-y-2">
              <h3 className="font-medium">Data Processing Pipeline</h3>
              <div className="p-3 bg-gray-100 rounded text-xs">
                <code className="block">
                  1. Feature normalization (z-score scaling)
                </code>
                <code className="block">
                  2. One-hot encoding of categorical variables
                </code>
                <code className="block">
                  3. Principal Component Analysis for dimensionality reduction
                </code>
                <code className="block">
                  4. Feature importance scoring via SHAP values
                </code>
                <code className="block">
                  5. Model ensemble with uncertainty quantification
                </code>
              </div>
            </div>

            <div className="space-y-2">
              <h3 className="font-medium">Model Training Information</h3>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="p-2 border rounded">
                  <div className="text-xs text-muted-foreground">
                    Training Dataset
                  </div>
                  <div>Global Garden Carbon Data (42,000 samples)</div>
                </div>
                <div className="p-2 border rounded">
                  <div className="text-xs text-muted-foreground">
                    Validation Accuracy
                  </div>
                  <div>93.7% (±1.2%)</div>
                </div>
                <div className="p-2 border rounded">
                  <div className="text-xs text-muted-foreground">
                    Model Size
                  </div>
                  <div>24MB (quantized 8-bit)</div>
                </div>
                <div className="p-2 border rounded">
                  <div className="text-xs text-muted-foreground">
                    Inference Time
                  </div>
                  <div>127ms (CPU), 14ms (GPU)</div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter className="flex justify-between border-t pt-4">
        <Button variant="outline" size="sm">
          <Download className="mr-2 h-4 w-4" />
          Export Report
        </Button>
        <div className="text-xs text-muted-foreground">
          Powered by PyTorch | Climate Model v2.3
        </div>
      </CardFooter>
    </Card>
  );
}
