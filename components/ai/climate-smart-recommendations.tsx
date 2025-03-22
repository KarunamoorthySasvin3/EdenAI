import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Leaf, Droplets, Wind, Thermometer, Sprout } from "lucide-react"

interface PlantRecommendation {
  id: number
  name: string
  scientificName: string
  carbonSequestration: number // kg CO2 per year
  waterEfficiency: number // 1-10 scale
  heatTolerance: number // 1-10 scale
  biodiversitySupport: number // 1-10 scale
  maintenanceLevel: number // 1-10 scale
  imageUrl: string
  suitable: boolean
  tags: string[]
}

export function ClimateSmartRecommendations() {
  const [priority, setPriority] = useState<"carbon" | "water" | "resilience">("carbon")
  const [recommendations, setRecommendations] = useState<PlantRecommendation[]>([
    {
      id: 1,
      name: "Native Oak",
      scientificName: "Quercus sp.",
      carbonSequestration: 21.8,
      waterEfficiency: 8,
      heatTolerance: 7,
      biodiversitySupport: 9,
      maintenanceLevel: 3,
      imageUrl: "/plants/oak.jpg",
      suitable: true,
      tags: ["native", "tree", "shade", "wildlife-friendly"]
    },
    {
      id: 2,
      name: "Drought-Resistant Lavender",
      scientificName: "Lavandula sp.",
      carbonSequestration: 4.2,
      waterEfficiency: 9,
      heatTolerance: 9,
      biodiversitySupport: 7,
      maintenanceLevel: 4,
      imageUrl: "/plants/lavender.jpg",
      suitable: true,
      tags: ["drought-resistant", "pollinator", "herb"]
    },
    {
      id: 3,
      name: "Climate-Adapted Vegetable Garden",
      scientificName: "Various",
      carbonSequestration: 6.5,
      waterEfficiency: 6,
      heatTolerance: 7,
      biodiversitySupport: 5,
      maintenanceLevel: 7,
      imageUrl: "/plants/vegetables.jpg",
      suitable: true,
      tags: ["edible", "food-security", "seasonal"]
    }
  ])

  // Simulated AI recommendation refresh
  const refreshRecommendations = () => {
    // In a real implementation, this would call the PyTorch AI backend
    console.log("Requesting new AI recommendations with priority:", priority)
    // Mock new data with slight variations
    setRecommendations(recommendations.map(rec => ({
      ...rec,
      carbonSequestration: rec.carbonSequestration * (0.9 + Math.random() * 0.2),
      waterEfficiency: Math.min(10, rec.waterEfficiency + (Math.random() > 0.5 ? 1 : -1)),
      heatTolerance: Math.min(10, rec.heatTolerance + (Math.random() > 0.5 ? 1 : -1)),
    })))
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Climate-Smart Plant Recommendations</CardTitle>
        <CardDescription>AI-powered plant suggestions optimized for your climate zone and sustainability goals</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="mb-6">
          <h3 className="text-sm font-medium mb-2">Optimization Priority</h3>
          <Tabs value={priority} onValueChange={(value) => setPriority(value as any)}>
            <TabsList className="grid grid-cols-3">
              <TabsTrigger value="carbon">Carbon Capture</TabsTrigger>
              <TabsTrigger value="water">Water Conservation</TabsTrigger>
              <TabsTrigger value="resilience">Climate Resilience</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          {recommendations.map((plant) => (
            <Card key={plant.id} className="overflow-hidden">
              <div className="aspect-square relative bg-muted">
                {/* Replace with actual image in production */}
                <div className="absolute inset-0 bg-gradient-to-br from-green-100 to-green-300 flex items-center justify-center">
                  <Sprout className="h-12 w-12 text-green-700" />
                </div>
              </div>
              <CardContent className="pt-4">
                <h3 className="font-semibold text-lg">{plant.name}</h3>
                <p className="text-sm text-muted-foreground italic">{plant.scientificName}</p>
                
                <div className="flex flex-wrap gap-1 mt-2">
                  {plant.tags.map(tag => (
                    <Badge key={tag} variant="outline" className="text-xs">{tag}</Badge>
                  ))}
                </div>
                
                <div className="mt-4 space-y-2">
                  <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="flex items-center"><Leaf className="h-3 w-3 mr-1" /> Carbon Sequestration</span>
                      <span className="font-medium">{plant.carbonSequestration.toFixed(1)} kg/yr</span>
                    </div>
                    <Slider disabled value={[plant.carbonSequestration * 5]} max={100} />
                  </div>
                  
                  <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="flex items-center"><Droplets className="h-3 w-3 mr-1" /> Water Efficiency</span>
                      <span className="font-medium">{plant.waterEfficiency}/10</span>
                    </div>
                    <Slider disabled value={[plant.waterEfficiency * 10]} max={100} />
                  </div>
                  
                  <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="flex items-center"><Thermometer className="h-3 w-3 mr-1" /> Heat Tolerance</span>
                      <span className="font-medium">{plant.heatTolerance}/10</span>
                    </div>
                    <Slider disabled value={[plant.heatTolerance * 10]} max={100} />
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button variant="default" className="w-full">Get Growing Guide</Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </CardContent>
      <CardFooter>
        <Button onClick={refreshRecommendations} className="w-full">
          Generate New Recommendations
        </Button>
      </CardFooter>
    </Card>
  )
}