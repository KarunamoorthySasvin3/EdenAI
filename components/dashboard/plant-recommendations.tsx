import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Info } from "lucide-react";

interface PlantRecommendation {
  id: string; // Required, non-optional string
  plantName?: string;
  name?: string;
  // add more fields if needed
}

interface PlantRecommendationsProps {
  className?: string;
  recommendedPlants: PlantRecommendation[];
}

export function PlantRecommendations({
  className,
  recommendedPlants,
}: PlantRecommendationsProps) {
  const plants = [
    {
      id: "lavender",
      name: "Lavender",
      latinName: "Lavandula angustifolia",
      description:
        "Drought-tolerant perennial with fragrant purple flowers that attract pollinators.",
      benefits: ["Low water needs", "Attracts bees", "Deer resistant"],
      carbonSequestration: "Medium",
      maintenance: "Low",
      image: "/placeholder.svg?height=200&width=300",
    },
    {
      id: "blueberry",
      name: "Blueberry Bush",
      latinName: "Vaccinium corymbosum",
      description:
        "Productive shrub that provides delicious berries and beautiful fall foliage.",
      benefits: ["Edible fruit", "Wildlife food", "Fall color"],
      carbonSequestration: "High",
      maintenance: "Medium",
      image: "/placeholder.svg?height=200&width=300",
    },
    {
      id: "fern",
      name: "Western Sword Fern",
      latinName: "Polystichum munitum",
      description:
        "Native evergreen fern that thrives in shady, moist conditions.",
      benefits: ["Native species", "Erosion control", "Year-round interest"],
      carbonSequestration: "Medium",
      maintenance: "Very low",
      image: "/placeholder.svg?height=200&width=300",
    },
  ];

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Plant Recommendations</CardTitle>
        <CardDescription>
          AI-powered suggestions for your garden
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="lavender">
          <TabsList className="grid w-full grid-cols-3">
            {plants.map((plant) => (
              <TabsTrigger key={plant.id} value={plant.id}>
                {plant.name}
              </TabsTrigger>
            ))}
          </TabsList>
          {plants.map((plant) => (
            <TabsContent key={plant.id} value={plant.id} className="space-y-4">
              <div className="aspect-video relative rounded-lg overflow-hidden">
                <img
                  src={plant.image || "/placeholder.svg"}
                  alt={plant.name}
                  className="object-cover w-full h-full"
                />
              </div>
              <div>
                <h3 className="text-lg font-semibold">{plant.name}</h3>
                <p className="text-sm text-muted-foreground italic">
                  {plant.latinName}
                </p>
                <p className="mt-2">{plant.description}</p>
                <div className="flex flex-wrap gap-2 mt-4">
                  {plant.benefits.map((benefit) => (
                    <Badge key={benefit} variant="secondary">
                      {benefit}
                    </Badge>
                  ))}
                </div>
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div>
                    <p className="text-sm font-medium">Carbon Sequestration</p>
                    <p className="text-sm">{plant.carbonSequestration}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium">Maintenance</p>
                    <p className="text-sm">{plant.maintenance}</p>
                  </div>
                </div>
                <div className="flex justify-between mt-6">
                  <Button variant="outline" size="sm">
                    <Info className="mr-2 h-4 w-4" />
                    Care Guide
                  </Button>
                  <Button size="sm">Add to Garden</Button>
                </div>
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  );
}
