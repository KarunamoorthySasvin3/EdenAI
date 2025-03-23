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
import React, { useState, useEffect } from "react";

interface PlantRecommendation {
  id: string; // Required, non-optional string
  plantName?: string;
  name?: string;
  latinName?: string;
  description?: string;
  benefits?: string[];
  carbonSequestration?: string;
  maintenance?: string;
  image?: string;
}

interface PlantRecommendationsProps {
  className?: string;
  recommendedPlants: PlantRecommendation[];
}

interface Plant {
  name: string;
  latinName: string;
  description: string;
  benefits: string[];
  carbonSequestration: string;
  maintenance: string;
  image?: string;
}

interface PlantRecommendationProps {
  plant: Plant;
}

export default function PlantRecommendation({
  plant,
}: PlantRecommendationProps) {
  const [imageUrl, setImageUrl] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setIsLoading(true);
    fetch(`/api/plantImage?plantName=${encodeURIComponent(plant.name)}`)
      .then((res) => res.json())
      .then((data) => {
        console.log("API response:", data); // Already logging this

        if (data.image) {
          setImageUrl(data.image);
          console.log("Setting image URL:", data.image);

          // Debug image loading
          const testImg = new Image();
          testImg.onload = () => console.log("Image loaded successfully");
          testImg.onerror = (e) => console.error("Image failed to load:", e);
          testImg.src = data.image;
        } else {
          console.error("No image data in response");
        }
        setIsLoading(false);
      })
      .catch((error) => {
        console.error("Error generating plant image:", error);
        setIsLoading(false);
      });
  }, [plant.name]);

  return (
    <div className="plant-recommendation">
      <h2>{plant.name}</h2>
      {isLoading ? (
        <div className="w-[200px] h-[200px] flex items-center justify-center bg-gray-100 rounded">
          Loading image...
        </div>
      ) : imageUrl ? (
        <div>
          <img
            src={imageUrl}
            alt={`${plant.name} plant`}
            className="w-[200px] h-[200px] object-cover rounded"
            onError={(e) => {
              console.error("Image failed to load, using fallback");
              e.currentTarget.src = `https://placehold.co/600x400/green/white?text=${encodeURIComponent(
                plant.name
              )}`;
            }}
          />
        </div>
      ) : (
        <div className="w-[200px] h-[200px] flex items-center justify-center bg-gray-100 rounded">
          Failed to load image.
        </div>
      )}
    </div>
  );
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
                {plant.name} {/* Keep only the name in the tabs */}
              </TabsTrigger>
            ))}
          </TabsList>
          {plants.map((plant) => (
            <TabsContent key={plant.id} value={plant.id} className="space-y-4">
              <div className="aspect-video relative rounded-lg overflow-hidden">
                <PlantRecommendation plant={plant} />
              </div>
              <div className="flex flex-col space-y-4">
                <h2 className="text-xl font-bold">
                  {plant.name} <span className="italic text-sm">({plant.latinName})</span> {/* Highlight Latin name */}
                </h2>
                <p>{plant.description}</p>
                <div className="flex flex-wrap gap-2">
                  {plant.benefits.map((benefit) => (
                    <Badge key={benefit} variant="secondary">
                      {benefit}
                    </Badge>
                  ))}
                </div>
                <div className="flex justify-between items-center mt-4">
                  <div className="space-y-1">{}</div>
                </div>
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  );
}
