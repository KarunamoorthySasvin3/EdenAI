// app/api/plant-recommendations/route.ts
import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();

    // In a real implementation, this would call your PyTorch model
    // For the hackathon demo, you could use a simpler approach:
    const recommendations = generatePlantRecommendations(
      data.climate,
      data.preferences
    );

    return NextResponse.json({ recommendations });
  } catch (error) {
    console.error("Error generating recommendations:", error);
    return NextResponse.json(
      { error: "Failed to generate recommendations" },
      { status: 500 }
    );
  }
}

function generatePlantRecommendations(climate: any, preferences: any) {
  // Simple rule-based system as a fallback/demo
  // This would be replaced with actual ML model inference
  const plants = [];

  if (climate.rainfall < 500) {
    plants.push("Lavender", "Rosemary", "Succulents");
  } else if (climate.rainfall < 1000) {
    plants.push("Tomatoes", "Peppers", "Marigolds");
  } else {
    plants.push("Hostas", "Ferns", "Hydrangeas");
  }

  // Filter based on user preferences
  if (preferences.goals.includes("food")) {
    plants.push("Kale", "Swiss Chard", "Berries");
  }

  return plants;
}
