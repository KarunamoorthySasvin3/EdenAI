// app/api/plant-recommendations/route.ts
import { NextRequest, NextResponse } from "next/server";

interface Climate {
  rainfall: number;
  temperature?: number;
  zone?: string;
  sunlight?: string;
}

interface Preferences {
  goals: string[];
  experience?: string;
  space?: number;
  maintenance?: string;
}

interface RecommendationRequest {
  climate: Climate;
  preferences: Preferences;
}

export async function POST(request: NextRequest) {
  try {
    const data: RecommendationRequest = await request.json();
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

export async function GET(request: NextRequest) {
  try {
    // Parse query parameters from the request URL
    const { searchParams } = new URL(request.url);

    const climate: Climate = {
      rainfall: Number(searchParams.get("rainfall")) || 0,
      temperature: searchParams.get("temperature")
        ? Number(searchParams.get("temperature"))
        : undefined,
      zone: searchParams.get("zone") || undefined,
      sunlight: searchParams.get("sunlight") || undefined,
    };

    const preferences: Preferences = {
      goals: searchParams.get("goals")
        ? searchParams.get("goals")!.split(",")
        : [],
      experience: searchParams.get("experience") || undefined,
      space: searchParams.get("space")
        ? Number(searchParams.get("space"))
        : undefined,
      maintenance: searchParams.get("maintenance") || undefined,
    };

    const recommendations = generatePlantRecommendations(climate, preferences);
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
  // Simple rule-based recommendation logic remains unchanged
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
