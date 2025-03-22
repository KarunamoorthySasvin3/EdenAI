import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import db from "@/lib/prisma";

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    // Get authenticated user
    const session = await getServerSession(authOptions);

    if (!(session?.user && (session.user as { id: string }).id)) {
      return NextResponse.json(
        { error: "Authentication required" },
        { status: 401 }
      );
    }

    const userId = (session.user as { id: string }).id;
    const data = await request.json();

    // Save user preferences to database
    await db.userPreferences.upsert({
      where: { userId },
      create: {
        userId,
        lightLevel: data.lightLevel,
        waterFrequency: data.waterFrequency,
        spaceAvailable: data.spaceAvailable,
        experienceLevel: data.experienceLevel,
        plantPurpose: data.plantPurpose,
      },
      update: {
        lightLevel: data.lightLevel,
        waterFrequency: data.waterFrequency,
        spaceAvailable: data.spaceAvailable,
        experienceLevel: data.experienceLevel,
        plantPurpose: data.plantPurpose,
      },
    });

    // Get local climate data (could come from an external API in production)
    const climateData = {
      rainfall: 800,
      temperature: 22,
      humidity: 65,
      sunlightHours: 6,
      zone: data.zone || 7,
    };

    // Call Python PyTorch model for plant recommendations
    const modelInputs = JSON.stringify({
      climate: climateData,
      preferences: data,
    });

    const { stdout, stderr } = await execAsync(
      `python -m lib.ml.recommend_plants '${modelInputs}'`
    );

    if (stderr) {
      console.error("Error from Python model:", stderr);
      throw new Error("Model inference failed");
    }

    const recommendations = JSON.parse(stdout);

    // Save recommendations to database
    await Promise.all(
      recommendations.map(async (plant: any) => {
        await db.plantRecommendation.create({
          data: {
            userId,
            plantName: plant.name,
            latinName: plant.latinName || null,
            confidence: plant.confidence,
            description: plant.description || null,
            imageUrl: plant.image || null,
            carbonSequestration: plant.carbonSequestration || "Medium",
          },
        });
      })
    );

    return NextResponse.json({ recommendations });
  } catch (error) {
    console.error("Plant recommendation error:", error);
    return NextResponse.json(
      { error: "Failed to generate recommendations" },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    // Get authenticated user
    const session = await getServerSession(authOptions);

    if (!session?.user || !(session.user as { id: string }).id) {
      return NextResponse.json(
        { error: "Authentication required" },
        { status: 401 }
      );
    }

    const userId = (session.user as { id: string }).id;

    // Fetch user's recommendations
    const recommendations = await db.plantRecommendation.findMany({
      where: { userId },
      orderBy: { confidence: "desc" },
    });

    return NextResponse.json({ recommendations });
  } catch (error) {
    console.error("Error fetching recommendations:", error);
    return NextResponse.json(
      { error: "Failed to fetch recommendations" },
      { status: 500 }
    );
  }
}
