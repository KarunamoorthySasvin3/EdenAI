import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import db from "@/lib/prisma";
import { generatePlantImage } from "@/lib/gemini";

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions);
    const userId = session?.user?.id || "demo-user";
    const data = await request.json();

    // Prepare climate data with defaults as needed
    const climateData = {
      rainfall: data.rainfall || 800,
      temperature: data.temperature || 22,
      humidity: data.humidity || 65,
      sunlightHours: data.sunlightHours || 6,
      zone: data.zone || 7,
    };

    // Call your Python inference as before
    const modelInputs = JSON.stringify({
      climate: climateData,
      preferences: data,
    });
    const { stdout, stderr } = await execAsync(
      `python -m lib.ml.recommend_plants '${modelInputs}'`
    );

    if (stderr.trim()) {
      console.error("Error from Python model:", stderr);
      throw new Error("Model inference failed");
    }

    // Ensure that stdout is valid JSON
    const recommendations = JSON.parse(stdout);
    // (Enhance recommendations with generated images if needed)
    return NextResponse.json({ recommendations });
  } catch (error: any) {
    console.error("Plant recommendation error:", error);
    return NextResponse.json(
      { error: "Failed to generate recommendations" },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  const session = await getServerSession(authOptions);

  // Check if user is authenticated
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    // This is a placeholder - replace with actual database queries
    // for real recommendations based on user preferences
    const mockRecommendations = [
      { id: "1", plantName: "Lavender", name: "Lavender" },
      { id: "2", plantName: "Basil", name: "Basil" },
      { id: "3", plantName: "Rosemary", name: "Rosemary" },
    ];

    return NextResponse.json({ recommendations: mockRecommendations });
  } catch (error) {
    console.error("Error fetching recommendations:", error);
    return NextResponse.json(
      { error: "Failed to fetch recommendations" },
      { status: 500 }
    );
  }
}
