import { NextRequest, NextResponse } from "next/server";
import { generatePlantImage } from "../../lib/gemini";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const plantName = searchParams.get("plantName");

  if (!plantName) {
    return NextResponse.json(
      { error: "Plant name is required" },
      { status: 400 }
    );
  }

  try {
    const image = await generatePlantImage(plantName);
    return NextResponse.json({ image: image });
  } catch (error) {
    console.error("Error generating image:", error);
    return NextResponse.json(
      { error: "Failed to generate image" },
      { status: 500 }
    );
  }
}
