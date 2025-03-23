import { generatePlantImage } from "@/lib/gemini";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const plantName = searchParams.get("plantName");

    if (!plantName) {
      return NextResponse.json(
        { error: "Plant name is required" },
        { status: 400 }
      );
    }

    const image = await generatePlantImage(plantName);
    return NextResponse.json({ image });
  } catch (error: any) {
    console.error("Error generating plant image:", error);
    return NextResponse.json(
      { error: error.message || "Failed to generate image" },
      { status: 500 }
    );
  }
}
