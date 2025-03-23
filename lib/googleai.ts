import { GoogleGenerativeAI } from "@google/generative-ai";
import { config } from "./config";
import { generatePlantImage as generateActualPlantImage } from "./gemini";

// Initialize the Google AI with the API key from config
const genAI = new GoogleGenerativeAI(config.googleApiKey);

/**
 * Generates an image of a plant using Google's AI models
 * @param plantName The name of the plant to generate an image for
 * @returns A promise that resolves to the image data URL or empty string on failure
 */
export async function generatePlantImage(plantName: string): Promise<string> {
  try {
    // Use the actual image generation function from gemini.ts
    return await generateActualPlantImage(plantName);
  } catch (error: any) {
    console.error("Failed to generate plant image:", error);
    return ""; // Or a default image URL
  }
}
