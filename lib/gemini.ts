import { GoogleGenerativeAI } from "@google/generative-ai";
import { config } from "./config";
import { VertexAI } from "@google-cloud/vertexai";
import * as path from "path";

// Set Google Application Credentials environment variable
process.env.GOOGLE_APPLICATION_CREDENTIALS =
  "C:\\Users\\ritvi\\Downloads\\GenAiGenesis\\proven-electron-454501-g6-0584ce0d4c1d.json";

const genAI = new GoogleGenerativeAI(config.googleApiKey);

// Add this before the generatePlantImage function
const delay = (ms: number) => new Promise((res) => setTimeout(res, ms));

// Simple in-memory cache for plant images
const plantImageCache = new Map<string, string>();

/**
 * Generates an image of a plant using Google's Imagen through Vertex AI
 * @param plantName The name of the plant to generate an image for
 * @returns A promise that resolves to the image data URL
 */
export async function generatePlantImage(plantName: string): Promise<string> {
  // Check if the image is already in the cache
  if (plantImageCache.has(plantName)) {
    console.log(`Returning cached image for ${plantName}`);
    return plantImageCache.get(plantName) as string;
  }

  try {
    // Add this at the beginning of the try block
    await delay(100000); // Delay for 1 second (adjust as needed)

    // First, generate a detailed description with Gemini for better image prompting
    const textModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    const prompt = `Create a detailed description of what a ${plantName} plant looks like, 
    including its key visual characteristics, colors, and habitat.`;

    console.log(`Generating description for ${plantName}`);
    const result = await textModel.generateContent(prompt);
    const description = await result.response.text();
    console.log(`Generated description: ${description.substring(0, 100)}...`);

    console.log(
      "Initializing Vertex AI with project:",
      config.googleCloudProjectId
    );

    // Initialize Vertex AI with your Google Cloud project
    const vertexAI = new VertexAI({
      project: config.googleCloudProjectId, // Make sure this matches your actual project ID
      location: "us-central1",
    });

    // Get the image generation model
    const imageModel = vertexAI.preview.getGenerativeModel({
      model: "imagegeneration@002",
    });

    // This is the correct request format for Imagen in Vertex AI
    const imageRequest = {
      prompt: `A detailed, photorealistic image of a ${plantName} plant. ${description}`,
      sampleCount: 1,
      sampleImageSize: "1024x1024", // Valid values are 1024x1024, 512x512, etc.
    };

    console.log(
      `Sending image generation request to Google Cloud for ${plantName}`
    );

    try {
      const imageResponse = await imageModel.generateContent({
        prompt: `A detailed, photorealistic image of a ${plantName} plant. ${description}`,
      });

      // Log the full response for debugging
      console.log(
        "Full image response:",
        JSON.stringify(imageResponse, null, 2)
      );

      // Extract the base64 image data from the response
      //const imageBytes = imageResponse.results[0].imageBytes;
      const imageBytes =
        imageResponse.candidates[0].content.parts[0].image.bytesBase64;

      if (!imageBytes) {
        throw new Error("No image bytes returned from Imagen API");
      }

      // Return as data URL
      const imageUrl = `data:image/png;base64,${imageBytes}`;

      // Store the image in the cache
      plantImageCache.set(plantName, imageUrl);

      return imageUrl;
    } catch (imageError) {
      console.error("Specific error in generateImage call:", imageError);
      // Return the cached image if available, even on error
      if (plantImageCache.has(plantName)) {
        console.log(`Returning cached image for ${plantName} after error`);
        return plantImageCache.get(plantName) as string;
      }
      throw imageError; // Re-throw if no cached image is available
    }
  } catch (error) {
    console.error("Error in image generation:", error);
    console.error("Full error object:", JSON.stringify(error, null, 2)); // Add this line
    // Return a fallback for development purposes
    return `https://placehold.co/600x400/green/white?text=${encodeURIComponent(plantName)}`;
  }
}
