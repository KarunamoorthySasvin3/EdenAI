// Type definitions for Vertex AI
declare module "@google-cloud/vertexai" {
  export class VertexAI {
    constructor(options: { project: string; location: string });

    preview: {
      getGenerativeModel(options: { model: string }): GenerativeModel;
    };
  }

  interface GenerativeModel {
    generateContent(request: any): Promise<any>;
    generateImage(request: {
      prompt: string;
      sampleCount?: number;
      sampleImageSize?: string;
    }): Promise<{
      images: Array<{
        bytesBase64: string;
      }>;
    }>;
  }
}
