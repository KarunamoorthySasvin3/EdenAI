// Custom type definitions for Vertex AI
declare module "@google-cloud/vertexai" {
  export class VertexAI {
    constructor(options: { project: string; location: string });

    preview: {
      getGenerativeModel(options: { model: string }): GenerativeModelPreview;
    };
  }

  interface GenerativeModelPreview {
    generateContent(request: any): Promise<any>;
  }
}
