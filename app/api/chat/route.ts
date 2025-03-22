import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import { getServerSession } from "next-auth";
import { authOptions } from "../../../lib/auth";
import app from "@/lib/firebase";
import {
  getFirestore,
  collection,
  addDoc,
  serverTimestamp,
} from "firebase/firestore";

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    // Get authenticated user
    const session = await getServerSession(authOptions);

    if (!session?.user?.id) {
      return NextResponse.json(
        { error: "Authentication required" },
        { status: 401 }
      );
    }

    const userId = session.user.id;
    const { message, plant, history } = await request.json();

    // Call Python PyTorch chatbot model
    const modelInputs = JSON.stringify({
      userId,
      message,
      plant,
      history: history || [],
    });

    const { stdout, stderr } = await execAsync(
      `python -m api.chat_endpoint '${modelInputs}'`
    );

    if (stderr) {
      console.error("Error from Python chatbot:", stderr);
      throw new Error("Chatbot inference failed");
    }

    const response = JSON.parse(stdout).response;

    // Save chat to Firestore database
    const db = getFirestore(app);
    const chatHistoryRef = collection(db, "chatHistory");
    await addDoc(chatHistoryRef, {
      userId,
      plantContext: plant || null,
      message,
      response,
      createdAt: serverTimestamp(),
    });

    return NextResponse.json({ response });
  } catch (error) {
    console.error("Chat error:", error);
    return NextResponse.json(
      {
        error: "Failed to generate response",
        response: "I'm sorry, I'm having trouble responding right now.",
      },
      { status: 500 }
    );
  }
}
