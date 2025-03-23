import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import { getServerSession } from "next-auth";
import { authOptions } from "../../../../lib/auth";
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
    const { message, plant } = await request.json();

    if (!message) {
      return NextResponse.json(
        { error: "Message is required" },
        { status: 400 }
      );
    }

    // Call Python NicksAI chatbot model
    const modelInputs = JSON.stringify({
      message,
      plant: plant || null,
    });

    const { stdout, stderr } = await execAsync(
      `python -m api.nicksai_chat '${modelInputs}'`
    );

    if (stderr) {
      console.error("Error from Python chatbot:", stderr);
      throw new Error("NicksAI chatbot inference failed");
    }

    const responseData = JSON.parse(stdout);
    const response = responseData.response;

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
