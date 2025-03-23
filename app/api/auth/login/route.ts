import { NextRequest, NextResponse } from "next/server";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "@/lib/firebase";
import jwt from "jsonwebtoken";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { email, password } = body;

    if (!email || !password) {
      return NextResponse.json(
        { error: "Email and password are required" },
        { status: 400 }
      );
    }

    // Authenticate with Firebase
    const userCredential = await signInWithEmailAndPassword(
      auth,
      email,
      password
    );
    const user = userCredential.user;

    // Create JWT token
    const secret = process.env.JWT_SECRET || "your-default-secret-key";
    const token = jwt.sign(
      {
        uid: user.uid,
        email: user.email,
        name: user.displayName || email.split("@")[0],
      },
      secret,
      { expiresIn: "7d" }
    );

    // Return the token and user data
    return NextResponse.json({
      token,
      user: {
        id: user.uid,
        email: user.email,
        name: user.displayName || email.split("@")[0],
      },
    });
  } catch (error) {
    console.error("Login error:", error);
    return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
  }
}
