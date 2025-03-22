import { NextRequest, NextResponse } from "next/server";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "@/lib/firebase";

export async function POST(req: NextRequest) {
  try {
    const { email, password } = await req.json();

    const userCredential = await signInWithEmailAndPassword(
      auth,
      email,
      password
    );
    const user = userCredential.user;

    return NextResponse.json({
      user: {
        id: user.uid,
        email: user.email,
        name: user.displayName || email.split("@")[0],
      },
    });
  } catch (error: any) {
    return NextResponse.json(
      { message: error.message || "Login failed" },
      { status: 400 }
    );
  }
}
