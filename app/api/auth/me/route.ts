import { NextRequest, NextResponse } from "next/server";
import jwt from "jsonwebtoken";
import { auth } from "@/lib/firebase";

export async function GET(req: NextRequest) {
  try {
    // Extract token from Authorization header
    const authHeader = req.headers.get("authorization");

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const token = authHeader.split(" ")[1];

    // Verify the token
    const secret = process.env.JWT_SECRET || "your-default-secret-key";
    const decoded = jwt.verify(token, secret);

    if (!decoded || typeof decoded !== "object") {
      return NextResponse.json({ error: "Invalid token" }, { status: 401 });
    }

    // Return user data
    return NextResponse.json({
      user: {
        id: decoded.uid,
        email: decoded.email,
        name: decoded.name,
      },
    });
  } catch (error) {
    console.error("Error in /api/auth/me:", error);
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
}
