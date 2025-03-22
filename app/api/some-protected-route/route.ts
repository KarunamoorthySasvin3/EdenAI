import { NextRequest, NextResponse } from "next/server";
import { authenticatedRoute } from "@/lib/auth-helpers";

export async function GET(req: NextRequest) {
  return authenticatedRoute(req, async (req, userId) => {
    // Your protected route logic here
    // userId contains the authenticated user's ID

    return NextResponse.json({
      message: "Authenticated route success",
      userId,
    });
  });
}
