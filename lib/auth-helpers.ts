import { getServerSession } from "next-auth/next";
import { NextRequest, NextResponse } from "next/server";
import { authOptions } from "@/lib/auth";

export async function getAuthSession() {
  try {
    return await getServerSession(authOptions);
  } catch (error) {
    console.error("Auth session error:", error);
    return null;
  }
}

export async function authenticatedRoute(
  req: NextRequest,
  handler: (req: NextRequest, userId: string) => Promise<NextResponse>
) {
  const session = await getAuthSession();

  if (!session?.user?.id) {
    console.log("Unauthorized access attempt - missing user ID");
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  return handler(req, session.user.id);
}

export async function authenticatedRouteWithFallback(
  req: NextRequest,
  handler: (req: NextRequest, userId: string | null) => Promise<NextResponse>
) {
  try {
    const session = await getAuthSession();
    return handler(req, session?.user?.id || null);
  } catch (error) {
    console.error("Auth error, continuing as unauthenticated:", error);
    return handler(req, null);
  }
}
