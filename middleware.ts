import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { getToken } from "next-auth/jwt";

export async function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname;

  // Define paths that should be protected
  const isProtectedPath =
    path.includes("/api/") &&
    !path.includes("/api/auth/login") &&
    !path.includes("/api/auth/register");

  if (isProtectedPath) {
    const token = await getToken({
      req: request,
      secret: process.env.NEXTAUTH_SECRET,
    });

    // Not authenticated, redirect to login
    if (!token) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
  }

  return NextResponse.next();
}

// Configure paths that will use this middleware
export const config = {
  matcher: [
    "/api/chat-history/:path*",
    "/api/recommendations/:path*",
    "/api/user/:path*",
    "/api/auth/me",
  ],
};
