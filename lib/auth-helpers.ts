import { getServerSession } from "next-auth/next";
import { NextRequest, NextResponse } from "next/server";
import { authOptions } from "@/lib/auth";

export async function getAuthSession() {
  return await getServerSession(authOptions);
}

export async function authenticatedRoute(
  req: NextRequest,
  handler: (req: NextRequest, userId: string) => Promise<NextResponse>
) {
  const session = await getServerSession(authOptions);

  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  return handler(req, session.user.id);
}
