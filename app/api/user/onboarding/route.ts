import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/lib/auth";

export async function POST(request: NextRequest) {
  const session = await getServerSession(authOptions);

  // Check if user is authenticated
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    // Get onboarding data from request
    const onboardingData = await request.json();

    // Here you would typically save this to your database
    // Example: await db.user.update({ where: { id: session.user.id }, data: { onboardingComplete: true, preferences: onboardingData } })

    // For now, we'll just return success
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to process onboarding data:", error);
    return NextResponse.json(
      { error: "Failed to process request" },
      { status: 500 }
    );
  }
}
