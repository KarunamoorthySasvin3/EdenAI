import React from "react";
import { Metadata } from "next";
import { DashboardClient } from "@/components/dashboard/dashboard-client";

export const metadata: Metadata = {
  title: "Garden Dashboard",
  description: "Your personalized climate-smart garden recommendations",
};

export default function DashboardPage() {
  return <DashboardClient />;
}
