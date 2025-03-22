"use client";

import React from "react";
import { CarbonFootprint } from "@/components/dashboard/carbon-footprint";
import { PlantRecommendationChat } from "@/components/ai/plant-recommendation-chat";

export function DashboardClient() {
  return (
    <div className="container py-8">
      <h1 className="text-3xl font-bold mb-6">Your Climate-Smart Garden</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
        <CarbonFootprint />
        <PlantRecommendationChat />
      </div>
    </div>
  );
}
