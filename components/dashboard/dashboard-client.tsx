"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import { useApi } from "@/lib/api-client";
import { CarbonFootprint } from "@/components/dashboard/carbon-footprint";
import { GardenChatbot } from "@/components/dashboard/garden-chatbot";
import { PlantRecommendations } from "@/components/dashboard/plant-recommendations";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";

export function DashboardClient() {
  const { user } = useAuth();
  const router = useRouter();
  const api = useApi();
  const [recommendations, setRecommendations] = useState<
    { id?: string; plantName?: string; name?: string }[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (user) {
      fetchRecommendations();
      syncOnboardingData();
    }
  }, [user]);

  const fetchRecommendations = async () => {
    try {
      setLoading(true);
      const data = await api.request("/api/recommendations");
      setRecommendations(data.recommendations || []);
      setError(null);
    } catch (error) {
      console.error("Failed to fetch recommendations:", error);
      setError(
        error instanceof Error
          ? error.message
          : "Failed to load recommendations"
      );
    } finally {
      setLoading(false);
    }
  };

  const syncOnboardingData = async () => {
    const savedData = localStorage.getItem("onboardingData");
    if (savedData) {
      try {
        await api.request("/api/user/onboarding", {
          method: "POST",
          body: savedData,
        });
        // Only remove from localStorage if successfully saved to server
        localStorage.removeItem("onboardingData");
      } catch (error) {
        console.error("Failed to sync onboarding data:", error);
      }
    }
  };

  const navigateHome = () => {
    router.push("/");
  };

  return (
    <div className="container py-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Your Climate-Smart Garden</h1>
        <Button variant="outline" onClick={navigateHome}>
          Home
        </Button>
      </div>

      {error && (
        <Alert className="mb-6" variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {!loading && recommendations.length === 0 && (
        <div className="mb-6">
          <Alert>
            <AlertTitle>No plant recommendations yet</AlertTitle>
            <AlertDescription>
              Complete the questionnaire to get your personalized plant
              recommendations.
              <Button
                variant="link"
                onClick={() => router.push("/questionnaire")}
                className="pl-2"
              >
                Start Questionnaire
              </Button>
            </AlertDescription>
          </Alert>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
        <PlantRecommendations
          className="h-[600px]"
          recommendedPlants={recommendations
            .filter((p) => p.id || p.plantName)
            .map((p) => ({
              id: String(
                p.id ||
                  p.plantName ||
                  `plant-${Math.random().toString(36).substring(2, 10)}`
              ),
              // other properties
            }))}
        />
        <GardenChatbot />
      </div>

      <CarbonFootprint />
    </div>
  );
}

export default DashboardClient;
