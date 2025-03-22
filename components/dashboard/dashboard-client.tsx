"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import { CarbonFootprint } from "@/components/dashboard/carbon-footprint";
import { GardenChatbot } from "@/components/dashboard/garden-chatbot";
import { PlantRecommendations } from "@/components/dashboard/plant-recommendations";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";

export function DashboardClient() {
  const { user } = useAuth();
  const router = useRouter();
  const [recommendations, setRecommendations] = useState<
    { id?: string; plantName?: string; name?: string }[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Redirect to questionnaire if no recommendations
    if (user) {
      fetchRecommendations();
    }
  }, [user]);

  const fetchRecommendations = async () => {
    try {
      setLoading(true);
      const res = await fetch("/api/recommendations");
      if (res.ok) {
        const data = await res.json();
        setRecommendations(data.recommendations || []);
        setError(null);
      } else {
        throw new Error("Failed to fetch recommendations");
      }
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      setError("Failed to load your plant recommendations");
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return (
      <div className="container py-8">
        <Alert>
          <AlertTitle>Not signed in</AlertTitle>
          <AlertDescription>
            Please sign in to view your personalized dashboard.
            <Button
              variant="link"
              onClick={() => router.push("/login")}
              className="pl-2"
            >
              Sign in
            </Button>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const startNewQuestionnaire = () => {
    router.push("/questionnaire");
  };

  return (
    <div className="container py-8">
      <h1 className="text-3xl font-bold mb-6">Your Climate-Smart Garden</h1>

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
                onClick={startNewQuestionnaire}
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
        <GardenChatbot
          className="h-[600px]"
          recommendedPlants={recommendations
            .filter((p) => p.id || p.plantName)
            .map((p) => ({
              id:
                p.id ||
                p.plantName ||
                `plant-${Math.random().toString(36).substring(2, 10)}`,
              name: p.plantName || p.name || "Unknown Plant",
            }))}
        />
      </div>

      <div className="mt-10">
        <CarbonFootprint />
      </div>
    </div>
  );
}
