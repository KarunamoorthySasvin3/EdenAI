"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { OnboardingStepOne } from "@/components/onboarding/onboarding-step-one";
import { OnboardingStepTwo } from "@/components/onboarding/onboarding-step-two";
import { OnboardingStepThree } from "@/components/onboarding/onboarding-step-three";
import { OnboardingStepFour } from "@/components/onboarding/onboarding-step-four";
import { Progress } from "@/components/ui/progress";

export default function OnboardingPage() {
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState({
    gardenType: "",
    location: "",
    size: "",
    soilType: "",
    preferences: [],
    goals: [],
    budget: "",
    maintenance: "",
  });
  const router = useRouter();

  const totalSteps = 4;
  const progress = (step / totalSteps) * 100;

  const handleNext = async () => {
    if (step < totalSteps) {
      setStep(step + 1);
    } else {
      try {
        // First, save to localStorage as backup
        localStorage.setItem("onboardingData", JSON.stringify(formData));

        // Then attempt to save to server
        const response = await fetch("/api/user/onboarding", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(formData),
          credentials: "include", // Important for auth cookies
        });

        // If unauthorized or other error, still proceed to dashboard
        // but data is saved in localStorage for later sync
        router.push("/dashboard");
      } catch (error) {
        console.error("Failed to save onboarding data:", error);
        // Save to localStorage and continue
        localStorage.setItem("onboardingData", JSON.stringify(formData));
        router.push("/dashboard");
      }
    }
  };

  const handleBack = () => {
    if (step > 1) {
      setStep(step - 1);
    } else {
      router.push("/");
    }
  };

  const updateFormData = (data: Partial<typeof formData>) => {
    setFormData({ ...formData, ...data });
  };

  return (
    <div className="container flex items-center justify-center min-h-screen py-8">
      <Card className="w-full max-w-3xl">
        <CardHeader>
          <CardTitle>Create Your Sustainable Garden</CardTitle>
          <CardDescription>
            Tell us about your garden environment and preferences so we can
            provide personalized recommendations.
          </CardDescription>
          <Progress value={progress} className="h-2 mt-2" />
        </CardHeader>
        <CardContent>
          {step === 1 && (
            <OnboardingStepOne
              formData={formData}
              updateFormData={updateFormData}
            />
          )}
          {step === 2 && (
            <OnboardingStepTwo
              formData={formData}
              updateFormData={updateFormData}
            />
          )}
          {step === 3 && (
            <OnboardingStepThree
              formData={formData}
              updateFormData={updateFormData}
            />
          )}
          {step === 4 && (
            <OnboardingStepFour
              formData={formData}
              updateFormData={updateFormData}
            />
          )}
        </CardContent>
        <CardFooter className="flex justify-between">
          <Button variant="outline" onClick={handleBack}>
            {step > 1 ? "Back" : "Home"}
          </Button>
          <Button onClick={handleNext}>
            {step === totalSteps ? "Complete Setup" : "Next"}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
