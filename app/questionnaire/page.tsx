"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function QuestionnairePage() {
  const { user } = useAuth();
  const router = useRouter();
  const [sunlight, setSunlight] = useState("full");
  const [maintenanceFrequency, setMaintenanceFrequency] = useState("weekly");
  const [spaceAvailable, setSpaceAvailable] = useState(50);
  const [experienceLevel, setExperienceLevel] = useState("beginner");
  const [plantPurpose, setPlantPurpose] = useState("decoration");

  useEffect(() => {
    if (!user) {
      router.push("/login");
    }
  }, [user, router]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    // Collect all form data
    const formData = {
      sunlight,
      maintenanceFrequency,
      spaceAvailable,
      experienceLevel,
      plantPurpose,
    };

    // Save preferences to user profile (implementation depends on your backend)
    try {
      // Redirect to recommendations page with preferences as query params
      router.push(
        `/recommendations?${new URLSearchParams(formData as any).toString()}`
      );
    } catch (error) {
      console.error("Error saving preferences:", error);
    }
  };

  if (!user) {
    return <p>Loading...</p>;
  }

  return (
    <div className="container max-w-xl py-10">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">Plant Preferences</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Sunlight Availability */}
            <div className="space-y-2">
              <Label>Sunlight availability:</Label>
              <RadioGroup
                value={sunlight}
                onValueChange={setSunlight}
                className="flex flex-col space-y-1"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="full" id="full" />
                  <Label htmlFor="full">
                    Full sun (6+ hours direct sunlight)
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="partial" id="partial" />
                  <Label htmlFor="partial">Partial sun (3-6 hours)</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="shade" id="shade" />
                  <Label htmlFor="shade">
                    Mostly shade (less than 3 hours)
                  </Label>
                </div>
              </RadioGroup>
            </div>

            {/* Maintenance Frequency */}
            <div className="space-y-2">
              <Label>How often can you maintain your plants?</Label>
              <Select
                value={maintenanceFrequency}
                onValueChange={setMaintenanceFrequency}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select frequency" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="daily">Daily</SelectItem>
                  <SelectItem value="weekly">Once a week</SelectItem>
                  <SelectItem value="biweekly">Every two weeks</SelectItem>
                  <SelectItem value="monthly">Monthly or less</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Space Available */}
            <div className="space-y-2">
              <Label>Available space (square feet): {spaceAvailable}</Label>
              <Slider
                min={1}
                max={500}
                step={1}
                value={[spaceAvailable]}
                onValueChange={(vals) => setSpaceAvailable(vals[0])}
              />
            </div>

            {/* Experience Level */}
            <div className="space-y-2">
              <Label>Your gardening experience level:</Label>
              <Select
                value={experienceLevel}
                onValueChange={setExperienceLevel}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select experience level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="beginner">Beginner</SelectItem>
                  <SelectItem value="intermediate">Intermediate</SelectItem>
                  <SelectItem value="advanced">Advanced</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Plant Purpose */}
            <div className="space-y-2">
              <Label>Primary purpose for your plants:</Label>
              <Select value={plantPurpose} onValueChange={setPlantPurpose}>
                <SelectTrigger>
                  <SelectValue placeholder="Select purpose" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="decoration">Home Decoration</SelectItem>
                  <SelectItem value="food">Growing Food</SelectItem>
                  <SelectItem value="air">Air Purification</SelectItem>
                  <SelectItem value="climate">Climate Impact</SelectItem>
                  <SelectItem value="wildlife">Supporting Wildlife</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button type="submit" className="w-full">
              Get Plant Recommendations
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
