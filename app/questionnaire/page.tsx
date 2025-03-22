"use client";

import { useState } from "react";
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

  // Preference states
  const [lightLevel, setLightLevel] = useState<string>("medium");
  const [waterFrequency, setWaterFrequency] = useState<string>("weekly");
  const [spaceAvailable, setSpaceAvailable] = useState<number>(50);
  const [experienceLevel, setExperienceLevel] = useState<string>("beginner");
  const [plantPurpose, setPlantPurpose] = useState<string>("decoration");

  // Submit form handler
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      const response = await fetch("/api/user/preferences", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lightLevel,
          waterFrequency,
          spaceAvailable,
          experienceLevel,
          plantPurpose,
        }),
      });

      if (response.ok) {
        router.push("/recommendations");
      } else {
        throw new Error("Failed to save preferences");
      }
    } catch (error) {
      console.error("Error saving preferences:", error);
    }
  };

  return (
    <div className="container max-w-3xl py-10">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">
            Tell Us About Your Plant Needs
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Light Level */}
            <div className="space-y-2">
              <Label>How much light does your space receive?</Label>
              <RadioGroup
                value={lightLevel}
                onValueChange={setLightLevel}
                className="flex flex-col space-y-1"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="low" id="light-low" />
                  <Label htmlFor="light-low">
                    Low Light (Shade, North-facing windows)
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="medium" id="light-medium" />
                  <Label htmlFor="light-medium">
                    Medium Light (Partial Shade, East/West-facing)
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="high" id="light-high" />
                  <Label htmlFor="light-high">
                    Bright Light (Full Sun, South-facing)
                  </Label>
                </div>
              </RadioGroup>
            </div>

            {/* Watering Frequency */}
            <div className="space-y-2">
              <Label>How often would you like to water plants?</Label>
              <Select value={waterFrequency} onValueChange={setWaterFrequency}>
                <SelectTrigger>
                  <SelectValue placeholder="Select watering frequency" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="daily">Daily</SelectItem>
                  <SelectItem value="weekly">Weekly</SelectItem>
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
