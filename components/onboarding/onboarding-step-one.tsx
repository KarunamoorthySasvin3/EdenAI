"use client"

import { useEffect } from "react"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Input } from "@/components/ui/input"
import { MapPin } from "lucide-react"


interface OnboardingStepOneProps {
  formData: any
  updateFormData: (data: any) => void
}

export function OnboardingStepOne({ formData, updateFormData }: OnboardingStepOneProps) {
  // Sync with localStorage whenever formData changes
  useEffect(() => {
    localStorage.setItem("onboardingData", JSON.stringify(formData))
  }, [formData])

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Garden Environment</h3>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Garden Type</Label>
            <RadioGroup
              defaultValue={formData.gardenType}
              onValueChange={(value) => updateFormData({ gardenType: value })}
              className="grid grid-cols-2 gap-4"
            >
              <div>
                <RadioGroupItem value="indoor" id="indoor" className="peer sr-only" />
                <Label
                  htmlFor="indoor"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>Indoor</span>
                </Label>
              </div>
              <div>
                <RadioGroupItem value="outdoor" id="outdoor" className="peer sr-only" />
                <Label
                  htmlFor="outdoor"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>Outdoor</span>
                </Label>
              </div>
              <div>
                <RadioGroupItem value="balcony" id="balcony" className="peer sr-only" />
                <Label
                  htmlFor="balcony"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>Balcony</span>
                </Label>
              </div>
              <div>
                <RadioGroupItem value="community" id="community" className="peer sr-only" />
                <Label
                  htmlFor="community"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>Community</span>
                </Label>
              </div>
            </RadioGroup>
          </div>
          <div className="space-y-2">
            <Label htmlFor="location">Location</Label>
            <div className="relative">
              <MapPin className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                id="location"
                placeholder="Enter your city or zip code"
                className="pl-8"
                value={formData.location}
                onChange={(e) => updateFormData({ location: e.target.value })}
              />
            </div>
            <p className="text-sm text-muted-foreground">
              We'll use this to determine your climate zone and local growing conditions.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

