"use client"

import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface OnboardingStepTwoProps {
  formData: any
  updateFormData: (data: any) => void
}

export function OnboardingStepTwo({ formData, updateFormData }: OnboardingStepTwoProps) {
  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Garden Size & Soil</h3>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Garden Size</Label>
            <RadioGroup
              defaultValue={formData.size}
              onValueChange={(value) => updateFormData({ size: value })}
              className="grid grid-cols-3 gap-4"
            >
              <div>
                <RadioGroupItem value="small" id="small" className="peer sr-only" />
                <Label
                  htmlFor="small"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>Small</span>
                  <span className="text-xs text-muted-foreground">(&lt;100 sq ft)</span>
                </Label>
              </div>
              <div>
                <RadioGroupItem value="medium" id="medium" className="peer sr-only" />
                <Label
                  htmlFor="medium"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>Medium</span>
                  <span className="text-xs text-muted-foreground">(100-500 sq ft)</span>
                </Label>
              </div>
              <div>
                <RadioGroupItem value="large" id="large" className="peer sr-only" />
                <Label
                  htmlFor="large"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>Large</span>
                  <span className="text-xs text-muted-foreground">(&gt;500 sq ft)</span>
                </Label>
              </div>
            </RadioGroup>
          </div>
          <div className="space-y-2">
            <Label htmlFor="soil-type">Soil Type</Label>
            <Select value={formData.soilType} onValueChange={(value) => updateFormData({ soilType: value })}>
              <SelectTrigger id="soil-type">
                <SelectValue placeholder="Select soil type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="clay">Clay</SelectItem>
                <SelectItem value="loam">Loam</SelectItem>
                <SelectItem value="sandy">Sandy</SelectItem>
                <SelectItem value="silt">Silt</SelectItem>
                <SelectItem value="unknown">I don't know</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-sm text-muted-foreground">
              Different plants thrive in different soil types. Don't worry if you're not sure - we can help!
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

