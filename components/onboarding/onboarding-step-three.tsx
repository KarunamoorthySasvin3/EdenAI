"use client"

import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"

interface OnboardingStepThreeProps {
  formData: any
  updateFormData: (data: any) => void
}

export function OnboardingStepThree({ formData, updateFormData }: OnboardingStepThreeProps) {
  const preferences = [
    { id: "edible", label: "Edible Plants (vegetables, fruits, herbs)" },
    { id: "flowers", label: "Ornamental Flowers" },
    { id: "native", label: "Native Plants" },
    { id: "drought", label: "Drought Resistant" },
    { id: "shade", label: "Shade Tolerant" },
    { id: "pollinator", label: "Pollinator Friendly" },
    { id: "year-round", label: "Year-round Interest" },
    { id: "low-allergen", label: "Low Allergen" },
  ]

  const handlePreferenceChange = (id: string, checked: boolean) => {
    const currentPreferences = [...(formData.preferences || [])]
    if (checked) {
      updateFormData({ preferences: [...currentPreferences, id] })
    } else {
      updateFormData({
        preferences: currentPreferences.filter((item) => item !== id),
      })
    }
  }

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Plant Preferences</h3>
        <p className="text-sm text-muted-foreground">Select the types of plants you're interested in growing.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {preferences.map((preference) => (
            <div key={preference.id} className="flex items-center space-x-2">
              <Checkbox
                id={preference.id}
                checked={(formData.preferences || []).includes(preference.id)}
                onCheckedChange={(checked) => handlePreferenceChange(preference.id, checked as boolean)}
              />
              <Label htmlFor={preference.id}>{preference.label}</Label>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

