"use client";

import { useEffect } from "react"; // Add this import
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Checkbox } from "@/components/ui/checkbox";

interface OnboardingStepFourProps {
  formData: any;
  updateFormData: (data: any) => void;
}

export function OnboardingStepFour({
  formData,
  updateFormData,
}: OnboardingStepFourProps) {
  // Add this useEffect hook to sync with localStorage
  useEffect(() => {
    localStorage.setItem("onboardingData", JSON.stringify(formData));
  }, [formData]);

  const goals = [
    { id: "carbon", label: "Carbon Sequestration" },
    { id: "water", label: "Water Conservation" },
    { id: "biodiversity", label: "Support Local Biodiversity" },
    { id: "food", label: "Grow Own Food" },
    { id: "beauty", label: "Create Beautiful Space" },
    { id: "cooling", label: "Reduce Urban Heat Island Effect" },
  ];


  const handleGoalChange = (id: string, checked: boolean) => {
    const currentGoals = [...(formData.goals || [])];
    if (checked) {
      updateFormData({ goals: [...currentGoals, id] });
    } else {
      updateFormData({
        goals: currentGoals.filter((item) => item !== id),
      });
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Sustainability Goals</h3>
        <p className="text-sm text-muted-foreground">
          What are your main goals for your sustainable garden?
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {goals.map((goal) => (
            <div key={goal.id} className="flex items-center space-x-2">
              <Checkbox
                id={goal.id}
                checked={(formData.goals || []).includes(goal.id)}
                onCheckedChange={(checked) =>
                  handleGoalChange(goal.id, checked as boolean)
                }
              />
              <Label htmlFor={goal.id}>{goal.label}</Label>
            </div>
          ))}
        </div>
      </div>
      <div className="space-y-4">
        <h3 className="text-lg font-medium">Maintenance & Budget</h3>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Maintenance Level</Label>
            <RadioGroup
              defaultValue={formData.maintenance}
              onValueChange={(value) => updateFormData({ maintenance: value })}
              className="grid grid-cols-3 gap-4"
            >
              <div>
                <RadioGroupItem
                  value="low"
                  id="low-maintenance"
                  className="peer sr-only"
                />
                <Label
                  htmlFor="low-maintenance"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>Low</span>
                  <span className="text-xs text-muted-foreground">
                    (1-2 hrs/week)
                  </span>
                </Label>
              </div>
              <div>
                <RadioGroupItem
                  value="medium"
                  id="medium-maintenance"
                  className="peer sr-only"
                />
                <Label
                  htmlFor="medium-maintenance"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>Medium</span>
                  <span className="text-xs text-muted-foreground">
                    (3-5 hrs/week)
                  </span>
                </Label>
              </div>
              <div>
                <RadioGroupItem
                  value="high"
                  id="high-maintenance"
                  className="peer sr-only"
                />
                <Label
                  htmlFor="high-maintenance"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>High</span>
                  <span className="text-xs text-muted-foreground">
                    (6+ hrs/week)
                  </span>
                </Label>
              </div>
            </RadioGroup>
          </div>
          <div className="space-y-2">
            <Label>Budget</Label>
            <RadioGroup
              defaultValue={formData.budget}
              onValueChange={(value) => updateFormData({ budget: value })}
              className="grid grid-cols-3 gap-4"
            >
              <div>
                <RadioGroupItem
                  value="low"
                  id="low-budget"
                  className="peer sr-only"
                />
                <Label
                  htmlFor="low-budget"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>Low</span>
                  <span className="text-xs text-muted-foreground">
                    (&lt;$100)
                  </span>
                </Label>
              </div>
              <div>
                <RadioGroupItem
                  value="medium"
                  id="medium-budget"
                  className="peer sr-only"
                />
                <Label
                  htmlFor="medium-budget"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>Medium</span>
                  <span className="text-xs text-muted-foreground">
                    ($100-$300)
                  </span>
                </Label>
              </div>
              <div>
                <RadioGroupItem
                  value="high"
                  id="high-budget"
                  className="peer sr-only"
                />
                <Label
                  htmlFor="high-budget"
                  className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary"
                >
                  <span>High</span>
                  <span className="text-xs text-muted-foreground">($300+)</span>
                </Label>
              </div>
            </RadioGroup>
          </div>
        </div>
      </div>
    </div>
  );
}