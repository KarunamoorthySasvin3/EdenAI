import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface CarbonFootprintProps {
  className?: string;
}

export function CarbonFootprint({ className }: CarbonFootprintProps) {
  const monthlyData = [
    { month: "Jan", sequestered: 5, prevented: 2 },
    { month: "Feb", sequestered: 8, prevented: 3 },
    { month: "Mar", sequestered: 12, prevented: 4 },
    { month: "Apr", sequestered: 20, prevented: 7 },
    { month: "May", sequestered: 32, prevented: 12 },
    { month: "Jun", sequestered: 45, prevented: 15 },
    { month: "Jul", sequestered: 52, prevented: 18 },
    { month: "Aug", sequestered: 48, prevented: 16 },
    { month: "Sep", sequestered: 35, prevented: 13 },
    { month: "Oct", sequestered: 22, prevented: 8 },
    { month: "Nov", sequestered: 12, prevented: 5 },
    { month: "Dec", sequestered: 7, prevented: 3 },
  ];

  const yearlyData = [
    { year: "2023", sequestered: 148, prevented: 52 },
    { year: "2024", sequestered: 298, prevented: 106 },
    { year: "2025", sequestered: 422, prevented: 157, projected: true },
  ];

  const impactMetrics = {
    totalSequestered: 446,
    equivalentTrees: 12,
    carbonFootprintOffset: 0.21, // For an average person
    waterSaved: 1280, // Liters saved
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Carbon Impact</CardTitle>
            <CardDescription>Your garden&apos;s climate contribution</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-4">
          <h4 className="text-lg font-medium">Monthly Data</h4>
          <ul className="list-disc ml-5">
            {monthlyData.map((entry) => (
              <li key={entry.month}>
                {entry.month}: Sequestered {entry.sequestered}, Prevented {entry.prevented}
              </li>
            ))}
          </ul>
        </div>
        <div className="mb-4">
          <h4 className="text-lg font-medium">Yearly Data</h4>
          <ul className="list-disc ml-5">
            {yearlyData.map((entry) => (
              <li key={entry.year}>
                {entry.year}: Sequestered {entry.sequestered}, Prevented {entry.prevented}
                {entry.projected ? " (Projected)" : ""}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="text-lg font-medium">Impact Metrics</h4>
          <p>Total Sequestered: {impactMetrics.totalSequestered}</p>
          <p>Equivalent Trees: {impactMetrics.equivalentTrees}</p>
          <p>Carbon Footprint Offset: {impactMetrics.carbonFootprintOffset}</p>
          <p>Water Saved: {impactMetrics.waterSaved} Liters</p>
        </div>
      </CardContent>
    </Card>
  );
}
