import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { MapPin, Cloud, Ruler, Droplets } from "lucide-react"

interface GardenProfileProps {
  className?: string
}

export function GardenProfile({ className }: GardenProfileProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Garden Profile</CardTitle>
        <CardDescription>Your garden environment and preferences</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2">
          <MapPin className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm">Seattle, WA (Zone 8b)</span>
        </div>
        <div className="flex items-center gap-2">
          <Cloud className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm">Temperate, Rainy</span>
        </div>
        <div className="flex items-center gap-2">
          <Ruler className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm">Medium Garden (400 sq ft)</span>
        </div>
        <div className="flex items-center gap-2">
          <Droplets className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm">Clay Soil, Good Drainage</span>
        </div>
        <div className="flex flex-wrap gap-2 mt-4">
          <Badge variant="outline">Edible Plants</Badge>
          <Badge variant="outline">Low Maintenance</Badge>
          <Badge variant="outline">Drought Resistant</Badge>
          <Badge variant="outline">Native Species</Badge>
        </div>
      </CardContent>
    </Card>
  )
}

