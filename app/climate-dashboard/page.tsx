import { Metadata } from "next";
import { CarbonFootprint } from "@/components/dashboard/carbon-footprint";
import { ClimateSmartRecommendations } from "@/components/ai/climate-smart-recommendations";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export const metadata: Metadata = {
  title: "Climate Impact Dashboard",
  description:
    "Track and optimize your garden's contribution to fighting climate change",
};

export default function ClimateDashboardPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <main className="flex-1">
        <div className="container relative py-6 px-4 md:px-6">
          <div className="mx-auto max-w-5xl space-y-8">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">
                Climate Impact Dashboard
              </h1>
              <p className="text-muted-foreground mt-2">
                Track your garden's contribution to fighting climate change and
                discover AI-powered optimizations.
              </p>
            </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <CarbonFootprint className="h-full" />
  
                <Card>
                  <CardHeader>
                    <CardTitle>Climate Impacts Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Tabs defaultValue="carbon">
                      <TabsList className="grid grid-cols-3">
                        <TabsTrigger value="carbon">Carbon</TabsTrigger>
                        <TabsTrigger value="water">Water</TabsTrigger>
                        <TabsTrigger value="biodiversity">
                          Biodiversity
                        </TabsTrigger>
                      </TabsList>
                      <TabsContent value="carbon" className="pt-4">
                        <div className="space-y-4">
                          <p>
                            Your garden currently sequesters an estimated{" "}
                            <strong>446 kg</strong> of CO₂ annually, equivalent to
                            offsetting <strong>21%</strong> of an average person's
                            carbon footprint.
                          </p>
  
                          <div className="rounded-lg bg-muted p-4">
                            <h4 className="font-medium mb-2">
                              How this helps fight climate change:
                            </h4>
                            <ul className="list-disc list-inside space-y-1 text-sm">
                              <li>Reduces atmospheric carbon dioxide levels</li>
                              <li>
                                Creates a carbon sink in soil through root systems
                              </li>
                              <li>
                                Reduces need for carbon-intensive store-bought
                                produce
                              </li>
                              <li>Helps mitigate urban heat island effect</li>
                            </ul>
                          </div>
                        </div>
                      </TabsContent>
                      <TabsContent value="water" className="pt-4">
                        <div className="space-y-4">
                          <p>
                            Your garden's water-efficient design saves
                            approximately <strong>1,280 liters</strong> of water
                            annually compared to conventional gardens of similar
                            size.
                          </p>
  
                          <div className="rounded-lg bg-muted p-4">
                            <h4 className="font-medium mb-2">
                              How this helps fight climate change:
                            </h4>
                            <ul className="list-disc list-inside space-y-1 text-sm">
                              <li>
                                Conserves increasingly scarce water resources
                              </li>
                              <li>
                                Reduces energy used for water treatment and
                                pumping
                              </li>
                              <li>
                                Creates resilience against drought conditions
                              </li>
                              <li>
                                Demonstrates sustainable water management
                                practices
                              </li>
                            </ul>
                          </div>
                        </div>
                      </TabsContent>
                      <TabsContent value="biodiversity" className="pt-4">
                        <div className="space-y-4">
                          <p>
                            Your garden supports{" "}
                            <strong>18 native pollinator species</strong> and
                            creates habitat connectivity in an urban environment.
                          </p>
  
                          <div className="rounded-lg bg-muted p-4">
                            <h4 className="font-medium mb-2">
                              How this helps fight climate change:
                            </h4>
                            <ul className="list-disc list-inside space-y-1 text-sm">
                              <li>
                                Supports declining pollinator populations
                                essential for food security
                              </li>
                              <li>
                                Creates resilient ecosystems that better adapt to
                                climate change
                              </li>
                              <li>
                                Maintains ecosystem services valued at $125-$150
                                annually
                              </li>
                              <li>
                                Preserves genetic diversity needed for climate
                                adaptation
                              </li>
                            </ul>
                          </div>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>
              </div>
            </div>
              <ClimateSmartRecommendations />
        </div>
      </main>
    </div>
  );
}
