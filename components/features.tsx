import {
  Bot,
  LineChart,
  Leaf,
  Sprout,
  Brain,
  CloudRain,
  BarChart3,
  Cpu,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function Features() {
  return (
    <section id="features" className="w-full py-12 md:py-24 lg:py-32">
      <div className="container px-4 md:px-6">
        <div className="flex flex-col items-center justify-center space-y-4 text-center">
          <div className="inline-block rounded-lg bg-primary/10 px-3 py-1 text-sm text-primary">
            PyTorch-Powered Features
          </div>
          <div className="space-y-2">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">
              Climate-Focused Garden Planning
            </h2>
            <p className="max-w-[900px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
              Use advanced AI models trained on climate data to design a garden
              that fights global warming.
            </p>
          </div>
        </div>
        <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3 mt-12">
          <Card>
            <CardHeader className="flex flex-row items-center gap-4">
              <Brain className="h-8 w-8 text-primary" />
              <div className="grid gap-1">
                <CardTitle>Climate-Adaptive AI</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <CardDescription>
                PyTorch-based neural networks analyze regional climate models to
                recommend plants that will thrive as temperatures rise.
              </CardDescription>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center gap-4">
              <Bot className="h-8 w-8 text-primary" />
              <div className="grid gap-1">
                <CardTitle>Climate Scientist Bot</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Chat with an AI trained on climate science research to
                understand how your garden choices impact global warming.
              </CardDescription>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center gap-4">
              <BarChart3 className="h-8 w-8 text-primary" />
              <div className="grid gap-1">
                <CardTitle>Carbon Sequestration Analytics</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Real-time calculations of your garden's carbon capture potential
                using PyTorch-powered predictive models.
              </CardDescription>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center gap-4">
              <CloudRain className="h-8 w-8 text-primary" />
              <div className="grid gap-1">
                <CardTitle>Extreme Weather Resilience</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <CardDescription>
                AI simulations of how your garden will perform during heat
                waves, droughts, and other climate change-driven events.
              </CardDescription>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center gap-4">
              <Cpu className="h-8 w-8 text-primary" />
              <div className="grid gap-1">
                <CardTitle>Neural Net Visualization</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Explore the AI models analyzing your garden with interactive
                visualizations of our PyTorch neural networks.
              </CardDescription>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center gap-4">
              <Leaf className="h-8 w-8 text-primary" />
              <div className="grid gap-1">
                <CardTitle>Climate Impact Forecasting</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Predictive modeling shows your garden's long-term impact on
                local temperature reduction and climate resilience.
              </CardDescription>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}
