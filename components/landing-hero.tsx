import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Leaf, Sprout, Cloud, Droplets } from "lucide-react";

export function LandingHero() {
  return (
    <section className="w-full py-12 md:py-24 lg:py-32 bg-gradient-to-b from-background to-muted">
      <div className="container px-4 md:px-6">
        <div className="grid gap-6 lg:grid-cols-[1fr_400px] lg:gap-12 xl:grid-cols-[1fr_600px]">
          <div className="flex flex-col justify-center space-y-4">
            <div className="space-y-2">
              <div className="inline-block rounded-lg bg-primary/10 px-3 py-1 text-sm text-primary">
                Powered by PyTorch AI Models
              </div>
              <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none">
                Combat Climate Change With Your Garden
              </h1>
              <p className="max-w-[600px] text-muted-foreground md:text-xl">
                Use advanced AI climate models to design a carbon-negative
                garden that helps mitigate global warming while adapting to
                changing climate conditions.
              </p>
            </div>
            <div className="flex flex-col gap-2 min-[400px]:flex-row">
              <Link href="/onboarding">
                <Button size="lg" className="gap-1">
                  <Sprout className="h-5 w-5" />
                  Start Your Garden
                </Button>
              </Link>
              <Link href="/#features">
                <Button size="lg" variant="outline">
                  Learn More
                </Button>
              </Link>
            </div>
          </div>
          <div className="flex items-center justify-center">
            <div className="grid grid-cols-2 gap-4 p-4 md:p-0">
              <div className="grid gap-4">
                <div className="rounded-lg bg-primary/10 p-8 flex items-center justify-center">
                  <Leaf className="h-12 w-12 text-primary" />
                </div>
                <div className="rounded-lg bg-primary/10 p-8 flex items-center justify-center">
                  <Cloud className="h-12 w-12 text-primary" />
                </div>
              </div>
              <div className="grid gap-4">
                <div className="rounded-lg bg-primary/10 p-8 flex items-center justify-center">
                  <Sprout className="h-12 w-12 text-primary" />
                </div>
                <div className="rounded-lg bg-primary/10 p-8 flex items-center justify-center">
                  <Droplets className="h-12 w-12 text-primary" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
