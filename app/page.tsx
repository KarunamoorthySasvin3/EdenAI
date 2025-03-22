import { LandingHero } from "@/components/landing-hero"
import { Features } from "@/components/features"
import { EnvironmentalImpact } from "@/components/environmental-impact"
import { Footer } from "@/components/footer"
import { Header } from "@/components/header"

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1">
        <LandingHero />
        <Features />
        <EnvironmentalImpact />
      </main>
      <Footer />
    </div>
  )
}

