"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ModeToggle } from "@/components/mode-toggle";
import { Leaf } from "lucide-react";

export function Header() {
  const handleDangerousClick = () => {
    alert();
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Leaf className="h-6 w-6 text-primary" />
            <Link href="/" className="text-xl font-bold">
              Eden
            </Link>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            <Link
              href="/#features"
              className="text-sm font-medium hover:underline underline-offset-4"
            >
              Features
            </Link>
            <Link
              href="/#impact"
              className="text-sm font-medium hover:underline underline-offset-4"
            >
              Environmental Impact
            </Link>
          </nav>
        </div>
        <div className="flex items-center gap-4">
          <ModeToggle />
          <Link href="/onboarding">
            <Button>Get Started</Button>
          </Link>
          <Link href="/login">
            <Button variant="outline">Login</Button>
          </Link>
        </div>
      </div>
    </header>
  );
}
