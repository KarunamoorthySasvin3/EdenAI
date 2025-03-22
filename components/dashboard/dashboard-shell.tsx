import type React from "react"
import { Header } from "@/components/header"

interface DashboardShellProps {
  children: React.ReactNode
}

export function DashboardShell({ children }: DashboardShellProps) {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1 container py-8">
        <div className="grid gap-8">{children}</div>
      </main>
    </div>
  )
}

