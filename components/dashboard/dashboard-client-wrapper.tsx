"use client";

import { AuthProvider } from "@/context/AuthContext";
import DashboardClient from "./dashboard-client"; 

export default function DashboardClientWrapper() {
  return (
    <AuthProvider>
      <DashboardClient />
    </AuthProvider>
  );
}
