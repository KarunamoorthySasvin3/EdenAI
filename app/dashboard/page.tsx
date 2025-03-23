"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import DashboardClientWrapper from "../../components/dashboard/dashboard-client-wrapper";
import { useAuth } from "@/context/AuthContext";

export default function DashboardPage() {
  const router = useRouter();
  const { user, loading, authChecked } = useAuth();

  useEffect(() => {
    // Wait until auth check is complete; if no user then redirect
    if (!loading && authChecked && !user) {
      router.replace("/login");
    }
  }, [user, loading, authChecked, router]);

  // Show loading state until auth is verified
  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
        <p className="ml-3">Verifying authentication...</p>
      </div>
    );
  }

  // If no user after auth check, let the useEffect handle redirection
  if (!user) {
    return null;
  }

  // If user exists, show the dashboard
  return <DashboardClientWrapper />;
}
