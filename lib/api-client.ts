import { SessionService } from "./session-service";
import { useRouter } from "next/navigation";

interface RequestOptions extends RequestInit {
  skipAuth?: boolean;
  requiresAuth?: boolean;
}

export async function apiRequest<T = any>(
  url: string,
  options: RequestOptions = {}
): Promise<T> {
  const { skipAuth = false, requiresAuth = true, ...fetchOptions } = options;

  // Prepare headers
  const headers = new Headers(fetchOptions.headers);
  headers.set("Content-Type", "application/json");

  // Add auth token if needed
  if (!skipAuth && SessionService.isAuthenticated()) {
    const token = SessionService.getToken();
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
  }

  // Make the request
  try {
    const response = await fetch(url, {
      ...fetchOptions,
      headers,
    });

    // Handle unauthorized
    if (response.status === 401 && requiresAuth) {
      console.warn("Authentication required:", url);
      SessionService.clearToken();

      // Instead of directly redirecting, inform the calling code
      throw new Error("UNAUTHORIZED");
    }

    // Parse response
    if (response.ok) {
      // Handle empty responses
      const contentType = response.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        const data = await response.json();
        return data as T;
      }
      return {} as T;
    }

    // Handle other errors
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.error || `Request failed with status ${response.status}`
    );
  } catch (error) {
    if ((error as Error).message === "UNAUTHORIZED") {
      throw error; // Let calling code handle this specific error
    }
    console.error(`API request to ${url} failed:`, error);
    throw error;
  }
}

// Hook to use API with automatic redirect on auth failures
export function useApi() {
  const router = useRouter();

  async function request<T = any>(
    url: string,
    options: RequestOptions = {}
  ): Promise<T> {
    try {
      return await apiRequest<T>(url, options);
    } catch (error) {
      if ((error as Error).message === "UNAUTHORIZED") {
        router.push("/login");
      }
      throw error;
    }
  }

  return { request };
}
