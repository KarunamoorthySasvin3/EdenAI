/**
 * Wrapper around fetch that includes credentials and handles common auth patterns
 */
export async function fetchWithAuth(url: string, options: RequestInit = {}) {
  // Ensure credentials are included
  const fetchOptions: RequestInit = {
    ...options,
    credentials: "include",
    headers: {
      ...options.headers,
      "Content-Type": options.headers && 'Content-Type' in Object(options.headers) 
        ? (options.headers as Record<string, string>)["Content-Type"] 
        : "application/json",
    },
  };

  try {
    const response = await fetch(url, fetchOptions);

    // Handle 401 Unauthorized errors specially
    if (response.status === 401) {
      // Could redirect to login here or throw a special error
      console.warn("Authentication required for", url);

      // If you want to force redirection to login:
      // window.location.href = '/login';
    }

    return response;
  } catch (error) {
    console.error(`API request failed for ${url}:`, error);
    throw error;
  }
}
