"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Send } from "lucide-react";

interface GardenChatbotProps {
  className?: string;
  recommendedPlants?: Array<{ id: string; name: string }>;
}

export function GardenChatbot({
  className,
  recommendedPlants = [],
}: GardenChatbotProps) {
  const { user } = useAuth();
  const [messages, setMessages] = useState<
    Array<{ role: string; content: string }>
  >([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [activePlant, setActivePlant] = useState<string | null>(null);

  // Fetch chat history when component mounts or user/plant changes
  useEffect(() => {
    if (user) {
      fetchChatHistory();
    } else {
      setMessages([
        {
          role: "bot",
          content: "Please log in to access your personalized plant assistant.",
        },
      ]);
    }
  }, [user, activePlant]);

  const fetchChatHistory = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(
        `/api/chat-history?${activePlant ? `plant=${activePlant}` : ""}`
      );

      if (response.ok) {
        const data = await response.json();
        setMessages(data.history || []);

        // If no history, add welcome message
        if (!data.history || data.history.length === 0) {
          setMessages([
            {
              role: "bot",
              content: activePlant
                ? `Hello! I'm your ${activePlant} specialist. What would you like to know about caring for your ${activePlant}?`
                : "Hello! I'm your personalized plant assistant. Select a plant to get specialized care advice.",
            },
          ]);
        }
      }
    } catch (error) {
      console.error("Failed to fetch chat history:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !user) return;

    // Add user message
    const userMessage = { role: "user", content: input };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");

    try {
      setIsLoading(true);

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input,
          plant: activePlant,
          history: messages.slice(-10), // Send recent context
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setMessages((prev) => [
          ...prev,
          { role: "bot", content: data.response },
        ]);
      } else {
        throw new Error("Failed to get response");
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content:
            "Sorry, I'm having trouble connecting right now. Please try again later.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className={`flex flex-col h-[600px] ${className}`}>
      <CardHeader>
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>Plant Care Assistant</CardTitle>
            <CardDescription>Your personal AI gardening expert</CardDescription>
          </div>

          {recommendedPlants && recommendedPlants.length > 0 && (
            <Select value={activePlant || ""} onValueChange={setActivePlant}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Select plant" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All plants</SelectItem>
                {recommendedPlants.map((plant) => (
                  <SelectItem key={plant.id} value={plant.name}>
                    {plant.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
      </CardHeader>

      <CardContent className="flex-1 overflow-hidden">
        <ScrollArea className="h-full pr-4">
          <div className="space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`${
                    message.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  } p-3 rounded-lg max-w-[80%]`}
                >
                  {message.role === "bot" && (
                    <Avatar className="h-6 w-6 mr-2 inline-block align-middle">
                      <AvatarImage src="/bot-avatar.png" alt="AI" />
                      <AvatarFallback>AI</AvatarFallback>
                    </Avatar>
                  )}
                  {message.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-muted p-3 rounded-lg">
                  <Avatar className="h-6 w-6 mr-2 inline-block align-middle">
                    <AvatarFallback>AI</AvatarFallback>
                  </Avatar>
                  Thinking...
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>

      <CardFooter>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex w-full items-center space-x-2"
        >
          <Input
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading || !user}
          />
          <Button type="submit" size="icon" disabled={isLoading || !user}>
            <Send className="h-4 w-4" />
            <span className="sr-only">Send</span>
          </Button>
        </form>
      </CardFooter>
    </Card>
  );
}
