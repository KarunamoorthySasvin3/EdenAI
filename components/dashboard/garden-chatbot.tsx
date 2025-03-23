"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
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
  const router = useRouter();
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
      const response = await fetch("/api/chat-history");
      const data = await response.json();
      setMessages(data.history || []);
    } catch (error) {
      console.error("Failed to fetch chat history:", error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !user) return;

    // Add user message
    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);

    const tempInput = input;
    setInput(""); // Clear input field
    setIsLoading(true);

    try {
      // Use the new NicksAI endpoint
      const response = await fetch("/api/nicksai/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: tempInput,
          plant: activePlant,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        // Add bot response
        setMessages((prev) => [
          ...prev,
          { role: "bot", content: data.response },
        ]);
      } else {
        throw new Error("Failed to get response");
      }
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content:
            "Sorry, I'm having trouble connecting to the server. Please try again later.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Card className={`flex flex-col h-[500px] ${className}`}>
      <CardHeader>
        <CardTitle>Plant AI Assistant</CardTitle>
        <CardDescription>
          Powered by NicksAI - Ask questions about plant care
        </CardDescription>
        <Select
          value={activePlant || ""}
          onValueChange={(value) => setActivePlant(value || null)}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select a plant" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All plants</SelectItem>
            {recommendedPlants.map((plant) => (
              <SelectItem key={plant.id} value={plant.name}>
                {plant.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent className="flex-grow overflow-hidden p-4">
        <ScrollArea className="h-full pr-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role === "user" ? "justify-end" : "justify-start"
              } mb-4`}
            >
              {message.role === "bot" && (
                <Avatar className="h-8 w-8 mr-2">
                  <AvatarImage src="/bot-avatar.png" alt="Bot" />
                  <AvatarFallback>AI</AvatarFallback>
                </Avatar>
              )}
              <div
                className={`max-w-[80%] p-3 rounded-lg ${
                  message.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted"
                }`}
              >
                {message.content}
              </div>
              {message.role === "user" && (
                <Avatar className="h-8 w-8 ml-2">
                  <AvatarImage src="/user-avatar.png" alt="User" />
                  <AvatarFallback>
                    {user?.name?.charAt(0) || "U"}
                  </AvatarFallback>
                </Avatar>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start mb-4">
              <Avatar className="h-8 w-8 mr-2">
                <AvatarFallback>AI</AvatarFallback>
              </Avatar>
              <div className="max-w-[80%] p-3 rounded-lg bg-muted">
                <div className="flex space-x-1">
                  <div className="animate-bounce h-2 w-2 bg-gray-500 rounded-full"></div>
                  <div className="animate-bounce-delay-100 h-2 w-2 bg-gray-500 rounded-full"></div>
                  <div className="animate-bounce-delay-200 h-2 w-2 bg-gray-500 rounded-full"></div>
                </div>
              </div>
            </div>
          )}
        </ScrollArea>
      </CardContent>
      <CardFooter className="border-t p-4">
        <div className="flex w-full items-center space-x-2">
          <Input
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading || !user}
            className="flex-grow"
          />
          <Button
            size="icon"
            onClick={handleSend}
            disabled={isLoading || !input.trim() || !user}
          >
            <Send className="h-4 w-4" />
            <span className="sr-only">Send</span>
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
}
