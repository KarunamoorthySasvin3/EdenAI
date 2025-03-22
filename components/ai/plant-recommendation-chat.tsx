// components/ai/plant-recommendation-chat.tsx
import React, { useState } from "react";
import { Button } from "../ui/button";
import { Textarea } from "../ui/textarea";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../ui/card";

export function PlantRecommendationChat() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>(
    [
      {
        role: "system",
        content:
          "Hello! I'm your garden planning assistant. Tell me about your garden needs and I'll help recommend climate-appropriate plants.",
      },
    ]
  );
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    const updatedMessages = [...messages, { role: "user", content: input }];
    setMessages(updatedMessages);
    setInput("");

    // TODO: Send message to backend AI service and get response
    // For now, simulate a response
    setTimeout(() => {
      setMessages([
        ...updatedMessages,
        {
          role: "system",
          content:
            "Based on your climate zone and preferences, I recommend native drought-resistant plants like lavender and rosemary.",
        },
      ]);
    }, 1000);
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Plant Recommendation Assistant</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {messages.map((message, i) => (
          <div
            key={i}
            className={`p-3 rounded-lg ${
              message.role === "user"
                ? "bg-primary/10 ml-6"
                : "bg-secondary/10 mr-6"
            }`}
          >
            {message.content}
          </div>
        ))}
      </CardContent>
      <CardFooter className="flex gap-2">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about recommended plants for your garden..."
          className="flex-1"
        />
        <Button onClick={sendMessage}>Send</Button>
      </CardFooter>
    </Card>
  );
}
