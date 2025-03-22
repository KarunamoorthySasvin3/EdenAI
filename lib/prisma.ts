import { PrismaClient } from "@prisma/client";

// Prevent multiple instances of Prisma Client in development
declare global {
  var prisma: PrismaClient | undefined;
}

// Use existing prisma instance if available to avoid multiple connections during hot reloading
const db = global.prisma || new PrismaClient();

// Store the prisma instance globally in development
if (process.env.NODE_ENV === "development") {
  global.prisma = db;
}

export default db;
