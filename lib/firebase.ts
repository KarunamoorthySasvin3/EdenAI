import { initializeApp, getApps } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyDAggDpQtyZGiNx7QXJUGROvzCf7W63pPk",
  authDomain: "genai-genesis-9af8d.firebaseapp.com",
  projectId: "genai-genesis-9af8d",
  storageBucket: "genai-genesis-9af8d.firebasestorage.app",
  messagingSenderId: "388202779096",
  appId: "1:388202779096:web:3b3c635261124036bfe2ad",
  measurementId: "G-TJC3WQCYWX",
};

if (typeof window !== "undefined" && !getApps().length) {
  initializeApp(firebaseConfig);
}

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export default app;
