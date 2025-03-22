// lib/firebase/garden-data.ts
import db  from "../firebase";
import { getFirestore, doc, setDoc, getDoc, collection } from "firebase/firestore";

export async function saveGardenPreferences(userId: string, preferences: any) {
  try {
    await setDoc(
      doc(getFirestore(db), "users", userId, "garden", "preferences"),
      preferences
    );
    return true;
  } catch (error) {
    console.error("Error saving garden preferences:", error);
    return false;
  }
}

export async function savePlantRecommendations(
  userId: string,
  recommendations: any
) {
  try {
    await setDoc(
      doc(getFirestore(db), "users", userId, "garden", "recommendations"),
      recommendations
    );
    return true;
  } catch (error) {
    console.error("Error saving plant recommendations:", error);
    return false;
  }
}

export async function getGardenData(userId: string) {
  try {
    const prefsDoc = await getDoc(
      doc(getFirestore(db), "users", userId, "garden", "preferences")
    );
    const recommendationsDoc = await getDoc(
      doc(getFirestore(db), "users", userId, "garden", "recommendations")
    );

    return {
      preferences: prefsDoc.exists() ? prefsDoc.data() : null,
      recommendations: recommendationsDoc.exists()
        ? recommendationsDoc.data()
        : null,
    };
  } catch (error) {
    console.error("Error getting garden data:", error);
    return { preferences: null, recommendations: null };
  }
}
