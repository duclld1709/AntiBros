import { useEffect, useState } from "react";

/** State mirrored to localStorage (UI preferences and recent questions only). */
export function useLocalStorage<T>(key: string, initial: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? initial : (JSON.parse(raw) as T);
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // storage unavailable (private mode): keep in-memory state
    }
  }, [key, value]);
  return [value, setValue];
}
