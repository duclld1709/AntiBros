/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The UI calls the FastAPI routes on its own origin. In dev, Vite proxies them to
// `antibros serve`; in production `antibros serve` serves the built `dist/` itself.
const API = process.env.ANTIBROS_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ask": API,
      "/health": API,
      "/feedback": API,
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
