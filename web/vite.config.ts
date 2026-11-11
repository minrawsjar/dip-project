import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: Vite on :5173 proxies the API to `ekyc serve` on :8000. Prod: `npm run build` -> web/dist, served by FastAPI.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/results": "http://127.0.0.1:8000",
    },
  },
});
