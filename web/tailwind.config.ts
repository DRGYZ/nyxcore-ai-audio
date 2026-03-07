import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#25e2f4",
        secondary: "#a855f7",
        "background-dark": "#102122",
        "surface-dark": "#1a2626",
        "border-dark": "#2a3a3a",
      },
      fontFamily: {
        display: ["Space Grotesk", "sans-serif"],
        body: ["Space Grotesk", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        soft: "0 24px 48px -28px rgba(37, 226, 244, 0.2)",
      },
    },
  },
  plugins: [],
} satisfies Config;
