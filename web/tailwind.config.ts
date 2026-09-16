import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0a0a0c",
        "background-dark": "#0a0a0c",
        surface: {
          DEFAULT: "#111114",
          low: "#0e0e11",
          mid: "#141418",
          high: "#18181c",
        },
        "surface-dark": "#111114",
        border: {
          DEFAULT: "rgba(255, 255, 255, 0.07)",
          muted: "rgba(255, 255, 255, 0.04)",
          bright: "rgba(255, 255, 255, 0.12)",
        },
        "border-dark": "rgba(255, 255, 255, 0.07)",
        primary: {
          DEFAULT: "#f1f0f3",
          muted: "#94a3b8",
          subtle: "#64748b",
        },
        accent: {
          DEFAULT: "#c4b5fd",
          muted: "rgba(196, 181, 253, 0.12)",
          glow: "rgba(196, 181, 253, 0.3)",
          dark: "#332664",
        },
        secondary: "#94a3b8",
        semantic: {
          success: "#34d399",
          "success-muted": "rgba(52, 211, 153, 0.12)",
          warning: "#fbbf24",
          "warning-muted": "rgba(251, 191, 36, 0.12)",
          danger: "#f87171",
          "danger-muted": "rgba(248, 113, 113, 0.12)",
          info: "#c4b5fd",
        },
      },
      fontFamily: {
        display: ["Space Grotesk", "sans-serif"],
        editorial: ["Newsreader", "serif"],
        sans: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
        body: ["Inter", "sans-serif"],
      },
      borderRadius: {
        sm: "2px",
        DEFAULT: "3px",
        md: "4px",
        lg: "6px",
        xl: "8px",
        full: "9999px",
      },
      boxShadow: {
        none: "none",
        soft: "0 16px 40px -12px rgba(0, 0, 0, 0.7)",
      },
    },
  },
  plugins: [],
} satisfies Config;
