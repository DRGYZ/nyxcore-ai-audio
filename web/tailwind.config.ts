import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#11100e",
        "background-dark": "#11100e",
        surface: {
          DEFAULT: "#181614",
          low: "#141311",
          mid: "#211e1b",
          high: "#292521",
          raised: "#26221e",
        },
        "surface-dark": "#181614",
        border: {
          DEFAULT: "rgba(245, 240, 230, 0.08)",
          muted: "rgba(245, 240, 230, 0.05)",
          bright: "rgba(245, 240, 230, 0.16)",
        },
        "border-dark": "rgba(245, 240, 230, 0.08)",
        primary: {
          DEFAULT: "#f5f0e6",
          muted: "#bdb5aa",
          subtle: "#9a9084",
        },
        accent: {
          DEFAULT: "#e09f5a",
          muted: "rgba(224, 159, 90, 0.14)",
          glow: "rgba(224, 159, 90, 0.28)",
          dark: "#5c3d1b",
        },
        secondary: "#bdb5aa",
        semantic: {
          success: "#52b788",
          "success-muted": "rgba(82, 183, 136, 0.14)",
          warning: "#e9b44c",
          "warning-muted": "rgba(233, 180, 76, 0.14)",
          danger: "#e06d53",
          "danger-muted": "rgba(224, 109, 83, 0.14)",
          info: "#e09f5a",
        },
      },
      fontFamily: {
        display: ["Space Grotesk", "Segoe UI Variable Display", "sans-serif"],
        editorial: ["Newsreader", "Georgia", "serif"],
        sans: ["Inter", "Segoe UI Variable Text", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"],
        body: ["Inter", "Segoe UI Variable Text", "Segoe UI", "sans-serif"],
      },
      borderRadius: {
        sm: "4px",
        DEFAULT: "6px",
        md: "8px",
        lg: "10px",
        xl: "14px",
        full: "9999px",
      },
      boxShadow: {
        none: "none",
        soft: "0 18px 42px -14px rgba(8, 7, 5, 0.72)",
        sleeve: "0 10px 24px -8px rgba(6, 5, 4, 0.75)",
      },
    },
  },
  plugins: [],
} satisfies Config;
