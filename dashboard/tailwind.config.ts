import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ["var(--font-raleway)", "sans-serif"],
        mono:    ["var(--font-fira)", "monospace"],
      },
      colors: {
        bg:       "#060D1F",
        card:     "#0C1830",
        border:   "#1E3A5F",
        primary:  "#4F80FF",
        positive: "#10CFAA",
        negative: "#FF4D6A",
        baseline: "#F5A623",
        muted:    "#6B89B0",
        heading:  "#E8F4FF",
        body:     "#9BB5D5",
      },
      boxShadow: {
        glow: "0 0 24px rgba(79,128,255,0.15)",
        card: "0 4px 24px rgba(0,0,0,0.4)",
      },
    },
  },
  plugins: [],
};
export default config;
