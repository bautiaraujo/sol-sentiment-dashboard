import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SOL/USD · Sentiment Dashboard",
  description: "Predicción de precios de Solana con análisis de sentimiento de Reddit — Tesina LCC Datos",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body className="min-h-screen bg-bg">{children}</body>
    </html>
  );
}
