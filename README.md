# SOL/USD Sentiment Dashboard

Dashboard en Next.js — predicciones de precio Solana + sentimiento Reddit.

## Stack
- **Frontend**: Next.js 14 + Recharts + Tailwind — deploy en Vercel  
- **Pipeline**: XGBoost + cardiffnlp RoBERTa sentiment
- **Datos**: CoinGecko + Reddit r/Solana
- **Cron**: GitHub Actions diario 06:00 UTC
