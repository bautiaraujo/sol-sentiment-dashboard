import { promises as fs } from "fs";
import path from "path";
import { Dashboard } from "./components/Dashboard";
import type { DashboardData } from "./components/Dashboard";

export const revalidate = 3600;

async function getData(): Promise<DashboardData | null> {
  try {
    const filePath = path.join(process.cwd(), "public", "data", "dashboard_data.json");
    const raw = await fs.readFile(filePath, "utf-8");
    return JSON.parse(raw) as DashboardData;
  } catch {
    return null;
  }
}

export default async function Home() {
  const data = await getData();
  return <Dashboard data={data} />;
}
