import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aquant · Candidate Pool",
  description: "Daily quantitative candidate pool for China A-shares.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
