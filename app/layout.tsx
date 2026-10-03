import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aquant · 短线候选池",
  description: "中国A股每日量化短线候选池。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
