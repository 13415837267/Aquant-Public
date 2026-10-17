import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
 title: "个人量化交易系统｜全因子研究",
 description: "T日全因子与T+1/T+2开盘收益三分类研究报告",
};
export default function RootLayout({children}:{children:React.ReactNode}){
 return <html lang="zh-CN"><body>{children}</body></html>;
}