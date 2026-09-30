import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath: "/Aquant-Public",
  images: { unoptimized: true },
  poweredByHeader: false,
};

export default nextConfig;
