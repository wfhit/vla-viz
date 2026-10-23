/** @type {import('next').NextConfig} */

let withBundleAnalyzer = require("@next/bundle-analyzer")({
    enabled: process.env.ANALYZE === "true",
});

const nextConfig = {
  reactStrictMode: false,
  productionBrowserSourceMaps: true,
  // Always static HTML export for Cloudflare Pages / any static host.
  // Dev-only file-write tooling lives outside the App Router (see scripts/dev-schematic-api.mjs).
  output: 'export',
  // Allow dev assets/HMR when the page is opened via a LAN IP or hostname instead of localhost.
  allowedDevOrigins: ['10.*.*.*', '192.168.*.*', '172.16.*.*', '*.local'],
};

module.exports = withBundleAnalyzer(nextConfig);
