import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Header } from "./components/header";
import { NotificationBanner } from "./components/notification-banner";
import { SEOSchema } from "./components/seo-schema";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "GPU Profiler - Interactive CUDA Performance Visualization | RightNow AI",
  description: "Free open-source web-based GPU profiler for CUDA engineers. Upload .nvprof/.nsys files and get instant insights with interactive timeline, flame graphs, heatmaps, and AI-powered bottleneck analysis. No installation required.",
  keywords: [
    "GPU profiler", "CUDA profiling", "NVIDIA profiler", "GPU performance", 
    "CUDA visualization", "GPU timeline", "flame graph", "heatmap", 
    "nvprof viewer", "nsys viewer", "GPU bottleneck", "CUDA optimization",
    "GPU utilization", "kernel analysis", "memory bandwidth", "occupancy analysis",
    "GPU debugging", "CUDA performance tuning", "GPU metrics", "web profiler"
  ],
  authors: [{ name: "RightNow AI", url: "https://www.rightnowai.co/" }],
  creator: "RightNow AI",
  publisher: "RightNow AI",
  applicationName: "GPU Profiler",
  category: "Developer Tools",
  classification: "GPU Performance Analysis Tool",
  openGraph: {
    title: "GPU Profiler - Interactive CUDA Performance Visualization",
    description: "Free open-source web-based GPU profiler for CUDA engineers. Upload .nvprof/.nsys files and get instant insights with interactive visualizations.",
    type: "website",
    siteName: "GPU Profiler by RightNow AI",
    url: "https://profiler.rightnowai.co",
    images: [
      {
        url: "/demo.png",
        width: 1200,
        height: 630,
        alt: "GPU Profiler interface showing interactive timeline, flame graph, and heatmap visualizations",
      },
    ],
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "GPU Profiler - Interactive CUDA Performance Visualization",
    description: "Free open-source web-based GPU profiler for CUDA engineers. Upload .nvprof/.nsys files and get instant insights.",
    creator: "@rightnowai_co",
    images: ["/demo.png"],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: undefined, // Add Google Search Console verification ID when available
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <head>
        <SEOSchema />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased h-full`}
      >
        <div className="min-h-screen flex flex-col">
          <NotificationBanner />
          <Header />
          <main className="flex-1">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
