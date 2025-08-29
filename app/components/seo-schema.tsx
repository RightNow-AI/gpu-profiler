export function SEOSchema() {
  const schemaData = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebApplication",
        "@id": "https://profiler.rightnowai.co/#webapp",
        "name": "GPU Profiler",
        "alternateName": "RightNow AI GPU Profiler",
        "description": "Free open-source web-based GPU profiler for CUDA engineers. Upload .nvprof/.nsys files and get instant insights with interactive timeline, flame graphs, heatmaps, and AI-powered bottleneck analysis.",
        "url": "https://profiler.rightnowai.co",
        "image": "https://profiler.rightnowai.co/demo.png",
        "applicationCategory": "DeveloperApplication",
        "operatingSystem": "Web Browser",
        "permissions": "No permissions required",
        "offers": {
          "@type": "Offer",
          "price": "0",
          "priceCurrency": "USD",
          "availability": "https://schema.org/InStock"
        },
        "creator": {
          "@type": "Organization",
          "name": "RightNow AI",
          "url": "https://www.rightnowai.co/"
        },
        "keywords": "GPU profiler, CUDA profiling, NVIDIA profiler, GPU performance, nvprof viewer, nsys viewer",
        "browserRequirements": "Requires modern web browser with JavaScript enabled"
      },
      {
        "@type": "Organization",
        "@id": "https://www.rightnowai.co/#organization",
        "name": "RightNow AI",
        "url": "https://www.rightnowai.co/",
        "logo": "https://www.rightnowai.co/logo.png",
        "description": "AI-powered development tools for CUDA engineers and GPU developers",
        "foundingDate": "2024",
        "sameAs": [
          "https://twitter.com/rightnowai_co",
          "https://github.com/RightNow-AI"
        ]
      },
      {
        "@type": "WebPage",
        "@id": "https://profiler.rightnowai.co/#webpage",
        "url": "https://profiler.rightnowai.co",
        "name": "GPU Profiler - Interactive CUDA Performance Visualization",
        "description": "Free open-source web-based GPU profiler for CUDA engineers. Upload .nvprof/.nsys files and get instant insights with interactive visualizations.",
        "isPartOf": {
          "@id": "https://profiler.rightnowai.co/#website"
        },
        "about": {
          "@id": "https://profiler.rightnowai.co/#webapp"
        },
        "datePublished": "2024-01-01",
        "dateModified": new Date().toISOString().split('T')[0],
        "inLanguage": "en-US",
        "potentialAction": {
          "@type": "UseAction",
          "name": "Use GPU Profiler",
          "target": "https://profiler.rightnowai.co"
        }
      }
    ]
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaData) }}
    />
  );
}