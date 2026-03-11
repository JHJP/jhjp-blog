import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

const config: QuartzConfig = {
  configuration: {
    pageTitle: "Jay Park",
    pageTitleSuffix: " — AI & ML Engineering",
    enableSPA: true,
    enablePopovers: true,
    analytics: null,
    locale: "en-US",
    baseUrl: "jhjp.github.io/jhjp-blog",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "created",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Source Serif 4",
        body: "Inter",
        code: "JetBrains Mono",
      },
      colors: {
        lightMode: {
          light: "#fafafa",
          lightgray: "#e8e8e8",
          gray: "#999999",
          darkgray: "#3d3d3d",
          dark: "#1a1a1a",
          secondary: "#1a5276",
          tertiary: "#2e86c1",
          highlight: "rgba(26, 82, 118, 0.08)",
          textHighlight: "#f0e68c88",
        },
        darkMode: {
          light: "#0f1419",
          lightgray: "#1e2630",
          gray: "#5c6b77",
          darkgray: "#c9d1d9",
          dark: "#e6edf3",
          secondary: "#58a6ff",
          tertiary: "#79c0ff",
          highlight: "rgba(88, 166, 255, 0.10)",
          textHighlight: "#c9a90288",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents({ maxDepth: 3 }),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
    ],
  },
}

export default config
