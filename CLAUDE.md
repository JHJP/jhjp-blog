# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A personal technical blog and digital garden built with [Quartz 4](https://quartz.jzhao.xyz/), deployed to GitHub Pages via GitHub Actions. The site covers ML, causal inference, and AI-assisted engineering workflows. Content is authored in Obsidian and published from the vault's `Blog` folder.

## Commands

- **Preview locally:** `npx quartz build --serve` (serves at http://localhost:8080)
- **Build site:** `npx quartz build`
- **Install deps:** `npm ci`

The site auto-deploys on push to `main` via `.github/workflows/deploy.yml` (builds with Node 22, deploys to GitHub Pages via `actions/deploy-pages`).

## Architecture

- `quartz.config.ts` — Site config (title, base URL, plugins, theme colors, typography)
- `quartz.layout.ts` — Page layout components (sidebar, header, footer, graph, TOC)
- `quartz/styles/custom.scss` — All custom styling (premium typography, graph, portfolio cards)
- `quartz/styles/base.scss` — Quartz core styles (do not edit unless necessary)
- `quartz/styles/variables.scss` — SCSS breakpoints and layout variables
- `quartz/components/` — Quartz component source (TSX + SCSS)
- `content/` — Markdown content (symlink to Obsidian vault's `Blog` folder)
- `public/` — Build output (gitignored)

## Content Structure

```
content/
├── index.md              ← Landing page
├── about.md              ← Author profile
├── portfolio/            ← ML project write-ups (5 posts)
│   └── <slug>/index.md
└── research/             ← Auto-populated by local RAG pipeline
```

## Content Conventions

- Frontmatter fields: `title`, `description`, `author`, `date` (YYYY-MM-DD), `tags` (array), `draft` (boolean)
- Math: standard `$...$` inline and `$$...$$` display (rendered via KaTeX)
- Diagrams: ` ```mermaid ` fenced code blocks (native Quartz support)
- Callouts: Obsidian syntax `> [!note]`, `> [!important]`, `> [!info]`, etc.
- Margin notes: `> **Margin note:**` blockquotes
- Internal links: `[[wikilink]]` syntax (Obsidian-flavored markdown enabled)

## Theming

- **Typography:** Source Serif 4 (headers), Inter (body), JetBrains Mono (code)
- **Colors:** Deep blue palette — light mode secondary `#1a5276`, dark mode secondary `#58a6ff`
- **Reading width:** Article body locked to `70ch` via `custom.scss`
- **CSS variables:** `var(--light)`, `var(--dark)`, `var(--secondary)`, `var(--tertiary)`, `var(--highlight)`, `var(--lightgray)`, `var(--gray)`, `var(--darkgray)`
- **Breakpoints:** `$mobile: 800px`, `$desktop: 1200px`

## Key Details

- The blog contains **redacted** versions of production ML work — never include real company names, client details, or proprietary data
- Base URL: `jhjp.github.io/jhjp-blog`
- GitHub repo: `JHJP/jhjp-blog`
- Obsidian vault integration: symlink vault's `Blog` folder → `content/`; `.obsidian` is in `ignorePatterns`
- The `RemoveDrafts` filter plugin excludes any content with `draft: true` in frontmatter

# Blogging Skill: The Data Sanitizer
Whenever I ask you to write a blog post based on local code:
1. You MUST NOT use raw column names from SQL or Python files (e.g., `active_policy_monthly_premium`).
2. You MUST abstract them into academic terms (e.g., "Monthly Premium Aggregates").
3. You MUST NEVER output raw baseline business metrics (e.g., conversion rates). Use relative multipliers (e.g., "5x lift").
4. If you are unsure if a variable is confidential, stop generating and ask me for permission.
