# Frontend Code Quality Tools Setup

## Overview

Added essential code quality tools to the frontend development workflow, including Prettier for automatic code formatting and ESLint for JavaScript linting.

## Files Created

### `frontend/package.json`
- Initialized npm package with development dependencies
- Added Prettier (^3.2.5) for code formatting
- Added ESLint (^8.57.0) for JavaScript linting
- Configured npm scripts for quality checks

### `frontend/.prettierrc`
- Configured Prettier with consistent formatting rules:
  - Semicolons enabled
  - Single quotes for strings
  - 4-space indentation
  - 100 character line width
  - ES5 trailing commas
  - LF line endings

### `frontend/.eslintrc.json`
- Configured ESLint for browser environment
- Extended from eslint:recommended
- Added `marked` as a global (used in script.js)
- Configured rules for code quality:
  - Enforce `===` over `==`
  - Require curly braces for blocks
  - Prefer `const` over `let` when possible
  - Disallow `var`
  - Enforce semicolons and consistent quotes

### `frontend/.prettierignore`
- Excludes `node_modules/` and `package-lock.json` from formatting

### `frontend/.eslintignore`
- Excludes `node_modules/` from linting

## Available Scripts

Run these from the `frontend/` directory after installing dependencies:

```bash
# Install dependencies
npm install

# Format all files (JS, CSS, HTML)
npm run format

# Check formatting without making changes
npm run format:check

# Run ESLint on JavaScript files
npm run lint

# Run ESLint and auto-fix issues
npm run lint:fix

# Run all quality checks (format:check + lint)
npm run quality

# Fix all issues (format + lint:fix)
npm run quality:fix
```

## Usage

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run quality checks:
   ```bash
   npm run quality
   ```

4. Auto-fix formatting and linting issues:
   ```bash
   npm run quality:fix
   ```

---

# Frontend Changes: Dark/Light Theme Toggle

## Summary
Added a toggle button that allows users to switch between dark and light themes with smooth transitions and persistent preference storage.

## Files Modified

### 1. `frontend/index.html`
- Added theme toggle button in the body, positioned at top-right
- Button includes both sun and moon SVG icons for visual feedback
- Includes accessibility attributes (`aria-label`, `title`)
- Updated cache-busting version to v10

### 2. `frontend/style.css`
- **Added Light Theme CSS Variables**: Created a complete light theme variant under `[data-theme="light"]` selector with:
  - Light background colors (`#f8fafc`, `#ffffff`)
  - Dark text for contrast (`#1e293b`, `#64748b`)
  - Adjusted surface and border colors
  - Theme-aware code block backgrounds
  - Theme-aware scrollbar colors
  - Theme-aware source link colors

- **Added Theme Toggle Button Styles**:
  - Fixed position at top-right corner
  - Circular button with hover/focus/active states
  - Icon swap between sun (dark mode) and moon (light mode)
  - Rotation animation on hover

- **Added Global Theme Transitions**:
  - Smooth 0.3s transitions for background-color, color, border-color, and box-shadow
  - Applies to all elements for cohesive theme switching

- **Refactored Hardcoded Colors to CSS Variables**:
  - Replaced hardcoded scrollbar colors with `--scrollbar-thumb` and `--scrollbar-thumb-hover`
  - Replaced hardcoded code background with `--code-bg`
  - Replaced hardcoded source link colors with `--source-link-color` and `--source-link-hover`

### 3. `frontend/script.js`
- Added `themeToggle` to DOM elements
- Added `initializeTheme()` function to load saved preference on page load
- Added `toggleTheme()` function to switch between themes
- Added `setTheme(theme)` function to apply theme and persist to localStorage
- Theme preference is saved to localStorage and restored on subsequent visits

## Features

1. **Toggle Button Design**
   - Positioned in top-right corner
   - Icon-based (sun for dark mode, moon for light mode)
   - Smooth hover, focus, and active animations
   - Keyboard accessible (focusable, has aria-label)

2. **Light Theme**
   - Light background colors
   - Dark text for readability
   - Adjusted primary/secondary colors maintain brand consistency
   - All UI elements properly themed

3. **Smooth Transitions**
   - 0.3s ease transition on all color properties
   - Creates a polished feel when switching themes

4. **Persistence**
   - Theme preference saved to localStorage
   - Restored automatically on page load
   - Default is dark theme

## CSS Variables Reference

| Variable | Dark Theme | Light Theme |
|----------|-----------|-------------|
| `--background` | `#0f172a` | `#f8fafc` |
| `--surface` | `#1e293b` | `#ffffff` |
| `--surface-hover` | `#334155` | `#f1f5f9` |
| `--text-primary` | `#f1f5f9` | `#1e293b` |
| `--text-secondary` | `#94a3b8` | `#64748b` |
| `--border-color` | `#334155` | `#e2e8f0` |
| `--code-bg` | `rgba(0, 0, 0, 0.2)` | `rgba(0, 0, 0, 0.05)` |
