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
