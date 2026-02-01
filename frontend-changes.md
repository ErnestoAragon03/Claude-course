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
