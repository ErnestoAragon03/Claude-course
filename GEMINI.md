# Gemini Code Assist Rules & Norms

This file defines the coding standards, operational commands, and interaction norms for Gemini Code Assist within this project.

## 📏 Coding Standards

### Python (Backend)
- **Style**: Follow PEP 8 guidelines.
- **Type Hinting**: Strictly use type hints for function arguments and return values.
- **Path Handling**: Use `pathlib` or `os.path` compatible with Windows environments.
- **Dependency Management**: Use `uv` for adding/removing packages.

### JavaScript/Frontend
- **Formatting**: 
  - Indentation: 4 spaces.
  - Quotes: Single quotes.
  - Semicolons: Enabled.
- **Linting**: 
  - Prefer `const` over `let`.
  - No `var`.
  - Use strict equality `===`.
- **CSS**: Use CSS variables (e.g., `--background`, `--text-primary`) to ensure Dark/Light theme compatibility.

## 🤖 Interaction Norms
- **Language**: Respond in the language of the user (Spanish/English).
- **Code Quality**: Ensure generated code is complete and runnable. Avoid placeholders unless necessary.
- **Context Awareness**: Always verify answers against the `CONTEXT.md` file to ensure architectural consistency.

## 🚀 Operational Commands

### Backend
```bash
# Install dependencies
uv sync

# Run Server
cd backend && uv run uvicorn app:app --reload --port 8000

# Run Tests
cd backend && uv run pytest
```

### Frontend Quality
```bash
cd frontend
npm run quality      # Check linting and formatting
npm run quality:fix  # Auto-fix issues
```