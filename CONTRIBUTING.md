# Contributing to GPU Profiler

Thank you for your interest in contributing to GPU Profiler! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)
- [Feature Requests](#feature-requests)
- [Bug Reports](#bug-reports)

## Getting Started

Before contributing, please:

1. Read this contributing guide
2. Check existing issues and pull requests
3. Join our [Discord community](https://discord.com/invite/sSJqgNnq6X)
4. Familiarize yourself with the [Code of Conduct](CODE_OF_CONDUCT.md)

## Development Setup

### Prerequisites

- Node.js 18+
- npm or yarn
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/RightNow-AI/gpu-profiler.git
cd gpu-profiler

# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:3000
```

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## Making Changes

### Branch Naming

Use descriptive branch names following this pattern:
- `feature/description` - For new features
- `fix/description` - For bug fixes
- `docs/description` - For documentation changes
- `refactor/description` - For code refactoring

### Commit Messages

Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, etc.)
- `refactor` - Code refactoring
- `test` - Adding or updating tests
- `chore` - Maintenance tasks

Examples:
```
feat(visualizations): add new heatmap component
fix(parsers): handle malformed nvprof files
docs(readme): update installation instructions
```

## Code Style

### TypeScript

- Use TypeScript for all new code
- Prefer interfaces over types for object shapes
- Use strict type checking
- Add JSDoc comments for complex functions

### React Components

- Use functional components with hooks
- Follow the naming convention: `PascalCase` for components
- Keep components focused and single-purpose
- Use proper prop types and interfaces

### File Structure

```
app/
├── components/          # React components
│   ├── ui/             # Reusable UI components
│   └── visualizations/ # D3.js visualization components
├── lib/                # Utility functions and parsers
├── store/              # State management
├── types/              # TypeScript type definitions
└── globals.css         # Global styles
```

### Styling

- Use Tailwind CSS for styling
- Follow utility-first approach
- Use CSS custom properties for theme values
- Maintain consistent spacing and typography

## Testing

### Manual Testing

Before submitting changes, test:

1. **File Upload**: Test with various .nvprof, .nsys-rep, and .json files
2. **Visualizations**: Verify timeline, flame graph, and heatmap work correctly
3. **Responsive Design**: Test on different screen sizes
4. **Browser Compatibility**: Test in Chrome, Firefox, Safari, Edge

### Test Files

You can use the demo data in `app/lib/demo-data.ts` for testing visualizations.

## Submitting Changes

### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Make** your changes following the guidelines above
4. **Test** your changes thoroughly
5. **Commit** with descriptive messages
6. **Push** to your fork
7. **Create** a pull request

### Pull Request Template

Your PR should include:

- Clear description of changes
- Screenshots for UI changes
- Test instructions
- Any breaking changes
- Related issues

### Review Process

- All PRs require review from maintainers
- Address feedback promptly
- Keep PRs focused and reasonably sized
- Update documentation as needed

## Issue Guidelines

### Before Creating an Issue

1. Search existing issues
2. Check the documentation
3. Try the latest version
4. Reproduce the issue

### Issue Templates

Use the appropriate issue template:
- **Bug Report** - For bugs and errors
- **Feature Request** - For new features
- **Documentation** - For documentation improvements

## Feature Requests

When requesting features:

1. **Describe** the problem you're solving
2. **Explain** why this feature is needed
3. **Provide** examples of similar features
4. **Consider** implementation complexity
5. **Be specific** about requirements

## Bug Reports

When reporting bugs:

1. **Use** the bug report template
2. **Describe** the expected vs actual behavior
3. **Provide** steps to reproduce
4. **Include** error messages and logs
5. **Specify** your environment (OS, browser, etc.)
6. **Attach** sample files if relevant

### Environment Information

Include:
- Operating System
- Browser and version
- Node.js version
- GPU Profiler version
- Sample profiling file (if applicable)

## Getting Help

- **Discord**: [Join our community](https://discord.com/invite/sSJqgNnq6X)
- **Issues**: [GitHub Issues](https://github.com/RightNow-AI/gpu-profiler/issues)
- **Documentation**: [docs.rightnowai.co](https://docs.rightnowai.co/)

## Recognition

Contributors will be recognized in:
- Repository contributors list
- Release notes
- Project documentation

Thank you for contributing to GPU Profiler! 🚀
