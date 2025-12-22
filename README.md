# Energy Management Puzzle

An interactive puzzle game built with Next.js and TypeScript where you allocate limited power across various systems.

## Game Overview

Manage a space station's power distribution by allocating limited energy across different systems. Each level presents unique challenges with:

- **Limited Power Budget**: Carefully distribute available power across all systems
- **System Requirements**: Each system has minimum and maximum power thresholds
- **Priority Levels**:
  - **Critical** (Red): Must be activated to complete the level
  - **Important** (Yellow): Recommended but optional
  - **Optional** (Blue): Nice to have but not required
- **Dependencies**: Some systems require other systems to be active first

## Features

- 3 Progressive Levels with increasing difficulty
- Interactive power sliders with min/max quick actions
- Real-time power usage tracking
- Visual feedback for system status
- Dependency validation
- Responsive design with custom animations
- Lightning bolt energy-themed favicon

## How to Play

1. **Understand the Level**: Read the level description and total power available
2. **Identify Critical Systems**: Focus on red (critical) systems first
3. **Allocate Power**: Use sliders or quick buttons (Off/Min/Max) to distribute power
4. **Check Dependencies**: Some systems won't activate unless their dependencies are met
5. **Monitor Usage**: Keep power usage within the available budget
6. **Submit Solution**: Click "Submit Solution" when ready
7. **Progress**: Complete levels to unlock harder challenges

## Tech Stack

- **Framework**: Next.js 15.5 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom animations
- **Runtime**: Turbopack for faster development

## Getting Started

### Prerequisites

- Node.js 18+ installed
- npm or yarn package manager

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

The application will be available at `http://localhost:3000` (or the next available port).

## Project Structure

```
energy_management/
├── app/
│   ├── globals.css          # Global styles and animations
│   ├── layout.tsx            # Root layout
│   ├── page.tsx              # Main game component
│   └── icon.svg              # Favicon
├── lib/
│   └── levels.ts             # Level definitions and success conditions
├── types/
│   └── game.ts               # TypeScript interfaces
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.ts
```

## Level Descriptions

### Level 1: Station Startup
Power up essential systems to bring the station online. Learn the basics of power distribution.

### Level 2: Power Crisis
Emergency situation with reduced power. Make tough choices about which systems to activate.

### Level 3: Maximum Efficiency
Complex power management with multiple dependencies. Optimize your distribution strategy.

## Development

```bash
# Run with Turbopack (faster HMR)
npm run dev

# Lint code
npm run lint

# Type check
npx tsc --noEmit
```

## License

MIT
