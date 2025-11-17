# RedVector Pitch Deck

An interactive, modern pitch deck website built with Next.js, React, Tailwind CSS v4, and Framer Motion.

## Features

- 🎨 **Modern Design**: Beautiful gradients, glassmorphism effects, and smooth animations
- 🌙 **Dark Mode**: Toggle between light and dark themes
- 📱 **Responsive**: Fully responsive design that works on all devices
- ♿ **Accessible**: Proper heading hierarchy, ARIA labels, and reduced motion support
- 🎭 **Animations**: Smooth scroll-triggered animations using Framer Motion
- 🧭 **Navigation**: Sticky navigation bar with active section highlighting

## Getting Started

### Prerequisites

- Node.js 18+ and npm (or pnpm/yarn)

### Installation

1. Install dependencies:
```bash
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
npm start
```

## Project Structure

```
my-pitch-deck/
├── app/
│   ├── page.tsx          # Main page with all sections
│   ├── layout.tsx        # Root layout with metadata
│   └── globals.css       # Global styles and Tailwind config
├── components/
│   ├── sections/         # Individual pitch deck sections
│   │   ├── Hero.tsx
│   │   ├── Problem.tsx
│   │   ├── Solution.tsx
│   │   ├── Benchmarks.tsx
│   │   ├── Market.tsx
│   │   ├── Roadmap.tsx
│   │   ├── Team.tsx
│   │   └── Ask.tsx
│   ├── Navigation.tsx    # Sticky navigation bar
│   └── DarkModeToggle.tsx
└── lib/
    └── animations.ts     # Framer Motion animation variants
```

## Sections

1. **Hero**: Title slide with gradient background
2. **Problem**: The problem statement
3. **Solution**: RedVector solution overview
4. **Benchmarks**: Performance comparison with Qdrant
5. **Market**: Market opportunity and TAM
6. **Roadmap**: Product roadmap and milestones
7. **Team**: Team information and hiring plans
8. **Ask**: Funding ask and use of proceeds

## Customization

- **Content**: Edit the section components in `components/sections/`
- **Styling**: Modify `app/globals.css` or use Tailwind classes
- **Animations**: Adjust animation variants in `lib/animations.ts`
- **Colors**: Update Tailwind theme in `globals.css` or use CSS variables

## Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Import your repository in [Vercel](https://vercel.com)
3. Deploy with default settings

### Static Export

```bash
npm run build
```

The `out` directory will contain the static files ready for deployment.

## Technologies

- **Next.js 16**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS v4**: Utility-first CSS framework
- **Framer Motion**: Animation library
- **Lucide React**: Icon library
- **shadcn/ui**: Component library foundation

## License

Private - For internal use only.
