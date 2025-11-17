"use client";

import { useState, useEffect } from "react";
import DarkModeToggle from "./DarkModeToggle";

const sections = [
  { id: "hero", label: "Home" },
  { id: "problem", label: "Problem" },
  { id: "solution", label: "Solution" },
  { id: "benchmarks", label: "Benchmarks" },
  { id: "market", label: "Market" },
  { id: "roadmap", label: "Roadmap" },
  { id: "team", label: "Team" },
  { id: "ask", label: "The Ask" },
];

export default function Navigation() {
  const [activeSection, setActiveSection] = useState("hero");

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + 100;

      for (let i = sections.length - 1; i >= 0; i--) {
        const section = document.getElementById(sections[i].id);
        if (section && section.offsetTop <= scrollPosition) {
          setActiveSection(sections[i].id);
          break;
        }
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-800">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="text-xl font-bold text-indigo-600 dark:text-indigo-400">
            RedVector
          </div>
          <div className="flex items-center gap-4">
            <div className="hidden md:flex gap-2">
              {sections.map((section) => (
                <button
                  key={section.id}
                  onClick={() => scrollToSection(section.id)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    activeSection === section.id
                      ? "bg-indigo-600 text-white"
                      : "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
                  }`}
                  aria-label={`Navigate to ${section.label} section`}
                >
                  {section.label}
                </button>
              ))}
            </div>
            <DarkModeToggle />
          </div>
        </div>
      </div>
    </nav>
  );
}

