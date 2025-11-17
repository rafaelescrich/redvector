"use client";

import { useState, useLayoutEffect } from "react";
import { Moon, Sun } from "lucide-react";

function getInitialDarkMode(): boolean {
  if (typeof window === "undefined") return false;
  const theme = localStorage.getItem("theme");
  if (theme) return theme === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export default function DarkModeToggle() {
  const [darkMode, setDarkMode] = useState(false);
  const [mounted, setMounted] = useState(false);

  /* eslint-disable react-hooks/set-state-in-effect */
  useLayoutEffect(() => {
    const isDark = getInitialDarkMode();
    // Necessary for SSR compatibility - initializing state from localStorage
    setDarkMode(isDark);
    setMounted(true);
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, []);
  /* eslint-enable react-hooks/set-state-in-effect */

  const toggleDarkMode = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    if (newMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  };

  if (!mounted) {
    return (
      <div className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 w-[40px] h-[40px]"></div>
    );
  }

  return (
    <button
      onClick={toggleDarkMode}
      className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
      aria-label="Toggle dark mode"
    >
      {darkMode ? <Sun size={20} /> : <Moon size={20} />}
    </button>
  );
}

