"use client";

import Hero from "@/components/sections/Hero";
import Problem from "@/components/sections/Problem";
import Solution from "@/components/sections/Solution";
import Benchmarks from "@/components/sections/Benchmarks";
import Market from "@/components/sections/Market";
import Roadmap from "@/components/sections/Roadmap";
import Team from "@/components/sections/Team";
import Ask from "@/components/sections/Ask";
import Navigation from "@/components/Navigation";

export default function Home() {
  return (
    <main className="relative">
      <Navigation />
      <div id="hero">
        <Hero />
      </div>
      <div id="problem">
        <Problem />
      </div>
      <div id="solution">
        <Solution />
      </div>
      <div id="benchmarks">
        <Benchmarks />
      </div>
      <div id="market">
        <Market />
      </div>
      <div id="roadmap">
        <Roadmap />
      </div>
      <div id="team">
        <Team />
      </div>
      <div id="ask">
        <Ask />
      </div>
    </main>
  );
}
