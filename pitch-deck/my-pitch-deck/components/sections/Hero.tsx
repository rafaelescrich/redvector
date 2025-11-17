"use client";

import { motion } from "framer-motion";
import { fadeInUp, staggerContainer } from "@/lib/animations";

export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800">
      <div className="absolute inset-0 opacity-20">
        <div className="absolute inset-0" style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
          backgroundSize: '50px 50px'
        }}></div>
      </div>
      <motion.div
        variants={staggerContainer}
        initial="initial"
        animate="animate"
        className="relative z-10 max-w-6xl mx-auto px-6 py-20 text-center"
      >
        <motion.h1
          variants={fadeInUp}
          className="text-6xl md:text-8xl font-bold text-white mb-6 tracking-tight"
        >
          RedVector
        </motion.h1>
        <motion.p
          variants={fadeInUp}
          className="text-2xl md:text-4xl text-white/90 mb-4 font-light"
        >
          Vector Database Built on Redis Architecture
        </motion.p>
        <motion.p
          variants={fadeInUp}
          className="text-lg md:text-xl text-white/80 mb-8 max-w-2xl mx-auto"
        >
          Unlocking Hybrid Vector + Text Workloads at Scale
        </motion.p>
        <motion.div
          variants={fadeInUp}
          className="flex flex-col sm:flex-row gap-4 justify-center items-center"
        >
          <div className="text-white/70 text-sm">
            Seeking $5-8M Seed+ / Early Series A
          </div>
        </motion.div>
      </motion.div>
    </section>
  );
}

