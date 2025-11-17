"use client";

import { motion } from "framer-motion";
import { fadeInUp, staggerContainer } from "@/lib/animations";

export default function Team() {
  return (
    <section className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-gray-900 dark:to-indigo-900 py-20 px-6">
      <motion.div
        variants={staggerContainer}
        initial="initial"
        whileInView="animate"
        viewport={{ once: true, margin: "-100px" }}
        className="max-w-4xl mx-auto text-center"
      >
        <motion.h2
          variants={fadeInUp}
          className="text-4xl md:text-6xl font-bold mb-8 text-gray-900 dark:text-white"
        >
          Solo Founder with<br />
          Redis/Vector Expertise
        </motion.h2>

        <motion.div
          variants={fadeInUp}
          className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-8 rounded-2xl border border-gray-200 dark:border-gray-700 mb-8"
        >
          <div className="text-lg text-gray-700 dark:text-gray-300 space-y-4 mb-6">
            <p>
              Built RedVector solo—Rust systems engineer with 5+ years in distributed DBs and ANN algorithms.
            </p>
            <p className="font-semibold">
              Proven execution: Matched Qdrant in 2 weeks. Vision: Default vector layer for every Redis deployment and real-time AI workload.
            </p>
          </div>
        </motion.div>

        <motion.div
          variants={fadeInUp}
          className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-8 rounded-2xl border border-gray-200 dark:border-gray-700"
        >
          <h3 className="text-2xl font-semibold mb-6 text-gray-900 dark:text-white">Hiring Ahead</h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <div className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">Vector Search Engineer</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">ANN algorithms, optimization</div>
            </div>
            <div>
              <div className="font-semibold text-purple-600 dark:text-purple-400 mb-2">Systems Lead</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Distributed systems, scaling</div>
            </div>
            <div>
              <div className="font-semibold text-pink-600 dark:text-pink-400 mb-2">Cloud Infra</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Managed service, DevOps</div>
            </div>
          </div>
        </motion.div>

        <motion.p
          variants={fadeInUp}
          className="mt-12 text-xl text-gray-700 dark:text-gray-300 italic"
        >
          &quot;Out-execute the funded giants on the Redis wave.&quot;
        </motion.p>
      </motion.div>
    </section>
  );
}

