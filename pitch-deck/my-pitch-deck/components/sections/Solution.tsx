"use client";

import { motion } from "framer-motion";
import { fadeInUp, staggerContainer, slideInLeft, slideInRight } from "@/lib/animations";

export default function Solution() {
  return (
    <section className="min-h-screen flex items-center justify-center bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-gray-900 dark:to-indigo-900 py-20 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div
          variants={staggerContainer}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true, margin: "-100px" }}
        >
          <motion.h2
            variants={fadeInUp}
            className="text-4xl md:text-6xl font-bold mb-12 text-center text-gray-900 dark:text-white"
          >
            RedVector: Full Redis Compatibility +<br />
            Native Vector Search
          </motion.h2>

          <div className="grid md:grid-cols-2 gap-12 mb-12">
            <motion.div
              variants={slideInLeft}
              className="bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm p-8 rounded-2xl border border-gray-200 dark:border-gray-700"
            >
              <h3 className="text-2xl font-semibold mb-4 text-red-600 dark:text-red-400">Before</h3>
              <ul className="space-y-3 text-gray-700 dark:text-gray-300">
                <li>• Redis + Sidecar Vector DB</li>
                <li>• Complex data pipelines</li>
                <li>• Schema mismatches</li>
                <li>• Extra latency & costs</li>
              </ul>
            </motion.div>

            <motion.div
              variants={slideInRight}
              className="bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm p-8 rounded-2xl border border-gray-200 dark:border-gray-700"
            >
              <h3 className="text-2xl font-semibold mb-4 text-green-600 dark:text-green-400">After</h3>
              <ul className="space-y-3 text-gray-700 dark:text-gray-300">
                <li>• Single RedVector service</li>
                <li>• Drop-in Redis upgrade</li>
                <li>• Hybrid text + vector queries</li>
                <li>• Zero data copies</li>
              </ul>
            </motion.div>
          </div>

          <motion.div
            variants={fadeInUp}
            className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-8 rounded-2xl border border-gray-200 dark:border-gray-700"
          >
            <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Key Features</h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div>
                <div className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">Drop-in Upgrade</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Same Redis protocol, persistence, and replication
                </div>
              </div>
              <div>
                <div className="font-semibold text-purple-600 dark:text-purple-400 mb-2">Hybrid Workloads</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Text queries + vector embeddings in one service
                </div>
              </div>
              <div>
                <div className="font-semibold text-pink-600 dark:text-pink-400 mb-2">Rust-Powered</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Built for speed and safety; open-source core
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}

