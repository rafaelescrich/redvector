"use client";

import { motion } from "framer-motion";
import { fadeInUp, staggerContainer } from "@/lib/animations";

export default function Problem() {
  return (
    <section className="min-h-screen flex items-center justify-center bg-white dark:bg-gray-900 py-20 px-6">
      <motion.div
        variants={staggerContainer}
        initial="initial"
        whileInView="animate"
        viewport={{ once: true, margin: "-100px" }}
        className="max-w-5xl mx-auto"
      >
        <motion.h2
          variants={fadeInUp}
          className="text-4xl md:text-6xl font-bold mb-8 text-gray-900 dark:text-white"
        >
          Vector Search is Exploding—<br />
          But It&apos;s an Operational Nightmare
        </motion.h2>
        <motion.div
          variants={staggerContainer}
          className="space-y-6 text-lg md:text-xl text-gray-700 dark:text-gray-300"
        >
          <motion.p variants={fadeInUp}>
            AI/ML teams need hybrid text + vector pipelines for LLMs, personalization, and observability.
          </motion.p>
          <motion.p variants={fadeInUp}>
            Existing vector DBs (e.g., Qdrant, Milvus) require sidecar services, data syncing, and schema mismatches—adding latency, complexity, and costs.
          </motion.p>
          <motion.p variants={fadeInUp} className="font-semibold">
            Redis powers 100M+ deployments, but lacks native vector search, forcing fragmented stacks.
          </motion.p>
        </motion.div>
        <motion.div
          variants={fadeInUp}
          className="mt-12 p-6 bg-gray-100 dark:bg-gray-800 rounded-lg"
        >
          <div className="grid grid-cols-2 gap-4 text-center">
            <div>
              <div className="text-3xl font-bold text-indigo-600 dark:text-indigo-400">100M+</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Redis Deployments</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">$4B+</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Vector DB Market by 2028</div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </section>
  );
}

