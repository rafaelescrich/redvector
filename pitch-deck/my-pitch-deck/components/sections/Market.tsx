"use client";

import { motion } from "framer-motion";
import { fadeInUp, staggerContainer } from "@/lib/animations";

export default function Market() {
  return (
    <section className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-50 to-pink-50 dark:from-gray-900 dark:to-purple-900 py-20 px-6">
      <motion.div
        variants={staggerContainer}
        initial="initial"
        whileInView="animate"
        viewport={{ once: true, margin: "-100px" }}
        className="max-w-6xl mx-auto"
      >
        <motion.h2
          variants={fadeInUp}
          className="text-4xl md:text-6xl font-bold mb-12 text-center text-gray-900 dark:text-white"
        >
          Tapping the $4B+ Vector DB Market<br />
          via Redis Ecosystem
        </motion.h2>

        <motion.div
          variants={fadeInUp}
          className="grid md:grid-cols-3 gap-8 mb-12"
        >
          <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <div className="text-4xl font-bold text-indigo-600 dark:text-indigo-400 mb-2">$1B+</div>
            <div className="text-gray-600 dark:text-gray-400">Redis Ecosystem Annual Revenue</div>
          </div>
          <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <div className="text-4xl font-bold text-purple-600 dark:text-purple-400 mb-2">$4B+</div>
            <div className="text-gray-600 dark:text-gray-400">Vector DB Market by 2028</div>
          </div>
          <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <div className="text-4xl font-bold text-pink-600 dark:text-pink-400 mb-2">100M+</div>
            <div className="text-gray-600 dark:text-gray-400">Redis Deployments</div>
          </div>
        </motion.div>

        <motion.div
          variants={fadeInUp}
          className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-8 rounded-xl border border-gray-200 dark:border-gray-700"
        >
          <h3 className="text-2xl font-semibold mb-6 text-gray-900 dark:text-white">Key Verticals</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <div className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">AI/ML</div>
              <div className="text-gray-600 dark:text-gray-400">LLMs, chatbots, embeddings</div>
            </div>
            <div>
              <div className="font-semibold text-purple-600 dark:text-purple-400 mb-2">Gaming</div>
              <div className="text-gray-600 dark:text-gray-400">Personalization, recommendations</div>
            </div>
            <div>
              <div className="font-semibold text-pink-600 dark:text-pink-400 mb-2">Ad Tech</div>
              <div className="text-gray-600 dark:text-gray-400">Real-time targeting, similarity search</div>
            </div>
            <div>
              <div className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">SaaS</div>
              <div className="text-gray-600 dark:text-gray-400">Observability, search, analytics</div>
            </div>
          </div>
        </motion.div>

        <motion.p
          variants={fadeInUp}
          className="mt-12 text-center text-lg text-gray-700 dark:text-gray-300"
        >
          We&apos;re not starting from scratch—we&apos;re enhancing the world&apos;s most popular key-value store. 
          Millions of Redis instances are ripe for vector upgrades.
        </motion.p>
      </motion.div>
    </section>
  );
}

