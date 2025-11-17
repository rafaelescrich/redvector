"use client";

import { motion } from "framer-motion";
import { fadeInUp, staggerContainer } from "@/lib/animations";

const roadmapItems = [
  {
    period: "Short-Term (3-6 mo)",
    items: [
      "Parallel ingestion (>500 vec/s)",
      "Real Recall@K metrics",
      "Replication coverage",
    ],
  },
  {
    period: "Mid-Term (6-12 mo)",
    items: [
      "Managed cloud beta",
      "Enterprise security (TLS/auth/HA)",
      "Embedding connectors",
    ],
  },
  {
    period: "Long-Term",
    items: [
      "Multi-tenant SaaS",
      "On-prem appliances",
    ],
  },
];

export default function Roadmap() {
  return (
    <section className="min-h-screen flex items-center justify-center bg-white dark:bg-gray-900 py-20 px-6">
      <motion.div
        variants={staggerContainer}
        initial="initial"
        whileInView="animate"
        viewport={{ once: true, margin: "-100px" }}
        className="max-w-6xl mx-auto"
      >
        <motion.h2
          variants={fadeInUp}
          className="text-4xl md:text-6xl font-bold mb-4 text-gray-900 dark:text-white text-center"
        >
          Clear Path: From Open-Source<br />
          to Managed Dominance
        </motion.h2>
        <motion.p
          variants={fadeInUp}
          className="text-xl text-gray-600 dark:text-gray-400 mb-12 text-center"
        >
          Ecosystem-first: Start with Redis pull, expand to cloud
        </motion.p>

        <div className="relative">
          <div className="absolute left-8 top-0 bottom-0 w-1 bg-gradient-to-b from-indigo-500 via-purple-500 to-pink-500"></div>
          <motion.div
            variants={staggerContainer}
            className="space-y-12"
          >
            {roadmapItems.map((item, index) => (
              <motion.div
                key={index}
                variants={fadeInUp}
                className="relative pl-20"
              >
                <div className="absolute left-6 w-4 h-4 bg-indigo-500 rounded-full border-4 border-white dark:border-gray-900"></div>
                <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-6 rounded-xl border border-gray-200 dark:border-gray-700">
                  <h3 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                    {item.period}
                  </h3>
                  <ul className="space-y-2">
                    {item.items.map((roadmapItem, itemIndex) => (
                      <li key={itemIndex} className="text-gray-700 dark:text-gray-300 flex items-start">
                        <span className="mr-2 text-indigo-500">•</span>
                        {roadmapItem}
                      </li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>

        <motion.div
          variants={fadeInUp}
          className="mt-12 p-6 bg-indigo-50 dark:bg-indigo-900/20 rounded-xl border border-indigo-200 dark:border-indigo-800"
        >
          <p className="text-center text-gray-700 dark:text-gray-300">
            <strong>GTM Strategy:</strong> Partnerships with Redis Labs/VCs; content marketing (benchmarks/blogs); 
            design partners in AI/gaming/SaaS. This roadmap turns experiments into revenue fast.
          </p>
        </motion.div>
      </motion.div>
    </section>
  );
}

