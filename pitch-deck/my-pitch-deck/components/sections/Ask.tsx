"use client";

import { motion } from "framer-motion";
import { fadeInUp, staggerContainer } from "@/lib/animations";

const useOfFunds = [
  { category: "Engineering", percentage: 50, description: "4-6 person team: search, systems, DevRel" },
  { category: "Cloud Product", percentage: 30, description: "Managed service, observability, billing" },
  { category: "GTM", percentage: 20, description: "Partnerships, content, design partners" },
];

const milestones = [
  {
    timeframe: "6 months",
    items: [
      "10 paying partners",
      "Managed beta",
      ">500 vec/s ingestion",
    ],
  },
  {
    timeframe: "12 months",
    items: [
      "GA cloud offering",
      "Multi-region HA",
      ">$1M ARR target",
      "Public benchmarks vs. Qdrant/Milvus",
    ],
  },
];

export default function Ask() {
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
          $5-8M Seed+ / Early Series A:<br />
          Fuel to Outpace the Field
        </motion.h2>
        <motion.p
          variants={fadeInUp}
          className="text-xl text-gray-600 dark:text-gray-400 mb-12 text-center"
        >
          Qdrant&apos;s $28M Series A shows the appetite. Our $5-8M is lean: Builds the moat while dilution stays low.
        </motion.p>

        <div className="grid md:grid-cols-2 gap-12 mb-12">
          <motion.div
            variants={fadeInUp}
            className="bg-gray-50 dark:bg-gray-800 p-8 rounded-xl border border-gray-200 dark:border-gray-700"
          >
            <h3 className="text-2xl font-semibold mb-6 text-gray-900 dark:text-white">Use of Proceeds</h3>
            <div className="space-y-4">
              {useOfFunds.map((item, index) => (
                <div key={index}>
                  <div className="flex justify-between mb-2">
                    <span className="font-medium text-gray-900 dark:text-white">{item.category}</span>
                    <span className="font-bold text-indigo-600 dark:text-indigo-400">{item.percentage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                    <div
                      className="h-3 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500"
                      style={{ width: `${item.percentage}%` }}
                    ></div>
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">{item.description}</div>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div
            variants={fadeInUp}
            className="bg-gray-50 dark:bg-gray-800 p-8 rounded-xl border border-gray-200 dark:border-gray-700"
          >
            <h3 className="text-2xl font-semibold mb-6 text-gray-900 dark:text-white">Milestones</h3>
            <div className="space-y-6">
              {milestones.map((milestone, index) => (
                <div key={index}>
                  <div className="font-semibold text-lg text-indigo-600 dark:text-indigo-400 mb-3">
                    {milestone.timeframe}
                  </div>
                  <ul className="space-y-2">
                    {milestone.items.map((item, itemIndex) => (
                      <li key={itemIndex} className="text-gray-700 dark:text-gray-300 flex items-start">
                        <span className="mr-2 text-green-500">✓</span>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        <motion.div
          variants={fadeInUp}
          className="bg-gradient-to-r from-indigo-600 to-purple-600 p-8 rounded-2xl text-white text-center"
        >
          <p className="text-xl mb-4">
            Qdrant proved demand with $37.8M. RedVector proves execution: Beating them in 2 weeks solo.
          </p>
          <p className="text-2xl font-bold">
            With $5-8M, we become the default inside every Redis deployment.
          </p>
        </motion.div>
      </motion.div>
    </section>
  );
}

