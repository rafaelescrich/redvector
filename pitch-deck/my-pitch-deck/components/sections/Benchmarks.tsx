"use client";

import { motion } from "framer-motion";
import { fadeInUp, staggerContainer } from "@/lib/animations";

const benchmarkData = [
  {
    scenario: "Small Dataset",
    dataset: "10,000 × 384d",
    redVectorQPS: 76.9,
    qdrantQPS: 364.0,
    redVectorP95: 16.5,
    qdrantP95: 4.2,
    winner: "qdrant",
  },
  {
    scenario: "Medium Dataset",
    dataset: "100,000 × 384d",
    redVectorQPS: 242.9,
    qdrantQPS: 131.5,
    redVectorP95: 5.6,
    qdrantP95: 10.2,
    winner: "redVector",
  },
  {
    scenario: "High Dimension",
    dataset: "100,000 × 768d",
    redVectorQPS: 162.4,
    qdrantQPS: 69.5,
    redVectorP95: 8.4,
    qdrantP95: 16.9,
    winner: "redVector",
  },
];

export default function Benchmarks() {
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
          Already Beating the Competition—<br />
          With Zero Funding
        </motion.h2>
        <motion.p
          variants={fadeInUp}
          className="text-xl text-gray-600 dark:text-gray-400 mb-12 text-center"
        >
          Solo dev effort in ~2 weeks using open-source components
        </motion.p>

        <motion.div
          variants={staggerContainer}
          className="space-y-8"
        >
          {benchmarkData.map((benchmark, index) => (
            <motion.div
              key={index}
              variants={fadeInUp}
              className="bg-gray-50 dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700"
            >
              <h3 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                {benchmark.scenario} ({benchmark.dataset})
              </h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">Search QPS</div>
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium text-indigo-600 dark:text-indigo-400">RedVector</span>
                        <span className="text-sm font-bold">{benchmark.redVectorQPS}</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                        <div
                          className={`h-4 rounded-full ${
                            benchmark.winner === "redVector"
                              ? "bg-green-500"
                              : "bg-indigo-500"
                          }`}
                          style={{
                            width: `${(benchmark.redVectorQPS / Math.max(benchmark.redVectorQPS, benchmark.qdrantQPS)) * 100}%`,
                          }}
                        ></div>
                      </div>
                    </div>
                    <div className="flex-1">
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium text-purple-600 dark:text-purple-400">Qdrant</span>
                        <span className="text-sm font-bold">{benchmark.qdrantQPS}</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                        <div
                          className={`h-4 rounded-full ${
                            benchmark.winner === "qdrant"
                              ? "bg-green-500"
                              : "bg-purple-500"
                          }`}
                          style={{
                            width: `${(benchmark.qdrantQPS / Math.max(benchmark.redVectorQPS, benchmark.qdrantQPS)) * 100}%`,
                          }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">P95 Latency (ms)</div>
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <div className="text-sm font-medium text-indigo-600 dark:text-indigo-400 mb-1">RedVector</div>
                      <div className="text-2xl font-bold">{benchmark.redVectorP95}ms</div>
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-medium text-purple-600 dark:text-purple-400 mb-1">Qdrant</div>
                      <div className="text-2xl font-bold">{benchmark.qdrantP95}ms</div>
                    </div>
                  </div>
                </div>
              </div>
              {benchmark.winner === "redVector" && (
                <div className="mt-4 text-center">
                  <span className="inline-block px-4 py-2 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 rounded-full text-sm font-semibold">
                    🏆 RedVector Wins
                  </span>
                </div>
              )}
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          variants={fadeInUp}
          className="mt-12 p-6 bg-indigo-50 dark:bg-indigo-900/20 rounded-xl border border-indigo-200 dark:border-indigo-800"
        >
          <p className="text-center text-gray-700 dark:text-gray-300">
            <strong>Performance Momentum:</strong> On medium/high workloads, RedVector wins where it matters. 
            Built in 2 weeks by one person. Imagine with a team.
          </p>
        </motion.div>
      </motion.div>
    </section>
  );
}

