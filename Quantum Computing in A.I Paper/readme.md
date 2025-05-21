# Quantum Computing in Artificial Intelligence: A Review of Quantum Machine Learning Algorithms

📄 **Full paper**: [Read here](https://docs.google.com/document/d/1-xxpKzi6t_dOV3oahvisnHetSFbRXogs/edit?usp=sharing&ouid=116911134464626065147&rtpof=true&sd=true)

## 📌 Abstract

This paper reviews the intersection of two disruptive fields: **Quantum Computing (QC)** and **Artificial Intelligence (AI)**, focusing on **Quantum Machine Learning (QML)**. It evaluates the theoretical frameworks, algorithmic implementations, and real-world applications of various QML algorithms, such as:

- Quantum Support Vector Machines (QSVM)
- Quantum k-Nearest Neighbors (QkNN)
- Quantum Principal Component Analysis (QPCA)
- Quantum Neural Networks (QNN)
- Quantum Reinforcement Learning (QRL)

While these methods show theoretical speed-ups in tasks like classification and optimization, practical limitations persist due to hardware constraints and algorithmic challenges.

---

## 🎯 Objectives

- Summarize key QML algorithms
- Present state-of-the-art advancements and tools
- Discuss empirical performance and practical applications
- Identify open challenges and future research directions

---

## 📚 Literature Background

- **Evolution**: QML emerged from combining classical ML and quantum computing principles. It builds on foundational algorithms like Shor’s and Grover’s.
- **Limitations of Classical ML**: Struggles with high-dimensional data, local minima, and computational bottlenecks.
- **Gaps in Existing Surveys**: Prior reviews lacked comparative analysis and real-world benchmarking of QML implementations.

---

## 🔬 Methodology

- **Sources**: Articles from IEEE, Springer, arXiv, etc., from 2014–2024.
- **Criteria**: Inclusion required a technical focus on ML + QC integration.
- **Framework**: Algorithms evaluated by structure, complexity, and implementation feasibility.

---

## 📐 Mathematical Foundations

- QML relies on **linear algebra of Hilbert spaces** and **Dirac notation (|ψ⟩)**.
- Techniques like **quantum kernels**, **density matrices**, and **phase estimation** underpin many QML models.
- Emphasis is on **hybrid quantum-classical systems** using parametrized quantum circuits (PQCs).

---

## 🔍 Key QML Algorithms Reviewed

| Algorithm | Type          | Speed-Up                   | Implementation     | Limitations                  |
|----------|---------------|----------------------------|--------------------|------------------------------|
| **QSVM** | Supervised     | Polynomial to exponential  | Early-stage        | Quantum kernel noise         |
| **QkNN** | Supervised     | Polynomial                 | Mostly theoretical | qRAM impractical             |
| **QPCA** | Unsupervised   | Exponential (idealized)    | Simulated          | Requires pure state input    |
| **QNN/VQC** | Supervised  | Moderate (hybrid)          | On NISQ hardware   | Barren plateaus              |
| **QRL**  | Reinforcement  | Unknown                    | Theoretical        | No quantum feedback systems  |

---

## 🛠️ Practical Applications

- **Natural Language Processing (QNLP)**: Quantum circuits for semantics and grammar.
- **Drug Discovery**: Molecular simulation and quantum chemistry.
- **Finance**: Portfolio optimization, fraud detection.
- **Recommender Systems**: Enhanced speed in high-dimensional matrices.

> Most applications are still **proof-of-concept**, with limited datasets and simulated environments.

---

## ⚠️ Limitations & Challenges

- **Hardware**: Noise, coherence times, and scalability are major obstacles.
- **Software**: Fragmented ecosystem with inconsistent tooling (e.g., Qiskit, Cirq, Pennylane).
- **Training**: Optimization plagued by barren plateaus.
- **Benchmarking**: Lack of standard datasets and APIs for fair comparison.

---

## ✅ Conclusions

Quantum Machine Learning offers theoretical advances that can potentially outperform classical ML in terms of **speed** and **scalability**. Yet, practical implementation remains limited by current hardware and engineering hurdles. Continued progress in quantum processors, hybrid algorithm development, and benchmarking standards is essential for the field’s maturation.

---

## 📖 For Further Reading

📄 **Full paper**: [Quantum Computing in AI – Google Doc](https://docs.google.com/document/d/1-xxpKzi6t_dOV3oahvisnHetSFbRXogs/edit?usp=sharing&ouid=116911134464626065147&rtpof=true&sd=true)
