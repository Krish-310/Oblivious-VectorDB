# Oblivious VectorDB Roadmap

- [x] Understand and fix ORAM Storage Layer
- [x] Code up HNSW search with ORAM
- [x] Benchmark Performance 

## Basic Evaluation Results
*Note: All ORAM readings fully encapsulate both the algorithm and the storage layer in H2O2RAM.*

### Dataset: Base 100 Vectors
| Method | `M` | `ef_search` | `T_` | `T0_` | 100 Q Time | 100 Q Recall | 1000 Q Time | 1000 Q Recall | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **Regular** | 16 | 50 | - | - | 0.037s | 100% | 6.74s | 100% | Baseline |
| **Oblivious** | 16 | 50 | 10 | 10 | 2.55s | 99.8% | 55.60s | 99.56% | Baseline Oblivious |
| **Oblivious** | 16 | 50 | 10 | 20 | 6.14s | 100% | 94.32s | 99.99% | T0 improved accuracy, increased runtime |
| **Oblivious** | 16 | 25 | 10 | 20 | 6.21s | 100% | 97.80s | 99.99% | Halving ef_search on small dataset changed nothing |
| **Oblivious** | 32 | 25 | 10 | 20 | 15.77s | 100% | 198.62s| 99.97% | Higher runtime, can't be sure about accuracy |

### Dataset: Base 1000 Vectors
| Method | `M` | `ef_search` | `T_` | `T0_` | 100 Q Time | 100 Q Recall | 1000 Q Time | 1000 Q Recall | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **Regular** | 16 | 100 | - | - | 2.11s | 100% | 46.65s | 99.97% | Baseline |
| **Oblivious** | 16 | 50 | 10 | 20 | 6.07s | 78.2% | 100.28s| 71.76% | Significant accuracy drop (Traversal bounds too short) |
| **Oblivious** | 16 | 100 | 20 | 40 | 15.88s | 99.7% | 195.68s| 99.46% | Accuracy fully recovered with properly scaled T bounds |