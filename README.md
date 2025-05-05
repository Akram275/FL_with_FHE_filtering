## 🔒 Robust Federated Learning via Encrypted SVM Filtering

This repository implements a Byzantine-resilient aggregation strategy for Federated Learning (FL) by filtering malicious client updates using Support Vector Machines (SVMs).  
Crucially, filtering is done **directly over Fully Homomorphically Encrypted (FHE) updates** using the **CKKS** encryption scheme — ensuring privacy is preserved throughout.

SVMs are trained on shadow updates representing both benign and adversarial behaviors, allowing the server to filter encrypted updates in a privacy-preserving and attack-aware manner.

### Highlights
-  **SVM Filtering**: Classifiers trained to detect adversarial updates (offline on shadow updates).
-  **FHE (CKKS)**: Enables inference over encrypted model updates.
-  **Attack-Resilience**: Tested against Backdoor Attacks, Gradient Ascent, and Label Flipping.

---

##  Repository Structure

```bash
.
├── CKKS_filtering_benchmarks/
│   ├── ckks_benchmark_SPCA_svm.py      # Full benchamrks for encrypted SVM inference using CKKS with the Tenseal Library https://arxiv.org/abs/2104.03152  
│   ├── estimate_params.py              # A sage file for estimating the security level of the used CKKS parameters using the lattice-estimator library :       |   |                                    https://github.com/malb/lattice-estimator
│   └── README.md                       # Describes encrypted filtering benchmarks and usage
│   ├── shadow_updates/                 # Folder containing the shadow updates used for training the SVM filters over each Byzantine behaviour (property).
|
├── FL_runs/
│   ├── fl_utils.py                      # Federated Learning helper functions for aggregation/local update/setting up malicious behaviours.
│   ├── femnist_fl_backdoor.py           # FL runs with adversarial backdoor workers
│   ├── femnist_fl_gradient_ascent.py    # FL runs with poisoned labels
│   ├── SupervisedPCA.py                 # Supervised PCA implemenation: Provides an SPCA component for the SVM filtering process.
│   └── README.md                        # Details the FL setup and results

