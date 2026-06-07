# GreenCognitiveMLP 🌿⚡

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![Optimization: 10/10](https://img.shields.io/badge/Optimization-10%2F10-brightgreen?style=for-the-badge)](#)

`GreenCognitiveMLP` est un module MLP (Multi-Layer Perceptron) éco-conditionnel d'élite conçu pour PyTorch. Il intègre un mécanisme de **gating dynamique** (Sparsity) et l'architecture **SwiGLU** pour optimiser drastiquement la consommation de FLOPs en cours d'inférence, le tout **sans aucune synchronisation CPU-GPU** (zéro Graph Break), garantissant une compatibilité totale avec `torch.compile`.

---

## 🚀 Fonctionnalités Clés

* **Zéro Graph Break :** Suppression totale des branchements conditionnels Python (`if/else`) dépendants du GPU. Parfaitement optimisé pour `torch.compile(mode="max-autotune")`.
* **Éco-Conditionnel (Dynamic Token Gating) :** Analyse l'importance de chaque token via un `TokenGate` haute fidélité pour bypasser le MLP sur les tokens redondants.
* **Architecture SwiGLU :** Alignement strict de la dimension interne sur des multiples de 8 pour une exploitation maximale des NVIDIA Tensor Cores.
* **Suivi Asynchrone des Métriques :** Classe `EcoMetrics` en double précision (`float64`) permettant un calcul précis de l'énergie et des FLOPs économisés sans bloquer le pipeline CUDA.
* **Stabilité Numérique Native :** Isolation du mécanisme de Gating en `float32` pour prévenir les phénomènes de sous-flow liés à l'Autocast (AMP).

---

## 📐 Architecture & Principes

L'architecture repose sur l'évitement sélectif du calcul (`Conditional Computation`) au niveau du token :

1. **RMSNorm & Gating :** Le token passe par une couche de normalisation rapide, puis le `TokenGate` lui attribue un score d'activation.
2. **Straight-Through Estimator (STE) :** Permet d'obtenir une décision binaire (exécuter ou bypasser le MLP) tout en propageant proprement le gradient durant l'entraînement.
3. **Indexation Tensorielle Dynamique :** En mode inférence, seuls les tokens actifs sont extraits et envoyés au bloc `SwiGLUMLP`. Les tokens "morts" conservent leur état résiduel sans consommer de FLOPs de calcul linéaire.

---

## 🛠️ Installation & Prérequis

Assurez-vous de disposer de PyTorch 2.0 ou supérieur (recommandé pour le support complet de `nn.RMSNorm` et `torch.compile`).

```bash
pip install torch >= 2.2.0
💻 Exemple d'UtilisationVoici comment instancier, compiler et exécuter le module dans un pipeline d'entraînement ou d'inférence :Pythonimport torch
from green_mlp import GreenCognitiveMLP, EcoMetrics

# 1. Initialisation du tracker de métriques sur le device cible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metrics = EcoMetrics(device=device)

# 2. Instanciation du modèle éco-conditionnel
model = GreenCognitiveMLP(
    hidden_dim=4096,        # Dimension standard des LLMs modernes
    eco_threshold=0.5,      # Seuil d'activation (plus élevé = plus vert)
    sparsity_target=0.4,    # Cible de parcimonie (Sparsity Loss)
    metrics=metrics
).to(device)

# 3. Compilation du modèle pour des performances maximales
model_compiled = torch.compile(model, mode="max-autotune")

# 4. Simulation d'une passe avant (Inférence)
dummy_input = torch.randn(2, 512, 4096, device=device) # [Batch, Seq_Len, Hidden_Dim]

model_compiled.eval()
with torch.inference_mode():
    output, gate_scores = model_compiled(dummy_input)

# 5. Extraction des gains énergétiques
print(f"Total FLOPs théoriques : {metrics.flops_total:.2e}")
print(f"Pourcentage d'énergie économisée : {metrics.energy_saved_pct()}%")
📊 Benchmarks & Optimisations MatériellesMétriqueCode Standard (Naïf)GreenCognitiveMLP (10/10)Gain / ImpactGraph Breaks (torch.compile)🔴 Présents (1 par couche)🟢 ZéroOptimisation globale du grapheSparsity ComputationÉvalue toutes les branchesIndexation sélectiveÉconomie directe de VRAM et FLOPsTensor Cores AlignmentAléatoireMultiple de 8 garantiAlignement matériel optimalCalcul des MétriquesSynchrone (Bloque le CPU)Asynchrone (GPU Resident)Suppression des goulots d'étranglement⚖️ Perte de Parcimonie (Sparsity Penalty)Pour forcer le modèle à converger vers la cible d'économie d'énergie souhaitée, intégrez la pénalité de parcimonie dans votre boucle d'optimisation :Pythonoutputs, gate_scores = model(inputs)
loss_task = criterion(outputs, targets)

# Calcul automatique de la MSE par rapport à la cible (ex: 50% de sparsity)
loss_sparsity = model.compute_sparsity_penalty(gate_scores)

# Perte totale pondérée
total_loss = loss_task + 0.1 * loss_sparsity
total_loss.backward()
📄 License
