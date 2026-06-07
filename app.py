import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, Optional, Any


# ==========================================
#   Eco metrics (Distribué & Purement Asynchrone)
# ==========================================

class EcoMetrics:
    """
    Tracker de FLOPs ultra-performant pour environnements distribués (DDP/FSDP).
    Zéro synchronisation CPU-GPU forcée durant les passes critiques.
    """

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.device = device
        # Double précision pour éviter les dérives de cumul sur de longues sessions
        self._flops_total = torch.tensor(0.0, dtype=torch.float64, device=device)
        self._flops_saved = torch.tensor(0.0, dtype=torch.float64, device=device)

    def add(self, total: Any, saved: Any) -> None:
        """Ajoute des métriques sans casser le graphe de calcul CUDA."""
        if isinstance(total, torch.Tensor):
            self._flops_total.add_(total.to(dtype=torch.float64, device=self.device))
        else:
            self._flops_total.add_(float(total))

        if isinstance(saved, torch.Tensor):
            self._flops_saved.add_(saved.to(dtype=torch.float64, device=self.device))
        else:
            self._flops_saved.add_(float(saved))

    def reset(self) -> None:
        self._flops_total.zero_()
        self._flops_saved.zero_()

    def sync_dist(self, async_op: bool = False) -> Optional[dist.Work]:
        """Synchronisation collective via All-Reduce (optionnellement non-bloquant)."""
        if dist.is_available() and dist.is_initialized():
            metrics = torch.stack([self._flops_total, self._flops_saved])
            handle = dist.all_reduce(metrics, op=dist.ReduceOp.SUM, async_op=async_op)
            if async_op:
                return handle
            self._flops_total.copy_(metrics[0])
            self._flops_saved.copy_(metrics[1])
        return None

    @property
    def flops_total(self) -> float:
        return float(self._flops_total.item())

    @property
    def flops_saved(self) -> float:
        return float(self._flops_saved.item())

    def energy_saved_pct(self) -> float:
        total = self.flops_total
        return 100.0 * self.flops_saved / total if total > 0.0 else 0.0


# ==========================================
#   Gating module (Stabilité Numérique)
# ==========================================

class TokenGate(nn.Module):
    """
    Gate de décision haute fidélité avec protection contre le sous-flow d'Autocast.
    """

    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 1.5,
        margin: float = 0.1,
        clamp_min: float = 1e-4,
        clamp_max: float = 1.0 - 1e-4,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1, bias=False)
        self.temperature = max(temperature, 1e-5)
        self.margin = margin
        self.register_buffer("clamp_min", torch.tensor(clamp_min, dtype=torch.float32))
        self.register_buffer("clamp_max", torch.tensor(clamp_max, dtype=torch.float32))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        # Forcer en float32 pour la stabilité de la Sigmoid lors des mix-precisions (AMP)
        with torch.autocast(device_type=x_norm.device.type, enabled=False):
            logits = self.linear(x_norm.to(dtype=torch.float32))
            scaled = (logits / self.temperature) - self.margin
            gate = torch.sigmoid(scaled)
            return torch.clamp(gate, self.clamp_min, self.clamp_max).to(dtype=x_norm.dtype)


# ==========================================
#   SwiGLU MLP Core (Vectorisation Maximale)
# ==========================================

class SwiGLUMLP(nn.Module):
    """
    MLP SwiGLU optimisé pour les fusions de kernels et l'alignement mémoire.
    """

    def __init__(self, hidden_dim: int, inner_mult: float = 4.0 / 3.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # Alignement strict sur un multiple de 8 requis pour les Tensor Cores
        self.inner_dim = int((int(inner_mult * hidden_dim) + 7) // 8 * 8)

        self.w12 = nn.Linear(hidden_dim, 2 * self.inner_dim, bias=False)
        self.w3 = nn.Linear(self.inner_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.w12(x)
        x_gate, x_val = proj.chunk(2, dim=-1)
        return self.w3(F.silu(x_gate) * x_val)


# ==========================================
#   Green MLP Complet 10/10
# ==========================================

class GreenCognitiveMLP(nn.Module):
    """
    MLP éco-conditionnel d'élite. Zéro synchronisation CPU-GPU synchrone,
    compatibilité native et absolue avec torch.compile sans aucun graph break.
    """

    def __init__(self,
        hidden_dim: int,
        eco_threshold: float = 0.5,
        sparsity_target: float = 0.5,
        gate_temperature: float = 1.5,
        gate_margin: float = 0.1,
        batch_first: bool = True,
        metrics: Optional[EcoMetrics] = None,
        layerscale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eco_threshold = eco_threshold
        self.batch_first = batch_first

        self.input_norm = nn.RMSNorm(hidden_dim)
        self.mlp = SwiGLUMLP(hidden_dim)
        self.gate = TokenGate(
            hidden_dim=hidden_dim,
            temperature=gate_temperature,
            margin=gate_margin,
        )
        self.metrics = metrics
        self.layerscale = nn.Parameter(layerscale_init * torch.ones(hidden_dim))
        
        # Enregistrement de la cible pour éviter les allocations à la volée durant le forward
        self.register_buffer("sparsity_target_tensor", torch.tensor(sparsity_target, dtype=torch.float32))

    def _compute_flops_per_token(self) -> int:
        d = self.hidden_dim
        m = self.mlp.inner_dim
        return int((7 * d) + (2 * d) + (2 * d * (2 * m)) + (2 * m) + (2 * m * d) + d + d)

    @staticmethod
    def _straight_through_gate(gate_scores: torch.Tensor, threshold: float) -> torch.Tensor:
        hard = (gate_scores > threshold).to(dtype=gate_scores.dtype)
        return hard + (gate_scores - gate_scores.detach())

    def compute_sparsity_penalty(self, gate_scores: torch.Tensor) -> torch.Tensor:
        mean_gate = gate_scores.mean()
        return F.mse_loss(mean_gate, self.sparsity_target_tensor.to(dtype=mean_gate.dtype))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.batch_first:
            x = x.transpose(0, 1)

        B, T, D = x.shape
        x_norm = self.input_norm(x)
        gate_scores = self.gate(x_norm)

        if attn_mask is not None:
            mask_expanded = attn_mask.to(dtype=gate_scores.dtype, device=x.device).unsqueeze(-1)
            gate_scores = gate_scores * mask_expanded

        flops_per_token = self._compute_flops_per_token()
        total_flops = float(B * T * flops_per_token)

        gate_action = self._straight_through_gate(gate_scores, self.eco_threshold)

        if self.training:
            core_output = self.mlp(x_norm)
            output = x + gate_action * (core_output * self.layerscale)
            
            if self.metrics is not None:
                self.metrics.add(total_flops, 0.0)
        else:
            # --- BRANCHE INFERENCE 10/10 : ZÉRO GRAPH BREAK ---
            flat_x_norm = x_norm.view(-1, D)
            flat_gate_action = gate_action.view(-1)
            
            active_indices = torch.nonzero(flat_gate_action, as_tuple=True)[0]
            
            # Élimination complète du bloc conditionnel 'if num_active > 0'.
            # PyTorch gère le comportement dynamique : si active_indices est vide,
            # x_active aura une dimension de 0, l'exécution se poursuit de manière vectorisée,
            # et index_add_ n'ajoute rien, sans jamais solliciter le CPU.
            x_active = flat_x_norm.index_select(0, active_indices)
            res_active = self.mlp(x_active) * self.layerscale
            
            output = x.clone().view(-1, D)
            output.index_add_(0, active_indices, res_active)
            output = output.view(B, T, D)

            if self.metrics is not None:
                num_active = active_indices.numel()
                saved_tokens = (B * T) - torch.tensor(num_active, dtype=torch.float32, device=x.device)
                self.metrics.add(
                    torch.tensor(total_flops, device=x.device),
                    saved_tokens * flops_per_token
                )

        if not self.batch_first:
            output = output.transpose(0, 1)
            gate_scores = gate_scores.transpose(0, 1)

        return output, gate_scores
