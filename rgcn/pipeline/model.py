"""RGCN_v2 — unchanged from the released architecture (train_gnn.ipynb cell 11).

Shared spatiotemporal LSTM core with a graph-convolution term on the cell state
(c_t = f_t*(c_t + A @ q_t) + i_t*g_t) and two task heads: wet/dry classification
(sigmoid) and log-discharge regression.

Input to forward(): (num_nodes, seq_len, input_dim) — nodes are the batch dim —
or (batch, num_nodes, seq_len, input_dim) to run several windows in one pass
(the graph convolution broadcasts A over the leading window dim; the math per
window is identical to the 3-D path).
Output: (num_nodes, seq_len, 2) / (batch, num_nodes, seq_len, 2) with
column 0 = P(wet), column 1 = log-discharge.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def build_adjacency_matrix(graph, node_ids: list) -> np.ndarray:
    """Row-normalized downstream adjacency. Edge (u -> v) means u flows to v, so
    A[v, u] = 1 (neighbor u influences v in the graph convolution)."""
    n = len(node_ids)
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    adj = np.zeros((n, n), dtype=np.float32)
    for u, v in graph.edges():
        if u in node_to_idx and v in node_to_idx:
            adj[node_to_idx[v], node_to_idx[u]] = 1.0
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return adj / row_sums


class RGCN_v2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        adj_matrix: np.ndarray,
        output_dim: int = 2,
        recur_dropout: float = 0.0,
        dropout: float = 0.0,
        return_states: bool = False,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if output_dim != 2:
            raise ValueError("RGCN_v2 expects output_dim=2 (wet/dry + discharge)")

        super().__init__()
        self.register_buffer("A", torch.from_numpy(adj_matrix).float())

        self.weight_q = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_q = nn.Parameter(torch.Tensor(hidden_dim))

        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.output_dim = output_dim
        self.weight_ih = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))

        self.dropout = nn.Dropout(dropout)
        self.recur_dropout = nn.Dropout(recur_dropout)
        self.return_states = return_states

        self.reg_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, 1)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(
        self, x: torch.Tensor,
        init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        single = x.dim() == 3  # (N, seq, F) -> window batch of 1
        if single:
            x = x.unsqueeze(0)
            if init_states is not None:
                init_states = (init_states[0].unsqueeze(0), init_states[1].unsqueeze(0))

        B, N, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(B, N, self.hidden_size, device=x.device)
            c_t = torch.zeros(B, N, self.hidden_size, device=x.device)
        else:
            h_t, c_t = init_states

        x = self.dropout(x)
        HS = self.hidden_size

        for t in range(seq_sz):
            x_t = x[:, :, t, :]
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t = torch.sigmoid(gates[..., :HS])
            f_t = torch.sigmoid(gates[..., HS:HS * 2])
            g_t = torch.tanh(gates[..., HS * 2:HS * 3])
            o_t = torch.sigmoid(gates[..., HS * 3:])

            q_t = torch.tanh(h_t @ self.weight_q + self.bias_q)
            # (N, N) @ (B, N, HS) broadcasts the graph conv over the window dim.
            c_t = f_t * (c_t + torch.matmul(self.A, q_t)) + i_t * self.recur_dropout(g_t)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(2))

        hidden_seq = torch.cat(hidden_seq, dim=2)
        reg_out = self.reg_head(hidden_seq)
        cls_prob = torch.sigmoid(self.cls_head(hidden_seq))
        out = torch.cat([cls_prob, reg_out], dim=3)

        if single:
            out = out.squeeze(0)
            h_t, c_t = h_t.squeeze(0), c_t.squeeze(0)
        if self.return_states:
            return out, (h_t, c_t)
        return out


def create_model(config, adj_matrix: np.ndarray, input_dim: int, device) -> RGCN_v2:
    m = config["model"]
    model = RGCN_v2(
        input_dim=input_dim,
        hidden_dim=int(m["hidden_dim"]),
        adj_matrix=adj_matrix,
        output_dim=2,
        recur_dropout=float(m["recur_dropout"]),
        dropout=float(m["dropout"]),
        return_states=bool(m["return_states"]),
        seed=int(config["training"]["seed"]),
    )
    return model.to(device)
