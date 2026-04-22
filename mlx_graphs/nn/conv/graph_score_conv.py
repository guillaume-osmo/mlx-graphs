"""Paper-facing SCORE aliases for the historical GraphODE implementation.

This module keeps the working `graph_ode_conv` implementation intact while
exposing the paper naming used by the SCORE-GNNs family.
"""

from .graph_ode_conv import (
    INTEGRATORS,
    UPDATE_MODES,
    MLPODEBlock,
    GraphODEAttFPRegressor,
    GraphODEBlock,
    GraphODEBlockAttFP,
    GraphODEBlockDMPNN,
    GraphODEBlockDiffGAT,
    GraphODEBlockDiffGINE,
    GraphODEBlockDiffGNN,
    GraphODEBlockGAT,
    GraphODEBlockGATv2,
    GraphODEBlockGCN,
    GraphODEBlockGINE,
    GraphODEBlockGT,
    GraphODEBlockKADMPNN,
    GraphODEBlockKAGAT,
    GraphODEBlockKAGCN,
    GraphODEDMPNNRegressor,
    GraphODEDiffGATRegressor,
    GraphODEDiffGINERegressor,
    GraphODEDiffGNNRegressor,
    GraphODEGATRegressor,
    GraphODEGATv2Regressor,
    GraphODEGCNRegressor,
    GraphODEGINERegressor,
    GraphODEGTRegressor,
    GraphODEKADMPNNRegressor,
    GraphODEKAGATRegressor,
    GraphODEKAGCNRegressor,
    GraphODERegressor,
)
from .groupgat_conv import GraphODEGroupGATRegressor
from .mogat_conv import GraphODEMoGATRegressor

MLPSCOREBlock = MLPODEBlock

GraphSCOREBlock = GraphODEBlock
GraphSCORERegressor = GraphODERegressor

GraphSCOREBlockDiffGNN = GraphODEBlockDiffGNN
GraphSCOREDiffGNNRegressor = GraphODEDiffGNNRegressor

GraphSCOREBlockDiffGAT = GraphODEBlockDiffGAT
GraphSCOREDiffGATRegressor = GraphODEDiffGATRegressor

GraphSCOREBlockDiffGINE = GraphODEBlockDiffGINE
GraphSCOREDiffGINERegressor = GraphODEDiffGINERegressor

GraphSCOREBlockDMPNN = GraphODEBlockDMPNN
GraphSCOREDMPNNRegressor = GraphODEDMPNNRegressor

GraphSCOREBlockKADMPNN = GraphODEBlockKADMPNN
GraphSCOREKADMPNNRegressor = GraphODEKADMPNNRegressor

GraphSCOREBlockGT = GraphODEBlockGT
GraphSCOREGTRegressor = GraphODEGTRegressor

GraphSCOREBlockGCN = GraphODEBlockGCN
GraphSCOREGCNRegressor = GraphODEGCNRegressor

GraphSCOREBlockKAGCN = GraphODEBlockKAGCN
GraphSCOREKAGCNRegressor = GraphODEKAGCNRegressor

GraphSCOREBlockKAGAT = GraphODEBlockKAGAT
GraphSCOREKAGATRegressor = GraphODEKAGATRegressor

GraphSCOREBlockGAT = GraphODEBlockGAT
GraphSCOREGATRegressor = GraphODEGATRegressor

GraphSCOREBlockGATv2 = GraphODEBlockGATv2
GraphSCOREGATv2Regressor = GraphODEGATv2Regressor

GraphSCOREBlockGINE = GraphODEBlockGINE
GraphSCOREGINERegressor = GraphODEGINERegressor

GraphSCOREBlockAttFP = GraphODEBlockAttFP
GraphSCOREAttFPRegressor = GraphODEAttFPRegressor

GraphSCOREGroupGATRegressor = GraphODEGroupGATRegressor
GraphSCOREMoGATRegressor = GraphODEMoGATRegressor

__all__ = [
    "INTEGRATORS",
    "UPDATE_MODES",
    "MLPSCOREBlock",
    "GraphSCOREBlock",
    "GraphSCORERegressor",
    "GraphSCOREBlockDiffGNN",
    "GraphSCOREDiffGNNRegressor",
    "GraphSCOREBlockDiffGAT",
    "GraphSCOREDiffGATRegressor",
    "GraphSCOREBlockDiffGINE",
    "GraphSCOREDiffGINERegressor",
    "GraphSCOREBlockDMPNN",
    "GraphSCOREDMPNNRegressor",
    "GraphSCOREBlockKADMPNN",
    "GraphSCOREKADMPNNRegressor",
    "GraphSCOREBlockGT",
    "GraphSCOREGTRegressor",
    "GraphSCOREBlockGCN",
    "GraphSCOREGCNRegressor",
    "GraphSCOREBlockKAGCN",
    "GraphSCOREKAGCNRegressor",
    "GraphSCOREBlockKAGAT",
    "GraphSCOREKAGATRegressor",
    "GraphSCOREBlockGAT",
    "GraphSCOREGATRegressor",
    "GraphSCOREBlockGATv2",
    "GraphSCOREGATv2Regressor",
    "GraphSCOREBlockGINE",
    "GraphSCOREGINERegressor",
    "GraphSCOREBlockAttFP",
    "GraphSCOREAttFPRegressor",
    "GraphSCOREGroupGATRegressor",
    "GraphSCOREMoGATRegressor",
]
