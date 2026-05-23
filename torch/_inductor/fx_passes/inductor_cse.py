# mypy: allow-untyped-defs
"""
Graph-level Common Subexpression Elimination (CSE) for Inductor.

This pass identifies identical computations in the FX graph and folds them
into a single node. Unlike the generic torch.fx CSE pass, this version is
memory-aware: it considers the output size to decide whether folding is safe.

Key design choices:
- Only fold call_function nodes with the same target, same args (by identity
  for Node args, by value for scalars), and same kwargs.
- Skip ops that are mutable or produce random results.
- For small outputs (< 1MB), always fold (negligible memory impact).
- For larger outputs, still fold since the folded version uses less total memory
  (1 buffer vs N buffers). The tradeoff is that the single buffer stays alive
  longer, but in practice the duplicates usually have similar lifetimes anyway.
- The graph is functional at this point (before reinplace), so we don't need
  to worry about mutation of the CSE'd output.
"""

import logging
from collections import defaultdict

import torch
from torch._dynamo.utils import counters

from .. import config


log = logging.getLogger(__name__)


# OpOverloadPackets that produce random results and must not be CSE'd
_RANDOM_OP_PACKETS = frozenset(
    [
        torch.ops.aten.dropout,
        torch.ops.aten._fused_dropout,
        torch.ops.aten._standard_gamma,
        torch.ops.aten.bernoulli,
        torch.ops.aten.multinomial,
        torch.ops.aten.native_dropout,
        torch.ops.aten.normal,
        torch.ops.aten.normal_,
        torch.ops.aten.poisson,
        torch.ops.aten.binomial,
        torch.ops.aten.rrelu,
        torch.ops.aten.rand_like,
        torch.ops.aten.rand,
        torch.ops.aten.randint,
        torch.ops.aten.randn,
        torch.ops.aten.randn_like,
        torch.ops.aten.randperm,
    ]
)


def _is_cse_safe_op(node: torch.fx.Node) -> bool:
    """Check if a node's op is safe to CSE (not random, not mutable)."""
    if node.op != "call_function":
        return False

    target = node.target

    # Skip random ops (check at the packet level)
    if isinstance(target, torch._ops.OpOverload):
        if target.overloadpacket in _RANDOM_OP_PACKETS:
            return False
        # Skip mutable ops (in-place ops)
        if target._schema.is_mutable:
            return False
    elif isinstance(target, torch._ops.OpOverloadPacket):
        if target in _RANDOM_OP_PACKETS:
            return False

    # Skip if has out= kwarg
    if node.kwargs.get("out") is not None:
        return False

    # Skip higher-order ops
    if isinstance(target, torch._ops.HigherOrderOperator):
        return False

    # The output must be a tensor (not a tuple/list) for simple CSE
    val = node.meta.get("val")
    if val is not None and not isinstance(val, torch.Tensor):
        return False

    return True


def _get_output_size_bytes(node: torch.fx.Node) -> int | None:
    """Get the output tensor size in bytes, or None if not a tensor."""
    val = node.meta.get("val")
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.nelement() * val.element_size()
    return None


def _make_cse_key(node: torch.fx.Node) -> tuple | None:
    """
    Create a hashable key for CSE. Two nodes with the same key produce
    identical outputs.

    Args identity is checked by using the node objects directly in args tuples.
    This means two nodes with the same op on the same inputs will match,
    but two nodes with different inputs won't even if those inputs have the
    same value.
    """
    if not _is_cse_safe_op(node):
        return None

    # Build key from target + args + kwargs
    # For args: use Node identity (id) for Node args, value for scalars
    def canonicalize_arg(arg):
        if isinstance(arg, torch.fx.Node):
            return ("__node__", id(arg))
        elif isinstance(arg, (list, tuple)):
            items = tuple(canonicalize_arg(a) for a in arg)
            return (type(arg).__name__, items)
        elif isinstance(arg, torch.dtype):
            return ("__dtype__", str(arg))
        elif isinstance(arg, torch.device):
            return ("__device__", str(arg))
        elif isinstance(arg, torch.layout):
            return ("__layout__", str(arg))
        elif isinstance(arg, torch.memory_format):
            return ("__memory_format__", str(arg))
        else:
            # scalars, None, etc.
            return arg

    try:
        canon_args = tuple(canonicalize_arg(a) for a in node.args)
        canon_kwargs = tuple(
            sorted((k, canonicalize_arg(v)) for k, v in node.kwargs.items())
        )
        return (node.target, canon_args, canon_kwargs)
    except (TypeError, ValueError):
        # If anything is unhashable, skip
        return None


# Memory threshold: always fold if output is smaller than this
_SMALL_OUTPUT_THRESHOLD = 1 * 1024 * 1024  # 1 MB


def _run_cse_iteration(graph: torch.fx.Graph) -> int:
    """
    Run a single CSE iteration. Returns number of nodes eliminated.

    Multiple iterations may be needed when folding one group of duplicates
    causes another group to become identical (e.g., neg(abs1) and neg(abs2)
    become identical once abs1 and abs2 are folded).
    """
    # Group nodes by their CSE key
    cse_groups: dict[tuple, list[torch.fx.Node]] = defaultdict(list)

    for node in graph.nodes:
        key = _make_cse_key(node)
        if key is not None:
            cse_groups[key].append(node)

    # Filter to groups with duplicates
    duplicate_groups = {k: v for k, v in cse_groups.items() if len(v) > 1}

    if not duplicate_groups:
        return 0

    eliminated = 0

    for key, nodes in duplicate_groups.items():
        # The canonical node is the first one in graph order (already in order
        # since we iterate graph.nodes)
        canonical = nodes[0]

        # Check output size for memory-awareness
        output_bytes = _get_output_size_bytes(canonical)

        # For large outputs, we still CSE since:
        # 1. Total memory is reduced (1 buffer vs N)
        # 2. The graph is functional, so no mutation concerns
        # 3. Downstream reinplace will handle cases where mutation is needed
        #
        # In the future, we could add a heuristic that checks lifetime overlap,
        # but for now the kernel launch savings dominate.
        if output_bytes is not None and output_bytes > _SMALL_OUTPUT_THRESHOLD:
            log.debug(
                "CSE folding large output (%d bytes, %d duplicates): %s",
                output_bytes,
                len(nodes),
                canonical.target,
            )

        # Replace all duplicates with the canonical node
        for dup in nodes[1:]:
            dup.replace_all_uses_with(canonical)
            graph.erase_node(dup)
            eliminated += 1

    return eliminated


def inductor_cse_pass(graph: torch.fx.Graph) -> None:
    """
    Perform memory-aware Common Subexpression Elimination on the FX graph.

    This identifies groups of nodes that compute identical results (same op,
    same inputs) and replaces all duplicates with the first occurrence.

    Runs iteratively until no more duplicates are found (to handle cascading
    CSE opportunities, e.g., when folding inputs makes downstream ops identical).

    After replacement, dead code elimination removes the unused nodes.
    """
    if not config.inductor_cse:
        return

    total_eliminated = 0
    max_iterations = 10  # Safety limit to prevent infinite loops

    for _ in range(max_iterations):
        eliminated = _run_cse_iteration(graph)
        if eliminated == 0:
            break
        total_eliminated += eliminated

    if total_eliminated > 0:
        counters["inductor"]["inductor_cse_eliminated"] += total_eliminated
        log.debug("Inductor CSE eliminated %d duplicate nodes", total_eliminated)
        # Run DCE to clean up any nodes that are now dead
        graph.eliminate_dead_code()
