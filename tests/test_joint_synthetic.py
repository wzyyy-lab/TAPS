from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN_SCRIPT = ROOT / "scripts" / "61_train_node_value_net.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("train_node_value_net", TRAIN_SCRIPT)
if TRAIN_SPEC is None or TRAIN_SPEC.loader is None:
    raise RuntimeError("Could not load train_node_value_net script")
train_node_value_net = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(train_node_value_net)

from joint.config import JointDDTConfig
from joint.lattice import extract_topk_lattice
from joint.model import NodeValueNet
from joint.pool import SOURCE_DIVERSE, SOURCE_ENTROPY, SOURCE_MARGINAL, build_marginal_candidate_trie, build_union_candidate_trie
from joint.segments import grouped_softmax, propagate_reach_from_edges, sibling_rank_loss
from joint.selector import SCALAR_FEATURE_NAMES, build_edge_feature_batch, propagate_reach, select_joint_tree
from joint.trace import assert_hidden_provenance_matches


class JointSyntheticTests(unittest.TestCase):
    def test_grouped_softmax_includes_other(self) -> None:
        edge_logits = torch.tensor([1.0, 2.0, 3.0])
        parent_ids = torch.tensor([0, 0, 1])
        other_logits = torch.tensor([0.0, 0.5])
        edge_probs, other_probs = grouped_softmax(edge_logits, parent_ids, other_logits)
        sums = torch.zeros(2)
        sums.scatter_add_(0, parent_ids, edge_probs)
        sums = sums + other_probs
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))

    def test_lattice_and_pool_are_prefix_closed(self) -> None:
        logits = torch.randn(4, 32)
        lattice = extract_topk_lattice(logits, topk=8)
        config = JointDDTConfig(candidate_pool_nodes=64, candidate_pool_sequences=8, joint_topk=8)
        trie = build_marginal_candidate_trie(lattice, config)
        self.assertEqual(trie.parents[0].item(), -1)
        for idx in range(1, trie.num_total_nodes):
            self.assertLess(int(trie.parents[idx].item()), idx)

    def test_batched_model_and_selector_shapes(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(5, 64)
        lattice = extract_topk_lattice(logits, topk=8)
        config = JointDDTConfig(candidate_pool_nodes=64, candidate_pool_sequences=8, max_verify_nodes=16, joint_topk=8)
        trie = build_marginal_candidate_trie(lattice, config)
        model = NodeValueNet(
            vocab_size=128,
            scalar_feature_dim=len(SCALAR_FEATURE_NAMES),
            context_hidden_dim=0,
            hidden_dim=32,
            token_embed_dim=16,
        )
        batch = build_edge_feature_batch(trie, lattice, torch.tensor(1), model)
        edge_logits, other_logits = model(batch)
        self.assertEqual(edge_logits.shape, (trie.num_nodes,))
        self.assertEqual(other_logits.shape, (trie.num_total_nodes,))
        q_cond, _ = grouped_softmax(edge_logits, trie.edge_parent_ids, other_logits)
        q_reach = propagate_reach(trie, q_cond, max_depth=lattice.horizon)
        self.assertEqual(q_reach.shape, (trie.num_total_nodes,))
        selection = select_joint_tree(trie, lattice, torch.tensor(1), model, config, prompt_length=4)
        self.assertLessEqual(selection.selected_tree.num_nodes, config.max_verify_nodes)


    def test_propagate_reach_from_edges_uses_roots_and_depths(self) -> None:
        edge_probs = torch.tensor([0.5, 0.25, 0.8])
        parent_ids = torch.tensor([0, 0, 1])
        child_node_ids = torch.tensor([1, 2, 3])
        root_node_ids = torch.tensor([0])
        depths = torch.tensor([1, 1, 2])
        q_reach = propagate_reach_from_edges(edge_probs, parent_ids, child_node_ids, root_node_ids, 4, depths=depths)
        expected = torch.tensor([1.0, 0.5, 0.25, 0.4])
        self.assertTrue(torch.allclose(q_reach, expected, atol=1e-6))

    def test_sibling_rank_loss_does_not_pair_across_parents(self) -> None:
        scores = torch.tensor([0.9, 0.1, 10.0])
        parent_ids = torch.tensor([0, 0, 1])
        positive_mask = torch.tensor([True, False, False])
        loss = sibling_rank_loss(scores, parent_ids, positive_mask, margin=0.05)
        self.assertEqual(float(loss.item()), 0.0)

    def test_stratified_split_keeps_each_dataset_in_validation(self) -> None:
        records = []
        for dataset in ("gsm8k", "mt_bench"):
            for file_idx in range(4):
                for record_idx in range(5):
                    records.append({
                        "_trace_dataset": dataset,
                        "_trace_file": f"{dataset}/{file_idx}.pt",
                        "idx": record_idx,
                    })
        train_records, val_records = train_node_value_net.split_train_val_records(
            records,
            validation_fraction=0.25,
            shuffle_seed=123,
            strategy="stratified",
        )
        val_counts = train_node_value_net.describe_record_groups(val_records)
        train_counts = train_node_value_net.describe_record_groups(train_records)
        self.assertEqual(val_counts, {"gsm8k": 5, "mt_bench": 5})
        self.assertEqual(train_counts, {"gsm8k": 15, "mt_bench": 15})
        train_groups = {train_node_value_net.record_split_group(record) for record in train_records}
        val_groups = {train_node_value_net.record_split_group(record) for record in val_records}
        self.assertFalse(train_groups & val_groups)

    def test_union_pool_has_multiple_sources_and_prefix_closure(self) -> None:
        torch.manual_seed(1)
        lattice = extract_topk_lattice(torch.randn(5, 128), topk=16)
        config = JointDDTConfig(
            candidate_pool_nodes=96,
            candidate_pool_sequences=16,
            joint_topk=16,
            enable_value_beam_pool=True,
            enable_diversity_pool=True,
            enable_entropy_pool=True,
        )
        trie = build_union_candidate_trie(lattice, config)
        sources = set(int(x) for x in trie.source_ids.cpu().tolist())
        self.assertIn(SOURCE_MARGINAL, sources)
        self.assertTrue({SOURCE_DIVERSE, SOURCE_ENTROPY} & sources)
        for idx in range(1, trie.num_total_nodes):
            self.assertLess(int(trie.parents[idx].item()), idx)

    def test_vocab_bounds_are_checked(self) -> None:
        model = NodeValueNet(vocab_size=8, scalar_feature_dim=len(SCALAR_FEATURE_NAMES), hidden_dim=16, token_embed_dim=8)
        logits = torch.randn(2, 16)
        lattice = extract_topk_lattice(logits, topk=4)
        config = JointDDTConfig(candidate_pool_nodes=8, candidate_pool_sequences=4, joint_topk=4)
        trie = build_marginal_candidate_trie(lattice, config)
        trie.token_ids[0] = 99
        with self.assertRaises(ValueError):
            batch = build_edge_feature_batch(trie, lattice, torch.tensor(1), model)
            model(batch)

    def test_no_leakage_provenance_mismatch_raises(self) -> None:
        trace = {"layer_ids": [1], "token_position": "future", "timing": "after_verify", "projection_version": "x"}
        runtime = {
            "layer_ids": [1],
            "token_position": "runtime_target_hidden_last_available",
            "timing": "before_current_round_draft_and_verification",
            "projection_version": "dflash_fc_input",
        }
        with self.assertRaises(ValueError):
            assert_hidden_provenance_matches(trace, runtime)


if __name__ == "__main__":
    unittest.main()
