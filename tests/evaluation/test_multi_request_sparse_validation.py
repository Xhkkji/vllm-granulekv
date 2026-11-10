"""Failure-injection tests for the independent multi-request correctness gate."""
from copy import deepcopy

import pytest
import torch

from evaluation.multi_request_sparse.benchmark import compare_reports
from evaluation.multi_request_sparse.validation import (
    BLOCK_SIZE, cache_tokens, check_cache_contents, check_mapping,
    logical_attention, numerical_error, request_slices, update_logical,
)
from evaluation.paper_reproduction.solidattention.adapter.solidattention import SolidAttentionPolicy


def report():
    payload = SolidAttentionPolicy._tensor_payload(torch.ones(1, 1))
    records = []
    for step in range(2):
        for layer in range(28):
            row = {'decode_step': step, 'layer_index': layer,
                   'attention_blocks': [0, 1, 2], 'history_tokens': 32 + step,
                   'block_size': 16, 'sequence_length': 33 + step,
                   'kv_fingerprints': {'0': 'original-0', '1': 'original-1', '2': 'live'},
                   'live_key': payload, 'live_value': payload, 'attention_output': payload}
            if layer == 27:
                row['logits'] = payload
            records.append(row)
    return {'attention_checks': 56, 'logit_checks': 2, 'records': records}


def test_selection_difference_is_not_reported_as_byte_corruption():
    expected, actual = report(), report()
    actual['records'][0]['attention_blocks'] = [0, 2]
    del actual['records'][0]['kv_fingerprints']['1']
    result = compare_reports(expected, actual, 2)
    assert not result['passed']
    assert result['selection_differences'][0]['expected_only'] == [1]
    assert not any('immutable KV' in f for f in result['failures'])


def test_common_immutable_byte_change_fails():
    expected, actual = report(), report()
    actual['records'][0]['kv_fingerprints']['1'] = 'corrupted'
    result = compare_reports(expected, actual, 2)
    assert not result['passed']
    assert any('immutable KV' in f for f in result['failures'])


@pytest.mark.parametrize('damage', ['missing_layer', 'duplicate', 'missing_fingerprint', 'missing_logits', 'extra_logits', 'bad_step'])
def test_missing_or_duplicate_evidence_cannot_pass(damage):
    expected, actual = report(), report()
    if damage == 'missing_layer':
        actual['records'].pop()
    elif damage == 'duplicate':
        actual['records'][1] = deepcopy(actual['records'][0])
    elif damage == 'missing_fingerprint':
        del actual['records'][0]['kv_fingerprints']['1']
    elif damage == 'missing_logits':
        del actual['records'][27]['logits']
    elif damage == 'extra_logits':
        actual['records'][0]['logits'] = actual['records'][27]['logits']
    else:
        actual['records'][-1]['decode_step'] = 3
    with pytest.raises(RuntimeError):
        compare_reports(expected, actual, 2)


@pytest.mark.parametrize('ids', [('a', 'b'), ('b', 'a')])
@pytest.mark.parametrize('lens,ends,prefills', [([3, 1], [34, 18], 1), ([2, 5], [18, 37], 2), ([1, 1], [18, 37], 0)])
def test_variable_query_boundaries_and_request_order(ids, lens, ends, prefills):
    positions = torch.tensor([p for n, end in zip(lens, ends) for p in range(end-n, end)])
    slices = request_slices(ids, lens, ends, positions, prefills, sum(lens[:prefills]), sum(lens[prefills:]))
    assert slices[ids[0]] == slice(0, lens[0])
    assert slices[ids[1]] == slice(lens[0], sum(lens))
    wrong_positions = positions.clone()
    wrong_positions[lens[0]] += 1
    with pytest.raises(ValueError, match='boundary'):
        request_slices(ids, lens, ends, wrong_positions, prefills, sum(lens[:prefills]), sum(lens[prefills:]))
    with pytest.raises(ValueError):
        request_slices(ids, lens, ends, positions[:-1], prefills, sum(lens[:prefills]), sum(lens[prefills:]))


def packed_fixture():
    # Distinct values per logical token, deliberately non-contiguous physical map.
    table = torch.tensor([[4, 0, 3]], dtype=torch.int32)
    k = torch.arange(48 * 16).reshape(48, 1, 16).half()
    v = -k
    kc, vc = torch.zeros(5, 1, 2, 16, 8).half(), torch.zeros(5, 1, 16, 16).half()
    for t in range(48):
        physical = int(table[0, t // 16])
        kc[physical, :, :, t % 16, :] = k[t].reshape(1, 2, 8)
        vc[physical, :, :, t % 16] = v[t]
    return kc, vc, table, (k, v)


def test_independent_cache_check_detects_swapped_read_mapping_and_bytes():
    kc, vc, table, logical = packed_fixture()
    check_cache_contents(kc, vc, table, [0, 1, 2], 33, logical)
    swapped = table[:, [1, 0, 2]]
    with pytest.raises(ValueError, match='logical source'):
        check_cache_contents(kc, vc, swapped, [0, 1, 2], 33, logical)
    vc[0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match='logical source'):
        check_cache_contents(kc, vc, table, [0, 1, 2], 33, logical)


def test_write_mapping_and_selection_count_are_checked_separately():
    _, _, table, _ = packed_fixture()
    pos = torch.tensor([31, 32])
    slots = torch.tensor([15, 48])
    indices = torch.tensor([0, 1, 2, 999])  # spare capacity must not be consumed
    assert check_mapping(table, slots, [0, 1, 2], indices, 3, pos, 5) == [4, 0, 3]
    with pytest.raises(ValueError, match='write slots'):
        check_mapping(table, slots.flip(0), [0, 1, 2], indices, 3, pos, 5)
    with pytest.raises(ValueError, match='valid length'):
        check_mapping(table, slots, [0, 1, 2], indices, 2, pos, 5)
    with pytest.raises(ValueError, match='aliased'):
        check_mapping(torch.tensor([[4, 4, 3]]), slots, [0, 1, 2], indices, 3, pos, 5)


def test_tail_padding_is_not_kv_and_recomputed_token_is_private():
    kc, vc, table, logical = packed_fixture()
    initial = tuple(x[:32].clone() for x in logical)
    first = update_logical(tuple(x.clone() for x in initial), torch.zeros(1, 1, 16), torch.ones(1, 1, 16), 31)
    second = update_logical(tuple(x.clone() for x in initial), torch.ones(1, 1, 16), torch.zeros(1, 1, 16), 31)
    assert torch.equal(first[0][:31], initial[0][:31])
    assert not torch.equal(first[0][31], second[0][31])
    assert torch.equal(logical[0][:32], initial[0])
    kc[3, :, :, 1:, :] = float('nan')
    vc[3, :, :, 1:] = float('nan')
    check_cache_contents(kc, vc, table, [0, 1, 2], 33, logical)
    assert cache_tokens(kc, vc, table, [0, 1, 2], 33)[0].shape[0] == 33
    with pytest.raises(ValueError, match='gap'):
        update_logical(initial, torch.ones(1, 1, 16), torch.ones(1, 1, 16), 33)


def test_fp32_reference_respects_logical_gaps_and_causal_suffix():
    k = torch.zeros(48, 1, 8)
    v = torch.arange(48).float()[:, None, None].expand(48, 1, 8).clone()
    q = torch.zeros(2, 2, 8)
    out = logical_attention(q, (k, v), [0, 2], torch.tensor([32, 33]), 1.)
    assert torch.allclose(out[0], torch.full((2, 8), (sum(range(16))+32)/17))
    assert torch.allclose(out[1], torch.full((2, 8), (sum(range(16))+32+33)/18))
    wrong = out.clone()
    wrong[0, 0, 0] += 1
    assert not numerical_error(wrong, out)['passed']
    assert not numerical_error(torch.full_like(out, float('nan')), out)['passed']


def test_populate_fixture_is_captured_before_cache_and_requires_full_coverage(tmp_path):
    from types import SimpleNamespace
    from evaluation.multi_request_sparse.validation import BatchReference
    reference = BatchReference.__new__(BatchReference)
    reference.reference_active = False
    reference.prefixes = [list(range(16)), list(range(16))]
    reference.populate_sources, reference.sources = {}, {}
    reference.directory, reference.layers = tmp_path, [None]
    rid = 'populate-0-0-0'
    reference.current = {'ids': [rid], 'positions': torch.arange(16),
                         'intervals': {rid: slice(0, 16)}}
    layer = SimpleNamespace(num_kv_heads=1, head_size=8, layer_name='layer0',
                            impl=SimpleNamespace(_layer_index_from_name=lambda _: 0))
    k, v = torch.arange(128).reshape(16, 8).half(), torch.ones(16, 8).half()
    reference._capture_inputs(layer, (k, k, v))
    reference.populate_sources[rid, 0][2][-1] = False
    with pytest.raises(ValueError, match='missing token'):
        reference.seal_populate(rid, 0)
    reference._capture_inputs(layer, (k, k, v))
    expected = k.clone()
    reference.seal_populate(rid, 0)
    k.zero_()
    assert torch.equal(reference.sources[0, 0][0].flatten(1), expected)
    assert (tmp_path / 'populate_kv.pt').is_file()


def test_sampling_rows_cannot_be_assigned_to_the_other_request():
    from types import SimpleNamespace
    from evaluation.multi_request_sparse.validation import BatchReference
    reference = BatchReference.__new__(BatchReference)
    reference.frames, reference.immutable = [], {}
    metadata = SimpleNamespace(num_prefills=1, num_prefill_tokens=3, num_decode_tokens=1)
    frame = SimpleNamespace(request_ids_to_seq_ids={'batch-1': [1], 'batch-0': [0]},
                            query_lens=[3, 1], seq_lens=[34, 18], attn_metadata=metadata,
                            input_positions=torch.tensor([31, 32, 33, 17]),
                            input_tokens=torch.tensor([7, 8, 9, 10]),
                            sampling_metadata=SimpleNamespace(selected_token_indices=torch.tensor([2, 3]),
                                seq_groups=[SimpleNamespace(seq_ids=[1], prompt_logprob_indices=[], do_sample=True, sample_indices=[0]),
                                            SimpleNamespace(seq_ids=[0], prompt_logprob_indices=[], do_sample=True, sample_indices=[1])]))
    assert reference.begin_forward(frame)['sampling_indices'] == [2, 3]
    frame.sampling_metadata.selected_token_indices = torch.tensor([3, 2])
    with pytest.raises(ValueError, match='sampling rows'):
        reference.begin_forward(frame)


def test_chunked_prefill_only_sampling_requests_have_logits_rows():
    from types import SimpleNamespace as NS
    from evaluation.multi_request_sparse.validation import BatchReference
    reference = BatchReference.__new__(BatchReference)
    reference.frames, reference.immutable = [], {}
    groups = [NS(seq_ids=[0], prompt_logprob_indices=[], do_sample=False, sample_indices=[]),
              NS(seq_ids=[1], prompt_logprob_indices=[], do_sample=True, sample_indices=[0])]
    frame = NS(request_ids_to_seq_ids={'populate-0': [0], 'decode-1': [1]},
               query_lens=[3, 1], seq_lens=[34, 18],
               attn_metadata=NS(num_prefills=1, num_prefill_tokens=3, num_decode_tokens=1),
               input_positions=torch.tensor([31, 32, 33, 17]), input_tokens=torch.tensor([7, 8, 9, 10]),
               sampling_metadata=NS(selected_token_indices=torch.tensor([3]), seq_groups=groups))
    assert reference.begin_forward(frame)['sampling_indices'] == [3]
    frame.sampling_metadata.selected_token_indices = torch.tensor([2, 3])
    with pytest.raises(ValueError, match='sampling rows'):
        reference.begin_forward(frame)
    groups[1].do_sample = False
    frame.sampling_metadata.selected_token_indices = torch.tensor([], dtype=torch.int64)
    assert reference.begin_forward(frame)['sampling_indices'] == []


def test_nonfinite_model_kv_is_not_misclassified_as_transfer_corruption():
    kc, vc, table, logical = packed_fixture()
    logical[0][32, 0, 0] = float('nan')
    kc[3, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='non-finite KV'):
        check_cache_contents(kc, vc, table, [0, 1, 2], 33, logical)
