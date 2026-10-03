import numpy as np
import pytest
import torch

from methods.srd_tnp_bqd.scan_attention import scan_attention, dense_attention
from methods.srd_tnp_bqd.historical import HistoricalModel
from methods.srd_tnp_bqd.data import SourceContext


@pytest.fixture(autouse=True)
def small_test_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dtype,atol,rtol", [(torch.float64, 1e-8, 1e-6), (torch.float32, 1e-5, 1e-4)])
def test_scan_forward_and_backward_dense(dtype, atol, rtol):
    torch.manual_seed(11)
    inputs = [torch.randn(2, n, 8, dtype=dtype, requires_grad=True) for n in (13, 19, 19)]
    dense = dense_attention(*inputs)
    gradient = torch.randn_like(dense)
    dg = torch.autograd.grad(dense, inputs, gradient)
    scan = scan_attention(*inputs, query_chunk=5, key_chunk=7, fused=False)
    sg = torch.autograd.grad(scan, inputs, gradient)
    torch.testing.assert_close(scan, dense, atol=atol, rtol=rtol)
    for a, b in zip(dg, sg):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def fixture():
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    cfg = dict(d_model=128, h_dim=32, heads=4, ffn_dim=256, dropout=.1,
               query_chunk=128, key_chunk=256, layers=3, fourier_features=32, fourier_scales=[1, 2, 4, 8])
    model = HistoricalModel(cfg).eval()
    sources = [SourceContext(str(i), rng.random((n, 4)), rng.normal(size=(n, 1)), np.ones(n, bool)) for i, n in enumerate([11, 17])]
    return model, sources, rng.random((7, 4))


def test_context_and_source_permutation_and_query_batches():
    model, contexts, x = fixture()
    output = model.predict(contexts, x)
    shuffled = [SourceContext(c.source_id, c.x[::-1].copy(), c.z[::-1].copy(), c.valid[::-1].copy()) for c in reversed(contexts)]
    second = model.predict(shuffled, x, batch_size=2)
    np.testing.assert_allclose(output.m, second.m, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(output.h, second.h, atol=1e-5, rtol=1e-4)
    reordered = model.predict(contexts, x[::-1].copy())
    np.testing.assert_allclose(output.m, reordered.m[::-1], atol=1e-5, rtol=1e-4)
    assert output.m.shape == (7, 1) and output.h.shape == (7, 32)
    np.testing.assert_allclose(np.linalg.norm(output.h, axis=1), 1, atol=1e-6)


def test_response_binding_matters():
    model, contexts, x = fixture()
    a = model.predict(contexts, x)
    changed = [SourceContext(c.source_id, c.x, c.z[::-1].copy(), c.valid) for c in contexts]
    b = model.predict(changed, x)
    assert np.max(np.abs(a.m - b.m)) > 1e-5
    assert np.max(np.abs(a.h - b.h)) > 1e-5


def test_full_context_cache_matches_direct_forward():
    model, contexts, x = fixture()
    with torch.no_grad():
        direct_m, direct_h = model(contexts, x)
    cached = model.predict(contexts, x, batch_size=2)
    np.testing.assert_allclose(cached.m, direct_m.numpy(), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(cached.h, direct_h.numpy(), atol=1e-5, rtol=1e-4)


def test_empty_sources_and_masked_invalid_rows():
    model, contexts, x = fixture()
    empty = SourceContext("empty", np.empty((0, 4)), np.empty((0, 1)), np.zeros(0, bool))
    masked = SourceContext("invalid", np.full((5, 4), np.nan), np.full((5, 1), np.nan), np.zeros(5, bool))
    a = model.predict(contexts, x)
    b = model.predict([empty, masked] + contexts, x)
    np.testing.assert_allclose(a.m, b.m)
    assert np.isfinite(model.predict([empty], x).h).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fused_cuda_attention_matches_scan():
    torch.manual_seed(11)
    args = [torch.randn(1, 4, n, 32, device="cuda", requires_grad=True) for n in (13, 19, 19)]
    a = scan_attention(*args, query_chunk=5, key_chunk=7, fused=True)
    b = scan_attention(*args, query_chunk=5, key_chunk=7, fused=False)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    ga = torch.autograd.grad(a.sum(), args)
    gb = torch.autograd.grad(b.sum(), args)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)
