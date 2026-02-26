Benchmarks and Explorations Using AVX-512F's BF16 ISE
=====================================================

## Cosine Similarity

Recall the formula for cosine similarity.

$$
\cfrac{A \cdot B}{\lVert A \rVert \lVert B \rVert}
$$

Where the vector magnitude, denoted as $\lVert A \rVert$, is computed as

$$
\lVert A \rVert = \sqrt{\sum_i a_i^2}
$$

Here we have an AVX-512F BF16 implementation that uses `vdpbf16ps` to compute
both the dot product as well as the squaring required when computing cosim on
non-L2 normalized vectors. In this implementation, quantization happens at
compare time using `vcvtneps2bf16`.

```
$ cargo run
   Compiling bf16 v0.1.0 (/home/dpitt/src/bf16)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.32s
     Running `target/debug/bf16`
processing 5749 vectors
semantic implementation:      187ms
single pass implementation:   123ms
AVX-512F BF16 implementation: 2ms
```

| Implementation | time spent comparing 5749 vectors |
| - | -|
| semantic implementation | 187ms |
| single pass implementation |  123ms |
| **AVX-512F BF16 implementation** | **2ms** |

This is on a GNR-based system
([`6787P`](https://www.intel.com/content/www/us/en/products/sku/241844/intel-xeon-6787p-processor-336m-cache-2-00-ghz/specifications.html))
with TurboBoost disabled.

The data used both for benchmarking and testing is
[`benchmark-sts`](https://huggingface.co/datasets/mteb/stsbenchmark-sts), which
is encoded into 768-d dense vectors using
[`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
The code that does the embedding and dumps the results to binary files can be
found in [`./sources`](./sources).

### Testing

The only verification I've done for this implementation is comparing its results
against two baseline implementations, a "semantic" implementation that is
implemented functionally, as close to the formula for cosim as possible. The
second is the "one pass" implementation, a high-level optimized implementation
that makes a single pass through the compared vectors, but all computations are
serial.

Tests pass with an epsilon of `0.0004`, a slight delta as a result of the
quantization.
