# Bindings Generator

This is a rust binary that does the following:
1. Downloads cuda headers from <https://developer.download.nvidia.com/compute/cuda/redist/> for ALL supported cuda versions
2. Generate bindings for each version separately
3. Merge the bindings together to:
    1. Unify static-linking/dynamic-linking/dynamic-loading
    2. Reduce code duplication across toolkit versions (they are generally additive)

Usage:
```bash
cargo run --release
```
