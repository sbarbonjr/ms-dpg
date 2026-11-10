# Paper reproducibility

This directory is deliberately separate from `src/msdpg`. It contains the frozen protocols, a pinned environment, and selected compact result artifacts from the accompanying study.

```bash
conda env create -f environment.yml
conda activate msdpg-paper
make check
```

The full OpenML refits are not run by default: the nine-benchmark extension requires roughly three hours on 36 CPU cores. `make test` validates the packaged numerical implementation and frozen parity fixtures.

## Protocol integrity note

The copied `extended_benchmarks/PROTOCOL.md` has SHA-256
`788406083c82e258b773f4341169c54efe95c0a37644e83dd38761db0d1af13f`,
whereas its source-side frozen checksum records `63cbe68fab0b04ebbea73fead4443858651db881b7ac817470306e2ae6ee6fda`. Both files are preserved unchanged. Consequently, `make check` fails loudly for this pre-existing discrepancy until the study records an explicit deviation or corrected frozen checksum.
