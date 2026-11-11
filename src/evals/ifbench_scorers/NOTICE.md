# IFBench scorer provenance

`instructions.py`, `instructions_registry.py`, `instructions_util.py`, and
`evaluation_lib.py` are vendored from
[allenai/IFBench](https://github.com/allenai/IFBench) commit
`1091c4c3de6c1f6ed12c012ed68f11ea450b0117` under Apache-2.0.

Repository-local changes are limited to package-relative imports and removing automatic
NLTK downloads at import time. The verifier logic is otherwise unchanged.
