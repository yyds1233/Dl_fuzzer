This directory contains TXT documents for the 170 APIs that were added to the shape-focused PyTorch API set.

Generation policy:
- direct_docstring: content came from the installed PyTorch object's official docstring.
- source_extracted: content came from PyTorch's bundled `_tensor_docs.py` source because the runtime symbol had no attached docstring.
- compat_note: the requested symbol was not exposed in the local torch build, so the TXT includes a clear compatibility note plus the nearest related official API doc.

Counts:
{
  "direct_docstring": 164,
  "source_extracted": 3,
  "compat_note": 3,
  "failed": 0
}
