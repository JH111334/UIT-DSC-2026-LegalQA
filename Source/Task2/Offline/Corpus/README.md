# Corpus offline

Nhánh này nhận code parser, chunking và metadata của `selected-contexts` sau khi owner
dữ liệu bàn giao. Index không thuộc canonical corpus release.

`Offline/release_builder.py` gọi corpus build và ghép output vào release. Các module
audit, sanitation, parser/chunker không build BM25, dense/FAISS hoặc citation graph.
Code index thuộc Online retrieval. Phase A chỉ có thể bàn giao candidate index ngoài
release; Phase B verify/rebuild và ký canonical index manifest.
