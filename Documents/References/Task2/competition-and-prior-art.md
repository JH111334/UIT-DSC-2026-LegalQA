# Task 2 evidence map

UIT DSC identity: Legal Question Answering on the dedicated
[Codabench competition](https://www.codabench.org/competitions/17716/). The local
warm-up file and Drive URL came from the organizer message supplied by the user;
the public page does not expose the full scoring/submission specification.

| Evidence | Finding | Use |
|---|---|---|
| [ALQAC 2024 task page](https://sites.google.com/view/alqac-2024/home) | Analogous QA includes extractive, yes/no, and multiple-choice forms | Add type analysis after official schema arrives |
| [ALQAC 2024 summary](https://openreview.net/forum?id=uDg8v77IBe) | Retrieval and human/expert answer evaluation are separate | Keep retrieval and answer evidence separate |
| [NeCo at ALQAC 2023](https://arxiv.org/abs/2309.05500) | Uses question-type-specific extraction and enrichment | Retain an extractive path before generation |
| [NOWJ1 at ALQAC 2023](https://arxiv.org/abs/2309.09070) | Splits answer extraction from classification | Avoid one prompt for every answer type |
| [ViLQA paper and code](https://github.com/ntphuc149/ViLQA) | Compares span extraction and answer generation for Vietnamese legal QA | Benchmark both paths when corpus is available |
| [se7enese implementation](https://github.com/baohl00/alqac24) | Public ALQAC 2024 code uses retrieval plus an LLM | Implementation reference; its non-allowlisted model is rejected |
| [Vietnamese Law QA system](https://github.com/ngothanhnam0910/Vietnamese-Law-Question-Answering-system) | Full hybrid application includes external APIs | Architecture reference only; external models/services rejected |

Decision: evidence-bound extraction is the baseline. Qwen2.5-1.5B-Instruct is the
first general generator comparator; Vi-Qwen2-1.5B-RAG is the retrieval-tuned
challenger. Neither is enabled or claimed better before held-out evaluation.
