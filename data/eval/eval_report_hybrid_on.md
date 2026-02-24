# RAG Evaluation Report

## Overview

* Dataset size: 30 queries
* Categories: common, complex, edge, ambiguous, not_available
* Evaluation focus: retrieval quality, grounded generation, answer quality, and system behavior under ambiguity and insufficient knowledge.

Evaluation follows the **RAG Triad**:

1. Context Relevance (retrieval quality)
2. Groundedness (faithfulness to retrieved context)
3. Answer Relevance (usefulness to the user's question)

---

## RAG Triad Results

| Metric            | Result |
| ----------------- | ------ |
| Context Relevance | 20 / 20 (100%) |
| Groundedness      | 19 / 20 (95%) |
| Answer Relevance  | 20 / 20 (100%) |

Notes:

* All evaluated responses were based on relevant retrieved context.
* Hallucination risk observed in 1 response(s).
* All answers directly addressed the user's intent.

---

## System Behavior Metrics

| Behavior                                   | Result |
| ------------------------------------------ | ------ |
| Insufficient context detected correctly    | 5 / 5 (100%) |
| Clarification required (ambiguous queries) | 5 / 5 |
| Clarification asked correctly              | 5 / 5 (100%) |
| Failures / incorrect responses             | 0 / 30 (0%) |

Details:

* For out-of-scope queries, the system declined to answer instead of guessing.
* For ambiguous queries, the system consistently asked clarifying questions before answering.

---

## Results by Category

| Category                     | Count | Outcome |
| ---------------------------- | ----- | ------- |
| Common | 10 | 100% correct |
| Complex | 5 | 100% correct |
| Edge cases | 5 | 4 / 5 (80%) correct |
| Ambiguous | 5 | Clarification triggered correctly (5/5) |
| Not available (out-of-scope) | 5 | Correctly identified insufficient context (5/5) |

---

## Key Observations

* Retrieval pipeline consistently returned relevant context across all answerable queries.
* Generation remained grounded to retrieved evidence.
* The system demonstrated strong scope control, declining queries outside the knowledge base.
* Ambiguity detection and clarification logic worked reliably.
* Overall behavior is consistent with production-oriented RAG system requirements.

---

## Limitations and Next Steps

* Current evaluation size is limited (30 queries); future work includes expanding to 100+ cases.
* Potential future metrics:
* Latency (retrieval and end-to-end response time)
* Context relevance scoring beyond hit/miss
* User satisfaction or preference testing
