# Section 11: Ethics & Responsible AI

**Project:** Fake News & Misinformation Detection System  
**Model:** Fine-tuned `roberta-base` on the LIAR dataset (binary classification: FAKE / REAL)

---

## 11.1 Overview

Automated fact-checking systems occupy a uniquely sensitive position in the NLP landscape. Unlike a spam filter or a product recommender, a fake news detector directly influences how people perceive truth and information. A system that is wrong — or systematically biased — can suppress legitimate speech, reinforce existing prejudices, or be weaponised for political purposes. This document critically examines the ethical implications of deploying such a system.

---

## 11.2 Ethics Impact Statement

### 11.2.1 Who Benefits from the System

| Beneficiary | Benefit |
|---|---|
| **General public / readers** | Quick, low-friction way to sanity-check news statements before sharing them |
| **Journalists and fact-checkers** | Automated triage tool that flags high-priority claims for human review, reducing workload |
| **Social media platforms** | Can be integrated as a first-pass content moderation layer to reduce viral spread of false claims |
| **Researchers and educators** | Demonstrates an end-to-end NLP pipeline; provides a reference implementation for academic study |
| **Non-profit organisations** | NGOs fighting disinformation (e.g., in public health or elections) gain a low-cost deployable tool |

### 11.2.2 Who Could Be Harmed

| Affected Party | Risk | Severity |
|---|---|---|
| **Political minorities and dissidents** | A model trained on US mainstream fact-checks may label legitimate minority political views as "FAKE" due to underrepresentation in training data | High |
| **Non-native English speakers** | The model performs worse on non-standard English (grammatical errors, slang, code-switching), disproportionately penalising non-native speakers' contributions | Medium |
| **Individuals falsely labelled** | A public figure or private individual whose statement is incorrectly classified as FAKE may suffer reputational damage if the system is used irresponsibly | High |
| **Journalists reporting on complex issues** | Nuanced reporting ("A study suggests...", "Some experts claim...") may be classified as uncertain or FAKE due to hedging language | Medium |
| **The model itself as victim of misuse** | Malicious actors could use the system's weaknesses to craft content specifically designed to pass as REAL while being false | High |

---

## 11.3 Bias and Fairness Risks

### 11.3.1 Dataset Bias

The LIAR dataset introduces several structural biases that propagate directly into model behaviour:

**1. Political bias:**  
LIAR is sourced exclusively from PolitiFact, which primarily fact-checks statements by US politicians, predominantly from the two major parties (Democrat and Republican). Statements by third-party politicians, international figures, or grassroots activists are absent. The model may be biased toward classifying rhetoric that deviates from mainstream US political discourse as FAKE.

**2. Temporal bias:**  
Data covers 2007–2016. Political language, cultural references, and misinformation tactics have evolved significantly since then (e.g., COVID-19 misinformation, AI-generated text). The model has no exposure to post-2016 language patterns.

**3. Speaker-level bias:**  
High-profile speakers who made many statements dominate the training set. Their individual speech patterns are over-represented. The model may perform better on statements stylistically similar to prolific speakers and worse on unfamiliar speakers.

**4. Topic bias:**  
LIAR covers predominantly US domestic policy, health, economy, and elections. Claims about science, culture, international affairs, or local governance are sparsely represented, leading to higher error rates in those domains.

### 11.3.2 Fairness Considerations

A truly fair deployment would require:

- **Demographic parity analysis:** Test whether the model's error rate differs across statements about different racial, gender, or political groups.
- **Counterfactual fairness:** Test whether substituting a proper noun (e.g., "Obama" → "Trump") changes the prediction for otherwise identical statements.
- **Calibration across subgroups:** Ensure the model's confidence scores are equally well-calibrated for different topic areas, not just on aggregate.

> **Current status:** Such a fairness audit has not been performed on this implementation. This is a critical gap that must be addressed before any production deployment that affects individuals.

---

## 11.4 Explainability for Non-Technical Stakeholders

### 11.4.1 The Black-Box Problem

RoBERTa is a large transformer model with ~125 million parameters. Its predictions are not inherently interpretable — a non-technical user cannot look at the model and understand *why* it labelled a statement as FAKE.

This is a critical concern in the context of fact-checking, where the *reasoning* behind a verdict is as important as the verdict itself (unlike, say, a movie recommendation).

### 11.4.2 Explainability Strategies Implemented

| Strategy | Implementation in This System |
|---|---|
| **Confidence scores** | The API and Streamlit app return a confidence percentage alongside the label (e.g., "FAKE — 87% confidence"). This signals uncertainty to users and discourages blind trust. |
| **Probability display** | Both FAKE and REAL probabilities are shown, so users understand the prediction is a distribution, not an absolute verdict. |
| **Agentic reasoning trace** | The agent's "Show intermediate steps" expander in the Streamlit app reveals the exact Wikipedia evidence retrieved and the Gemini LLM's natural-language reasoning. This transforms the system from a black box into an auditable reasoning chain. |

### 11.4.3 Recommended Future Explainability Tools

For a production deployment, the following should be added:

- **LIME (Local Interpretable Model-agnostic Explanations):** Highlights which specific words most influenced the prediction.
- **Attention visualisation:** Visualises which tokens the RoBERTa model attended to most strongly.
- **Plain-language verdicts:** The Gemini agent already produces natural-language summaries. These should always be surfaced to users rather than only the raw label.

### 11.4.4 Communication to Non-Technical Users

The following disclaimer must accompany every prediction shown to a non-technical user:

> *"This prediction is generated by an AI model trained on a limited dataset of US political statements from 2007–2016. It should be used as a starting point for further investigation, not as a definitive verdict. Always consult primary sources and professional fact-checkers for important claims."*

---

## 11.5 Potential Misuse

Fake news detection systems carry significant dual-use risks:

| Misuse Scenario | Description | Risk Level |
|---|---|---|
| **Censorship tool** | A government or platform could use the system to suppress political opposition by labelling inconvenient truths as FAKE | Critical |
| **Adversarial content generation** | Bad actors can probe the system to discover what linguistic patterns it considers REAL, then craft false news designed to bypass detection | High |
| **False authority** | Presenting the system's output as ground truth (*"AI confirms: FAKE"*) to manipulate public opinion, without disclosing the model's limitations | High |
| **Selective deployment** | Applying the system only to one political side of a debate, using the tool as a biased moderator rather than a neutral one | High |
| **Surveillance of speech** | Logging all submitted statements for mass surveillance of what claims people are curious about or trying to verify | Medium |

### 11.5.1 Misuse Mitigation Measures in This Implementation

1. **No logging of user inputs** — the API and demo do not record submitted text, preventing surveillance misuse.
2. **Confidence + probabilities displayed always** — makes it harder to present predictions as absolute ground truth.
3. **Open-source and auditable** — the full system code, training data source, and model configurations are publicly available, enabling independent scrutiny.
4. **Agent provides reasoning, not just verdicts** — the agentic component's natural-language reasoning chain makes the decision auditable by any reader.

### 11.5.2 Recommended Governance Measures (Pre-Production)

- Establish a **usage policy** that explicitly prohibits use for political censorship.
- Conduct a **red-team exercise** before deployment to identify adversarial inputs that bypass detection.
- Implement **rate limiting** on the API to prevent automated bulk querying for adversarial content generation.
- Require **human review** for any enforcement action (e.g., content removal) triggered by the model — the model must never be the sole decision-maker.

---

## 11.6 Summary

| Ethical Dimension | Assessment | Status |
|---|---|---|
| **Beneficiaries** | Wide range of public, journalistic, and research use cases | ✅ Documented |
| **Harm potential** | Significant risks to political minorities, non-native speakers, and misrepresented individuals | ⚠️ Risk acknowledged; fairness audit pending |
| **Dataset bias** | Political, temporal, speaker-level, and topic biases present in LIAR | ⚠️ Documented; mitigation is retraining with broader data |
| **Explainability** | Confidence scores + agent reasoning trace provided | ✅ Partially addressed |
| **Misuse** | Dual-use risks identified; technical mitigations in place | ✅ Documented |
| **Governance** | Usage policy and red-teaming recommended before production | ❌ Not yet implemented |
