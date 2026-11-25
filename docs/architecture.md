# Local pipeline decisions

The integrator froze the Python and wire interfaces after the independent architecture review. [Contracts](contracts.md)

**One audio model lifetime.** A dedicated Python child decodes and normalizes to disk, runs stateful Silero and loads Hojo for detected speech. Process exit releases the weights and confines the pinned Hojo package's global initialization hooks to that interpreter. The parent retains ordered typed results and invokes a numeric-loopback Ollama endpoint. HF resolution is disabled for inference. Preparation verifies the full merged weight SHA-256 and the tokenizer, config and frontend files against the pinned Hub revision.

**Offsets and evidence.** The normalized 16 kHz timeline includes all silence. A disjoint core identifies each chunk, and its potentially overlapping padded range identifies the samples passed to Hojo. Duration bounds apply to the padded range. The original file hash survives transcoding. The public contract keeps the returned model dictionary, derived text, clip-level sample bounds, and exact character evidence separately.

**Validation and history.** Inventory candidates carry lexical evidence through deterministic French number, unit, packaging and revision validation. Numeric values are serialized as decimal strings. Correction events keep the superseded observations available. SQLite transactions and constraints determine committed state before JSON, CSV and text views are generated. The processor fingerprint includes the source code digest, configured behavior, relevant dependency versions, schema version, pinned asset IDs, actual ASR backend and dtype. Distinct source coordinates retain distinct utterances.

**Mac backend gate.** The bundle requests a 30-second frontend. Wheel 0.1.3 constructs the frontend with 40 seconds and requests CUDA autocast on any non-CPU device. The conservatively capped application passes at most 400,000 samples, including padding and overlap. A scoped Hojo subclass bypasses the invalid autocast branch and casts only speech embeddings at the mixed-dtype decoder boundary. Source inspection supports this design. Runtime status and measurements are recorded separately in the acceptance report.

**Offline evidence.** The acceptance driver runs a dedicated Ollama process and installed CLI under the same process-local OS network policy. Each process must show a successful loopback request and a permission-denied external socket under the sandbox. Each run has a fresh output directory. The positive external control runs before entering the sandbox, and the local service uses a separate port. File and microphone runs produce separate records.

**Evaluation data.** Two preselected original French FLEURS validation recordings carry human-checked references. Their provenance, selection rule and hashes accompany the manifest. Synthetic French inventory prompts serve as functional fixtures. Metrics for the two sources are reported separately.
