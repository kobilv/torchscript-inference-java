## DJL TorchScript Inference Demo

Single-module Gradle Java project demonstrating truly device-agnostic TorchScript inference (one `.pt` runs on CPU and any `cuda:N`) via DJL (Deep Java Library) PyTorch engine.

---
### Key Features
- One TorchScript artifact can be loaded on CPU or any available GPU index.
- Automatic or explicit device selection: `--device=auto|cpu|gpu` plus optional `--gpu-index=N`.
- Simple NDList pass-through translator (easy to replace with domain-specific logic).
- Example verified device-agnostic model: `models/simple_tc_movable/simple_tc.pt` exported with `torch.jit.script`.

---
### Repository Layout
```
.
├─ src/
│  └─ main/java/com/example/inference/InferenceApp.java
├─ models/
│  └─ simple_tc_movable/
│     ├─ simple_tc.pt           # TorchScript (scripted) test model – portable across CPU / multi-GPU
│     ├─ tokenizer.json         # (Optional) tokenizer or vocabulary artifacts
│     ├─ vocab.txt              # (Optional) sample vocab
│     └─ labels.txt             # (Optional) class labels for downstream decoding
├─ build.gradle
├─ settings.gradle
└─ README.md
```

---
### Requirements
- Java 17+
- Internet (first run) for DJL engine native downloads
- For GPU inference: CUDA-capable GPU + proper driver + compatible CUDA toolkit (DJL will resolve correct native preset automatically)

---
### Build
```powershell
./gradlew build
```

---
### Running Inference
Use CLI flags to choose model/device. (Defaults are defined in `InferenceApp.Config`).

```powershell
# 1. Auto device: picks first GPU if present, otherwise CPU
./gradlew run --args "--device=auto --model-name=simple_tc_movable"

# 2. Force CPU
./gradlew run --args "--device=cpu --model-name=simple_tc_movable"

# 3. Force specific GPU index (e.g., GPU 1)
./gradlew run --args "--device=gpu --gpu-index=1 --model-name=simple_tc_movable"

# 4. Override model file explicitly
./gradlew run --args "--model-name=simple_tc_movable --model-file=simple_tc.pt"

# Convenience Gradle tasks (GPU / CPU)
./gradlew runCpu --args "--model-name=simple_tc_movable"
./gradlew runGpu --args "--model-name=simple_tc_movable"
```

---
### Exporting A Device-Agnostic TorchScript Model
To ensure a single `.pt` works across CPU and any GPU index:
1. Prefer `torch.jit.script` over `torch.jit.trace` (tracing can bake in CPU tensors).
2. Create auxiliary tensors (masks, position ids, segment ids) using `input_ids.new_*` or by moving them to `input_ids.device`.
3. Register static positional buffers with `register_buffer` (if needed) instead of hardcoding CPU.
4. Avoid Python constructs unsupported by TorchScript (complex list comprehensions with conditionals, `**kwargs` expansion).
5. Validate portability:
	```python
	for i in range(torch.cuda.device_count()):
		 m = torch.jit.load("model.pt", map_location=f"cuda:{i}").eval()
		 x = torch.randint(0, vocab_size, (1, seq_len), device=f"cuda:{i}")
		 _ = m(x, x.new_ones(x.shape), x.new_zeros(x.shape))
	```

---
### Adding Your Own Model
1. Place `your_model.pt` under `models/your_model/`
2. Run:
	```powershell
	./gradlew run --args "--device=auto --model-name=your_model --model-file=your_model.pt"
	```
3. Adjust sequence length via `--seq-len=N` (default 16) to match your model’s expected input shape.

---
### Translators
Currently an identity translator passes raw NDList to/from the predictor. For real use cases (text, vision, audio), implement a `Translator` that:
- Builds input tensors from domain objects in `processInput`
- Decodes output tensors in `processOutput`

---
### Troubleshooting
| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| Mixed CPU/GPU tensor error in attention mask | Traced model baked CPU constants | Re-export with pure scripting and input-device tensor creation |
| Model loads on CPU when expecting GPU | `--device=auto` but no GPU visible | Force with `--device=gpu --gpu-index=0` and verify CUDA install |
| Slow first inference | Native libs downloading / graph optimizer warmup | Re-run; add caching layer |

---
### Roadmap / Ideas
- Add tokenizer-driven translator for `simple_tc_movable`.
- Example DistilBERT scripted NER export to contrast with non-scriptable large models.
- Unit tests that spin up CPU and (mock) GPU device selection logic.

---
### License
Add your preferred license information here.

---
### Acknowledgements
- DJL: https://djl.ai
- PyTorch & TorchScript
- (Optional) Hugging Face Transformers for architecture references

