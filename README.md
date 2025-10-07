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
### Adding Your Own Model
1. Place `your_model.pt` under `models/your_model/`
2. Run:
	```powershell
	./gradlew run --args "--device=auto --model-name=your_model --model-file=your_model.pt"
	```
3. Adjust sequence length via `--seq-len=N` (default 16) to match your model’s expected input shape.

---
### Acknowledgements
- DJL: https://djl.ai
- PyTorch & TorchScript
