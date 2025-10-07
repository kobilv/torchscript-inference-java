package com.example.inference;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

import java.nio.file.Path;
import java.nio.file.Paths;

public class InferenceApp {

    public static void main(String[] args) throws Exception {
        Config cfg = Config.fromArgs(args);
        Device device = selectDevice(cfg);

        Path modelDir = Paths.get(cfg.modelDir);
        Path modelPath = modelDir.resolve(cfg.modelFile);
        if (!modelPath.toFile().exists()) {
            System.err.println("Model file not found: " + modelPath.toAbsolutePath());
            System.err.println("Expected structure: models/" + cfg.modelName + "/" + cfg.modelFile);
            System.err.println("Override with --model-name=NAME --model-file=FILE or --model-dir=DIR");
            System.exit(2);
        }

        System.out.printf("Loading model %s on device: %s%n", modelPath.toAbsolutePath(), device);

    var criteria = Criteria.builder()
        .setTypes(NDList.class, NDList.class)
                .optModelPath(modelPath)
                .optDevice(device)
                .optEngine("PyTorch")
        // Ensure weights/constants are remapped to the selected device on load
        .optOption("mapLocation", "true")
        .optTranslator(new IdentityNDListTranslator())
                .build();

       try (ZooModel<NDList, NDList> model = criteria.loadModel();
           Predictor<NDList, NDList> predictor = model.newPredictor();
           NDManager manager = NDManager.newBaseManager(device)) {

            // Build dummy transformers-style inputs expected by the model:
            int seqLen = cfg.seqLen; // adjustable via --seq-len
            Shape shape = new Shape(1, seqLen);
            NDArray inputIds = manager.zeros(shape, ai.djl.ndarray.types.DataType.INT64);
            NDArray attentionMask = manager.ones(shape, ai.djl.ndarray.types.DataType.INT64);
            NDArray tokenTypeIds = manager.zeros(shape, ai.djl.ndarray.types.DataType.INT64);

            NDList inputs = new NDList(inputIds, attentionMask, tokenTypeIds);
            NDList outputs = predictor.predict(inputs);
            System.out.printf("Inference success on %s. Output list size: %d; first output shape: %s%n",
                    device, outputs.size(), (outputs.isEmpty() ? "-" : outputs.get(0).getShape()));
        }
    }


    static Device selectDevice(Config cfg) {
        String dev = cfg.device.toLowerCase();
        if (dev.equals("cpu")) {
            return Device.cpu();
        }
        if (dev.equals("gpu") || dev.equals("cuda")) {
            return Device.gpu(Math.max(0, cfg.gpuIndex));
        }
        // auto
        int gpus = 0;
        try {
            gpus = Engine.getInstance().getGpuCount();
        } catch (Throwable ignore) {
            // engine may not be initialized yet; treat as 0
        }
        if (gpus > 0) {
            int idx = Math.max(0, Math.min(cfg.gpuIndex, gpus - 1));
            return Device.gpu(idx);
        }
        return Device.cpu();
    }

    static class IdentityNDListTranslator implements NoBatchifyTranslator<NDList, NDList> {
        @Override
        public NDList processInput(TranslatorContext ctx, NDList input) {
            return input;
        }

        @Override
        public NDList processOutput(TranslatorContext ctx, NDList list) {
            return list;
        }
    }

    static class Config {
        String device = "auto"; // cpu|gpu|auto
        int gpuIndex = 0;        // which GPU when device=gpu or auto
        String modelName = "OpenMed-NER-ChemicalDetect-ModernMed-149M";
        String modelDir = Paths.get("models", modelName).toString();
        String modelFile = modelName + ".pt";
        boolean modelFileOverridden = false; // true if set via --model-file or -DmodelFile
        int seqLen = Integer.getInteger("seqLen", 16);

        static Config fromArgs(String[] args) {
            Config c = new Config();
            // Allow system property override: -DmodelFile=... (preferred)
            String sysModelFile = System.getProperty("modelFile");
            if (sysModelFile != null && !sysModelFile.isBlank()) {
                c.modelFile = sysModelFile;
                c.modelFileOverridden = true;
            }
            if (args == null) return c;
            for (String a : args) {
                if (a.startsWith("--device=")) {
                    c.device = a.substring("--device=".length());
                } else if (a.startsWith("--gpu-index=")) {
                    try { c.gpuIndex = Integer.parseInt(a.substring("--gpu-index=".length())); } catch (NumberFormatException ignore) {}
                } else if (a.startsWith("--model-name=")) {
                    c.modelName = a.substring("--model-name=".length());
                    c.modelDir = Paths.get("models", c.modelName).toString();
                    // If modelFile was not explicitly overridden, follow modelName
                    if (!c.modelFileOverridden) {
                        c.modelFile = c.modelName + ".pt";
                    }
                } else if (a.startsWith("--model-dir=")) {
                    c.modelDir = a.substring("--model-dir=".length());
                } else if (a.startsWith("--model-file=")) {
                    c.modelFile = a.substring("--model-file=".length());
                    c.modelFileOverridden = true;
                } else if (a.startsWith("--seq-len=")) {
                    try { c.seqLen = Integer.parseInt(a.substring("--seq-len=".length())); } catch (NumberFormatException ignore) {}
                } else if (a.equalsIgnoreCase("cpu") || a.equalsIgnoreCase("gpu") || a.equalsIgnoreCase("auto") || a.toLowerCase().startsWith("gpu:")) {
                    // positional shorthand: cpu|gpu|auto or gpu:N
                    if (a.toLowerCase().startsWith("gpu:")) {
                        c.device = "gpu";
                        String idxStr = a.substring("gpu:".length());
                        try { c.gpuIndex = Integer.parseInt(idxStr); } catch (NumberFormatException ignore) {}
                    } else {
                        c.device = a;
                    }
                }
            }
            return c;
        }
    }
}
