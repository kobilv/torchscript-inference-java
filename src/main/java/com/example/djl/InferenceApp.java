package com.example.djl;

import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Minimal demo that loads a TorchScript model and runs inference on CPU or GPU
 * using the same .pt file. Device can be provided via args[0]: cpu | gpu:0 | auto.
 */
public class InferenceApp {

    public static void main(String[] args) throws Exception {
        String deviceArg = args.length > 0 ? args[0] : "auto"; // cpu | gpu[:index] | auto
        String modelDir = args.length > 1 ? args[1] : "models/my_model";
        String modelName = args.length > 2 ? args[2] : "my_model"; // my_model.pt in modelDir

        Device device = selectDevice(deviceArg);
        System.out.println("Using device: " + device);

        Path modelPath = Paths.get(modelDir);
        Path modelFile = modelPath.resolve(modelName + ".pt");

        if (!Files.isDirectory(modelPath) || !Files.exists(modelFile)) {
            System.err.println("Model not found. Expected: " + modelFile.toAbsolutePath());
            System.err.println("Place your TorchScript file and artifacts in '" + modelPath + "' (e.g., '" + modelName + ".pt').");
            System.err.println("Usage: <cpu|gpu[:index]|auto> <model_dir> <model_name_without_ext>");
            System.exit(2);
        }

        // Build a generic translator for NDArray in/out for flexibility.
        Translator<NDArray, NDArray> translator = new Translator<>() {
            @Override
            public NDList processInput(TranslatorContext ctx, NDArray input) {
                // Wrap input into NDList as required by DJL translator API
                return new NDList(input);
            }

            @Override
            public NDArray processOutput(TranslatorContext ctx, NDList list) {
                return list.singletonOrThrow();
            }

            @Override
            public Batchifier getBatchifier() {
                return Batchifier.STACK;
            }
        };

        Criteria<NDArray, NDArray> criteria = Criteria.builder()
                .setTypes(NDArray.class, NDArray.class)
                .optDevice(device)
                .optModelPath(modelPath)
                .optModelName(modelName)
                .optEngine("PyTorch")
                .optTranslator(translator)
                .build();

    try (ZooModel<NDArray, NDArray> model = ModelZoo.loadModel(criteria);
             Predictor<NDArray, NDArray> predictor = model.newPredictor()) {

            // Create a tiny dummy input tensor as a placeholder demo input.
            try (NDManager manager = NDManager.newBaseManager(device)) {
                NDArray input = manager.linspace(0, 1, 16).reshape(new Shape(1, 16));
                NDArray output = predictor.predict(input);
                System.out.println("Output shape: " + output.getShape());
                long n = Math.min(output.size(), 10);
                System.out.println("Output head (" + n + "): " + output.reshape(-1).get("0:" + n));
            }
        }
    }

    private static Device selectDevice(String arg) {
        if (arg == null || arg.equalsIgnoreCase("auto")) {
            // Prefer GPU if available, else CPU
            int gpus = Engine.getInstance().getGpuCount();
            if (gpus > 0) {
                return Device.gpu();
            }
            return Device.cpu();
        }
        if (arg.equalsIgnoreCase("cpu")) {
            return Device.cpu();
        }
        if (arg.toLowerCase().startsWith("gpu")) {
            String[] parts = arg.split(":");
            int index = parts.length > 1 ? Integer.parseInt(parts[1]) : 0;
            return Device.gpu(index);
        }
        return Device.cpu();
    }
}
