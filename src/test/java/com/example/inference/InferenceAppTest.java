package com.example.inference;

import ai.djl.Device;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class InferenceAppTest {

    @Test
    void selectDevice_respectsCpuArg() {
        InferenceApp.Config cfg = InferenceApp.Config.fromArgs(new String[]{"cpu"});
        Device d = InferenceApp.selectDevice(cfg);
        assertEquals(Device.cpu(), d, "Expected CPU device");
    }

    @Test
    void selectDevice_gpuIndexParsing() {
        InferenceApp.Config cfg = InferenceApp.Config.fromArgs(new String[]{"gpu:1"});
        Device d = InferenceApp.selectDevice(cfg);
        // We can't guarantee GPU availability in CI, but Device.gpu(1) creation doesn't require a physical GPU
        assertEquals("gpu(1)", d.toString());
    }
}
