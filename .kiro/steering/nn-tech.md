# SKaiNET Neural Network DSL Guide

This document provides technical instructions for AI agents and developers on using SKaiNET's Neural Network DSL for creating and managing models.

## 1. Core Architecture: Model vs. Module

SKaiNET distinguishes between a **Model** and a **Module**:

*   **`Module<T, V>`**: The actual implementation of a neural network layer or a whole network. It contains:
    *   **Parameters**: Weights and biases (via `ModuleNode` interface).
    *   **Sub-modules**: A list of child modules for hierarchical structures.
    *   **`forward()` method**: The execution logic that takes an input `Tensor` and returns an output `Tensor`.
*   **`Model<T, V, I, O>`**: A high-level descriptor and life-cycle manager for a neural network. It provides:
    *   **`create(executionContext)`**: A factory method to instantiate the `Module` (allocating weights).
    *   **`calculate(...)`**: Logic to perform inference, often wrapping the `Module.forward()` with progress reporting.
    *   **`fit(...)`**: A method returning a `Flow` of `ModelExecutionResult` for training or long-running processing.
    *   **`modelCard()`**: Metadata about the model (license, dataset, intended use).

**AI Agent Tip**: When requested to "create a model", you usually define a class implementing `Model` and use the DSL inside its `create` method to build the `Module` hierarchy.

## 2. Model Creation DSL

SKaiNET provides a type-safe DSL to define network architectures concisely.

### The Entry Point: `definition` and `network`

To start building a network, use the `definition` block, followed by the `network` block which requires an `ExecutionContext`.

```kotlin
val myModule: Module<FP32, Float> = definition {
    network<FP32, Float>(executionContext) {
        // DSL starts here
    }
}
```

### Building with `sequential` and `stage`

*   **`sequential`**: Defines a linear stack of layers.
*   **`stage`**: Groups layers into named blocks for better organization and reflection.

```kotlin
network<FP32, Float>(ctx) {
    sequential {
        stage("conv_block") {
            conv2d(outChannels = 16, kernelSize = 3 to 3)
            activation { it.relu() }
        }
        stage("fc") {
            flatten()
            dense(outputDimension = 10)
            activation { it.softmax(dim = 1) }
        }
    }
}
```

## 3. Type Safety and Context

### Type-Safe Generics
SKaiNET uses Kotlin generics `<T : DType, V>` to ensure type consistency across the entire network. `T` represents the data type (e.g., `FP32`, `INT32`), and `V` represents the underlying value type (e.g., `Float`, `Int`).

### The Role of `ExecutionContext`
The `ExecutionContext` is mandatory for model creation and execution. It provides:
*   The execution backend (CPU, GPU, etc.) through `TensorOps`.
*   Memory management via `TensorDataFactory`.
*   Execution phase (EVAL vs. TRAIN).

**Proper Use**: Always propagate the `ExecutionContext` to your modules and operations. Never create tensors or modules without a context.

## 4. Introspection and Reflections: `describe()`

To inspect a model's structure, parameters, and shape propagation, use the **`reflection`** capabilities. The most important tool is the **`describe()`** extension function.

```kotlin
// Example of using reflections
val model: Module<FP32, Float> = loadModel() 
val inputShape = Shape(1, 28, 28)

// Returns a human-readable table with layers, output shapes, and parameter counts
val summary: String = model.describe(inputShape, ctx, FP32::class)
println(summary)
```

### Programmable Reflection: `ModuleNode`
All `Module` objects implement `ModuleNode`, allowing you to programmatically traverse the hierarchy:
*   `module.children`: Access sub-modules.
*   `module.params`: Access weights and biases.
*   `module.name` / `module.id`: Identification.

## 5. Comprehensive Use Case Sample

Below is a complete example of an MNIST CNN defined using the SKaiNET DSL.

```kotlin
class MnistCnn : Model<FP32, Float, Tensor<FP32, Float>, Tensor<FP32, Float>> {

    override fun create(ctx: ExecutionContext): Module<FP32, Float> = definition {
        network(ctx) {
            sequential<FP32, Float> {
                stage("conv1") {
                    conv2d(outChannels = 16, kernelSize = 5 to 5, padding = 2 to 2)
                    activation { it.relu() }
                    maxPool2d(kernelSize = 2 to 2, stride = 2 to 2)
                }
                stage("conv2") {
                    conv2d(outChannels = 32, kernelSize = 5 to 5, padding = 2 to 2)
                    activation { it.relu() }
                    maxPool2d(kernelSize = 2 to 2, stride = 2 to 2)
                }
                stage("output") {
                    flatten()
                    dense(outputDimension = 10)
                    activation { it.softmax(dim = 1) }
                }
            }
        }
    }

    override suspend fun calculate(
        module: Module<FP32, Float>,
        inputValue: Tensor<FP32, Float>,
        ctx: ExecutionContext,
        reportProgress: suspend (current: Int, total: Int, message: String?) -> Unit
    ): Tensor<FP32, Float> {
        return module.forward(inputValue, ctx)
    }

    override fun modelCard() = ModelCard(libraryName = "skainet", pipelineTag = "image-classification")
}
```
