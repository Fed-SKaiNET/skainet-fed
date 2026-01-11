package sk.ainet.fed.core.data

import sk.ainet.fed.core.types.DType
import sk.ainet.fed.core.types.Shape
import kotlin.reflect.KClass

/**
 * Model architecture definition for federated learning strategies.
 * Defines the structure and data types of model parameters.
 */
public data class ModelShape(
    val layers: List<LayerShape>
)

/**
 * Definition of a single layer in the model architecture.
 * 
 * @param name Unique identifier for the layer (e.g., "dense1.weight", "conv2.bias")
 * @param shape Tensor shape for this layer's parameters
 * @param dtype Data type class for the tensor (e.g., FP32::class)
 */
public data class LayerShape(
    val name: String,
    val shape: Shape,
    val dtype: KClass<out DType>
)