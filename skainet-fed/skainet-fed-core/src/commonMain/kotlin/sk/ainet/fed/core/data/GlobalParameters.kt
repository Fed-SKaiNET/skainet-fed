package sk.ainet.fed.core.data

import sk.ainet.fed.core.types.Tensor
import sk.ainet.fed.core.types.FP32

/**
 * Represents global model parameters as SKaiNET tensors.
 * Contains the current state of the federated model shared across all clients.
 * 
 * @param weights Map of parameter names to their corresponding tensor values
 * @param round Current federated learning round number
 * @param metadata Additional strategy-specific metadata
 */
public data class GlobalParameters(
    val weights: Map<String, Tensor<FP32, Float>>,
    val round: Int,
    val metadata: Map<String, Any> = emptyMap()
)