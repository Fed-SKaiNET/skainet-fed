package sk.ainet.fed.core.data

import sk.ainet.fed.core.types.Tensor
import sk.ainet.fed.core.types.FP32

/**
 * Mathematical state prepared for client updates.
 * Contains the information sent from server to clients for local training.
 * 
 * @param globalWeights Current global model weights to be used by clients
 * @param strategySpecificData Additional tensors required by specific FL strategies (e.g., control variates for SCAFFOLD)
 * @param round Current federated learning round number
 */
public data class StrategyUpdate(
    val globalWeights: Map<String, Tensor<FP32, Float>>,
    val strategySpecificData: Map<String, Tensor<FP32, Float>> = emptyMap(),
    val round: Int
)