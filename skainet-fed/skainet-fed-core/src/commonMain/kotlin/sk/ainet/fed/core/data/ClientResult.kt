package sk.ainet.fed.core.data

import sk.ainet.fed.core.types.Tensor
import sk.ainet.fed.core.types.FP32

/**
 * Client training results with tensor updates.
 * Contains the results sent from clients back to the server after local training.
 * 
 * @param clientId Unique identifier for the client
 * @param weightUpdates Model weight updates computed by the client
 * @param sampleCount Number of training samples used by the client
 * @param trainingLoss Final training loss achieved by the client
 * @param strategySpecificData Additional tensors required by specific FL strategies
 */
public data class ClientResult(
    val clientId: String,
    val weightUpdates: Map<String, Tensor<FP32, Float>>,
    val sampleCount: Int,
    val trainingLoss: Float,
    val strategySpecificData: Map<String, Tensor<FP32, Float>> = emptyMap()
)

/**
 * Client evaluation results for model performance assessment.
 * 
 * @param clientId Unique identifier for the client
 * @param accuracy Evaluation accuracy achieved by the client
 * @param loss Evaluation loss computed by the client
 * @param sampleCount Number of evaluation samples used
 * @param additionalMetrics Additional evaluation metrics (e.g., precision, recall)
 */
public data class EvaluationResult(
    val clientId: String,
    val accuracy: Float,
    val loss: Float,
    val sampleCount: Int,
    val additionalMetrics: Map<String, Float> = emptyMap()
)

/**
 * Aggregated metrics from multiple client evaluations.
 * 
 * @param meanAccuracy Weighted mean accuracy across all clients
 * @param meanLoss Weighted mean loss across all clients
 * @param totalSamples Total number of evaluation samples
 * @param clientCount Number of participating clients
 * @param additionalMetrics Additional aggregated metrics
 */
public data class AggregatedMetrics(
    val meanAccuracy: Float,
    val meanLoss: Float,
    val totalSamples: Int,
    val clientCount: Int,
    val additionalMetrics: Map<String, Float> = emptyMap()
)