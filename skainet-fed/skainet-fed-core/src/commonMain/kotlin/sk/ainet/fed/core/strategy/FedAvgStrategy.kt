package sk.ainet.fed.core.strategy

import sk.ainet.fed.core.types.*
import sk.ainet.fed.core.data.*
import sk.ainet.fed.core.management.ParameterManager
import sk.ainet.fed.core.utils.FederatedMathUtils

/**
 * Federated Averaging (FedAvg) strategy implementation.
 * 
 * This class implements the foundational FedAvg algorithm as described in the original
 * "Communication-Efficient Learning of Deep Networks from Decentralized Data" paper.
 * FedAvg performs weighted averaging of client model updates, where weights are
 * proportional to the number of training samples each client used.
 * 
 * Key characteristics:
 * - Simple weighted averaging based on client sample sizes
 * - No additional strategy-specific data required
 * - Serves as the baseline for more advanced federated learning strategies
 * - Mathematically equivalent to centralized SGD under IID data distribution
 * 
 * Mathematical formula:
 * w_{t+1} = Σ(n_k / n) * w_k^{t+1}
 * where:
 * - w_{t+1} is the new global model
 * - n_k is the number of samples for client k
 * - n is the total number of samples across all clients
 * - w_k^{t+1} is the updated model from client k
 * 
 * @param parameterManager Manages global parameters and state across FL rounds
 */
public class FedAvgStrategy(
    private val parameterManager: ParameterManager
) : FederatedStrategy {

    /**
     * Initialize global model parameters from a SKaiNET Model instance.
     * 
     * This method creates the initial state of the federated model by instantiating
     * the provided SKaiNET Model and extracting its parameters through the ParameterManager.
     * For FedAvg, no additional strategy-specific initialization is required.
     * 
     * @param ctx SKaiNET execution context for tensor operations and model instantiation
     * @param model SKaiNET Model instance defining the neural network architecture
     * @return Initial global parameters extracted from the instantiated model
     */
    override suspend fun initializeGlobalParameters(
        ctx: ExecutionContext,
        model: Model<FP32, Float, *, *>
    ): GlobalParameters {
        return parameterManager.initializeParameters(model)
    }

    /**
     * Prepare mathematical state for client update.
     * 
     * For FedAvg, clients only need the current global weights to perform local training.
     * No additional strategy-specific data is required, making this the simplest
     * federated learning strategy in terms of communication overhead.
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param round Current federated learning round number
     * @param globalParameters Current global model parameters
     * @return Strategy update containing global weights for client training
     */
    override suspend fun prepareClientUpdate(
        ctx: ExecutionContext,
        round: Int,
        globalParameters: GlobalParameters
    ): StrategyUpdate {
        return StrategyUpdate(
            globalWeights = globalParameters.weights,
            strategySpecificData = emptyMap(), // FedAvg requires no additional data
            round = round
        )
    }

    /**
     * Aggregate client updates using weighted averaging.
     * 
     * This is the core FedAvg operation: compute a weighted average of client model updates
     * where each client's contribution is proportional to their number of training samples.
     * This ensures that clients with more data have proportionally more influence on the
     * global model update.
     * 
     * The aggregation is performed parameter-wise, meaning each parameter tensor
     * (weights, biases) is aggregated separately using the same weighting scheme.
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param round Current federated learning round number
     * @param currentGlobalParameters Current global model parameters
     * @param clientResults List of client training results to aggregate
     * @return Updated global parameters after weighted averaging
     */
    override suspend fun aggregateClientUpdates(
        ctx: ExecutionContext,
        round: Int,
        currentGlobalParameters: GlobalParameters,
        clientResults: List<ClientResult>
    ): GlobalParameters {
        require(clientResults.isNotEmpty()) { "Cannot aggregate empty client results" }
        
        // Extract sample counts for weighting
        val sampleCounts = clientResults.map { it.sampleCount.toFloat() }
        require(sampleCounts.all { it > 0f }) { "All client sample counts must be positive" }
        
        // Get all parameter names from the first client (assuming all clients have same architecture)
        val parameterNames = clientResults.first().weightUpdates.keys
        
        // Aggregate each parameter separately using weighted averaging
        val aggregatedWeights = mutableMapOf<String, Tensor<FP32, Float>>()
        
        for (paramName in parameterNames) {
            // Extract this parameter from all clients
            val clientParameters = clientResults.map { clientResult ->
                clientResult.weightUpdates[paramName]
                    ?: throw IllegalArgumentException("Client ${clientResult.clientId} missing parameter: $paramName")
            }
            
            // Compute weighted average for this parameter
            val aggregatedParam = FederatedMathUtils.weightedAverage(
                ctx = ctx,
                clientUpdates = clientParameters,
                weights = sampleCounts
            )
            
            aggregatedWeights[paramName] = aggregatedParam
        }
        
        // Update global state through ParameterManager
        return parameterManager.updateGlobalState(aggregatedWeights, round)
    }

    /**
     * Aggregate evaluation metrics from clients.
     * 
     * This method combines evaluation results from multiple clients to provide
     * a comprehensive assessment of the global model performance. Metrics are
     * weighted by the number of evaluation samples each client used.
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param round Current federated learning round number
     * @param globalParameters Current global model parameters
     * @param evaluationResults List of client evaluation results
     * @return Aggregated metrics summarizing global model performance
     */
    override suspend fun evaluateGlobalModel(
        ctx: ExecutionContext,
        round: Int,
        globalParameters: GlobalParameters,
        evaluationResults: List<EvaluationResult>
    ): AggregatedMetrics {
        require(evaluationResults.isNotEmpty()) { "Cannot evaluate with empty results" }
        
        val totalSamples = evaluationResults.sumOf { it.sampleCount }
        require(totalSamples > 0) { "Total evaluation samples must be positive" }
        
        // Compute weighted mean accuracy and loss
        var weightedAccuracySum = 0f
        var weightedLossSum = 0f
        
        for (result in evaluationResults) {
            val weight = result.sampleCount.toFloat() / totalSamples.toFloat()
            weightedAccuracySum += result.accuracy * weight
            weightedLossSum += result.loss * weight
        }
        
        // Aggregate additional metrics if present
        val additionalMetrics = mutableMapOf<String, Float>()
        val allMetricNames = evaluationResults.flatMap { it.additionalMetrics.keys }.toSet()
        
        for (metricName in allMetricNames) {
            var weightedSum = 0f
            var totalWeight = 0f
            
            for (result in evaluationResults) {
                result.additionalMetrics[metricName]?.let { metricValue ->
                    val weight = result.sampleCount.toFloat() / totalSamples.toFloat()
                    weightedSum += metricValue * weight
                    totalWeight += weight
                }
            }
            
            if (totalWeight > 0f) {
                additionalMetrics[metricName] = weightedSum
            }
        }
        
        return AggregatedMetrics(
            meanAccuracy = weightedAccuracySum,
            meanLoss = weightedLossSum,
            totalSamples = totalSamples,
            clientCount = evaluationResults.size,
            additionalMetrics = additionalMetrics
        )
    }
}