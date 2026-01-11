package sk.ainet.fed.core.utils

import sk.ainet.fed.core.data.EvaluationResult
import sk.ainet.fed.core.data.AggregatedMetrics
import sk.ainet.fed.core.types.ExecutionContext
import kotlin.math.sqrt

/**
 * Utility class for aggregating evaluation metrics from multiple federated learning clients.
 * 
 * The MetricAggregator provides comprehensive statistical analysis of client evaluation results,
 * including weighted aggregation, statistical summaries, and robust handling of missing or
 * invalid client data. All computations are designed to be mathematically sound and handle
 * edge cases gracefully.
 * 
 * Key features:
 * - Weighted aggregation of loss and accuracy metrics
 * - Statistical summaries including mean, variance, and confidence intervals
 * - Robust handling of missing, invalid, or corrupted client results
 * - Strategy-specific metric computation (e.g., consensus magnitude)
 * - Cross-platform consistency through mathematical precision
 * 
 * Requirements addressed:
 * - 7.1: Aggregate loss values across multiple clients
 * - 7.2: Compute accuracy metrics from client evaluation results
 * - 7.3: Calculate strategy-specific metrics including consensus magnitude
 * - 7.4: Handle missing or invalid client results gracefully
 * - 7.5: Provide statistical summaries including mean, variance, and confidence intervals
 */
public class MetricAggregator(private val ctx: ExecutionContext) {
    
    /**
     * Aggregate evaluation metrics from multiple client results.
     * 
     * This is the primary method for combining client evaluation results into
     * a comprehensive statistical summary. The aggregation uses sample-weighted
     * averaging to ensure clients with more evaluation data have proportionally
     * more influence on the final metrics.
     * 
     * @param evaluationResults List of client evaluation results
     * @return Aggregated metrics with statistical summaries
     * @throws IllegalArgumentException if no valid results are provided
     */
    public suspend fun aggregateMetrics(
        evaluationResults: List<EvaluationResult>
    ): AggregatedMetrics {
        require(evaluationResults.isNotEmpty()) { "Evaluation results cannot be empty" }
        
        // Filter out invalid results
        val validResults = filterValidResults(evaluationResults)
        require(validResults.isNotEmpty()) { "No valid evaluation results found" }
        
        // Compute weighted aggregates
        val totalSamples = validResults.sumOf { it.sampleCount }
        val weightedAccuracy = computeWeightedMean(validResults) { it.accuracy }
        val weightedLoss = computeWeightedMean(validResults) { it.loss }
        
        // Compute additional metrics
        val additionalMetrics = mutableMapOf<String, Float>()
        
        // Add statistical summaries
        additionalMetrics.putAll(computeStatisticalSummary(validResults))
        
        // Add strategy-specific metrics
        additionalMetrics.putAll(computeStrategySpecificMetrics(validResults))
        
        return AggregatedMetrics(
            meanAccuracy = weightedAccuracy,
            meanLoss = weightedLoss,
            totalSamples = totalSamples,
            clientCount = validResults.size,
            additionalMetrics = additionalMetrics
        )
    }
    
    /**
     * Aggregate loss values across multiple clients with weighted averaging.
     * 
     * Computes the sample-weighted mean loss across all participating clients.
     * Clients with more evaluation samples contribute proportionally more to
     * the final aggregated loss value.
     * 
     * @param evaluationResults List of client evaluation results
     * @return Weighted mean loss value
     */
    public suspend fun aggregateLoss(
        evaluationResults: List<EvaluationResult>
    ): Float {
        val validResults = filterValidResults(evaluationResults)
        require(validResults.isNotEmpty()) { "No valid evaluation results for loss aggregation" }
        
        return computeWeightedMean(validResults) { it.loss }
    }
    
    /**
     * Aggregate accuracy metrics from client evaluation results.
     * 
     * Computes the sample-weighted mean accuracy across all participating clients.
     * This provides a global view of model performance across the federated system.
     * 
     * @param evaluationResults List of client evaluation results
     * @return Weighted mean accuracy value
     */
    public suspend fun aggregateAccuracy(
        evaluationResults: List<EvaluationResult>
    ): Float {
        val validResults = filterValidResults(evaluationResults)
        require(validResults.isNotEmpty()) { "No valid evaluation results for accuracy aggregation" }
        
        return computeWeightedMean(validResults) { it.accuracy }
    }
    
    /**
     * Compute statistical summary including mean, variance, and confidence intervals.
     * 
     * Provides comprehensive statistical analysis of client metrics including:
     * - Unweighted mean and variance for accuracy and loss
     * - Standard deviation and coefficient of variation
     * - 95% confidence intervals for mean estimates
     * - Min/max values across clients
     * 
     * @param evaluationResults List of valid client evaluation results
     * @return Map of statistical summary metrics
     */
    public suspend fun computeStatisticalSummary(
        evaluationResults: List<EvaluationResult>
    ): Map<String, Float> {
        require(evaluationResults.isNotEmpty()) { "Cannot compute statistics on empty results" }
        
        val accuracies = evaluationResults.map { it.accuracy }
        val losses = evaluationResults.map { it.loss }
        
        val summaryMetrics = mutableMapOf<String, Float>()
        
        // Accuracy statistics
        summaryMetrics["accuracy_mean"] = accuracies.average().toFloat()
        summaryMetrics["accuracy_variance"] = computeVariance(accuracies)
        summaryMetrics["accuracy_std"] = sqrt(summaryMetrics["accuracy_variance"]!!)
        summaryMetrics["accuracy_min"] = accuracies.minOrNull() ?: 0f
        summaryMetrics["accuracy_max"] = accuracies.maxOrNull() ?: 0f
        
        // Loss statistics
        summaryMetrics["loss_mean"] = losses.average().toFloat()
        summaryMetrics["loss_variance"] = computeVariance(losses)
        summaryMetrics["loss_std"] = sqrt(summaryMetrics["loss_variance"]!!)
        summaryMetrics["loss_min"] = losses.minOrNull() ?: 0f
        summaryMetrics["loss_max"] = losses.maxOrNull() ?: 0f
        
        // Confidence intervals (95%)
        val n = evaluationResults.size
        if (n > 1) {
            val tValue = 1.96f // Approximate t-value for 95% CI with large n
            
            val accuracySE = summaryMetrics["accuracy_std"]!! / sqrt(n.toFloat())
            summaryMetrics["accuracy_ci_lower"] = summaryMetrics["accuracy_mean"]!! - tValue * accuracySE
            summaryMetrics["accuracy_ci_upper"] = summaryMetrics["accuracy_mean"]!! + tValue * accuracySE
            
            val lossSE = summaryMetrics["loss_std"]!! / sqrt(n.toFloat())
            summaryMetrics["loss_ci_lower"] = summaryMetrics["loss_mean"]!! - tValue * lossSE
            summaryMetrics["loss_ci_upper"] = summaryMetrics["loss_mean"]!! + tValue * lossSE
        }
        
        // Coefficient of variation (relative variability)
        if (summaryMetrics["accuracy_mean"]!! > 0f) {
            summaryMetrics["accuracy_cv"] = summaryMetrics["accuracy_std"]!! / summaryMetrics["accuracy_mean"]!!
        }
        if (summaryMetrics["loss_mean"]!! > 0f) {
            summaryMetrics["loss_cv"] = summaryMetrics["loss_std"]!! / summaryMetrics["loss_mean"]!!
        }
        
        return summaryMetrics
    }
    
    /**
     * Calculate strategy-specific metrics including consensus magnitude.
     * 
     * Computes metrics that are specific to federated learning strategies:
     * - Consensus magnitude: Measures agreement between client results
     * - Performance spread: Range of performance across clients
     * - Participation rate: Fraction of expected clients that participated
     * 
     * @param evaluationResults List of valid client evaluation results
     * @return Map of strategy-specific metrics
     */
    public suspend fun computeStrategySpecificMetrics(
        evaluationResults: List<EvaluationResult>
    ): Map<String, Float> {
        require(evaluationResults.isNotEmpty()) { "Cannot compute strategy metrics on empty results" }
        
        val strategyMetrics = mutableMapOf<String, Float>()
        
        // Consensus magnitude - measures how much clients agree
        // Computed as 1 - (coefficient of variation), where higher values indicate more consensus
        val accuracies = evaluationResults.map { it.accuracy }
        val losses = evaluationResults.map { it.loss }
        
        val accuracyCV = if (accuracies.average() > 0) {
            sqrt(computeVariance(accuracies)) / accuracies.average().toFloat()
        } else 0f
        
        val lossCV = if (losses.average() > 0) {
            sqrt(computeVariance(losses)) / losses.average().toFloat()
        } else 0f
        
        strategyMetrics["consensus_magnitude_accuracy"] = maxOf(0f, 1f - accuracyCV)
        strategyMetrics["consensus_magnitude_loss"] = maxOf(0f, 1f - lossCV)
        
        // Overall consensus magnitude (average of accuracy and loss consensus)
        strategyMetrics["consensus_magnitude"] = (
            strategyMetrics["consensus_magnitude_accuracy"]!! + 
            strategyMetrics["consensus_magnitude_loss"]!!
        ) / 2f
        
        // Performance spread metrics
        val accuracyRange = (accuracies.maxOrNull() ?: 0f) - (accuracies.minOrNull() ?: 0f)
        val lossRange = (losses.maxOrNull() ?: 0f) - (losses.minOrNull() ?: 0f)
        
        strategyMetrics["accuracy_spread"] = accuracyRange
        strategyMetrics["loss_spread"] = lossRange
        
        // Sample distribution metrics
        val sampleCounts = evaluationResults.map { it.sampleCount }
        strategyMetrics["total_samples"] = sampleCounts.sum().toFloat()
        strategyMetrics["mean_samples_per_client"] = sampleCounts.average().toFloat()
        strategyMetrics["sample_distribution_cv"] = if (sampleCounts.average() > 0) {
            sqrt(computeVariance(sampleCounts.map { it.toFloat() })) / sampleCounts.average().toFloat()
        } else 0f
        
        return strategyMetrics
    }
    
    /**
     * Filter out invalid or corrupted client results.
     * 
     * Removes client results that contain invalid data such as:
     * - NaN or infinite values in accuracy or loss
     * - Negative sample counts
     * - Accuracy values outside [0, 1] range
     * - Negative loss values
     * 
     * @param evaluationResults Raw list of client evaluation results
     * @return Filtered list containing only valid results
     */
    private fun filterValidResults(
        evaluationResults: List<EvaluationResult>
    ): List<EvaluationResult> {
        return evaluationResults.filter { result ->
            // Check for valid accuracy (0 to 1 range, not NaN/Inf)
            val validAccuracy = result.accuracy.isFinite() && 
                                result.accuracy >= 0f && 
                                result.accuracy <= 1f
            
            // Check for valid loss (non-negative, not NaN/Inf)
            val validLoss = result.loss.isFinite() && result.loss >= 0f
            
            // Check for valid sample count (positive integer)
            val validSampleCount = result.sampleCount > 0
            
            // Check additional metrics for validity
            val validAdditionalMetrics = result.additionalMetrics.values.all { 
                it.isFinite() 
            }
            
            validAccuracy && validLoss && validSampleCount && validAdditionalMetrics
        }
    }
    
    /**
     * Compute sample-weighted mean of a metric across clients.
     * 
     * Uses each client's sample count as the weight for computing the weighted
     * average. This ensures that clients with more evaluation data have
     * proportionally more influence on the final aggregated metric.
     * 
     * @param results List of valid evaluation results
     * @param metricExtractor Function to extract the metric value from each result
     * @return Sample-weighted mean of the metric
     */
    private fun computeWeightedMean(
        results: List<EvaluationResult>,
        metricExtractor: (EvaluationResult) -> Float
    ): Float {
        val totalSamples = results.sumOf { it.sampleCount }
        require(totalSamples > 0) { "Total sample count must be positive" }
        
        val weightedSum = results.sumOf { result ->
            metricExtractor(result).toDouble() * result.sampleCount
        }
        
        return (weightedSum / totalSamples).toFloat()
    }
    
    /**
     * Compute variance of a list of values.
     * 
     * Uses the sample variance formula: Var(X) = Σ(x_i - μ)² / (n - 1)
     * where μ is the sample mean and n is the number of samples.
     * 
     * @param values List of numeric values
     * @return Sample variance
     */
    private fun computeVariance(values: List<Float>): Float {
        if (values.size <= 1) return 0f
        
        val mean = values.average()
        val sumSquaredDeviations = values.sumOf { (it - mean) * (it - mean) }
        
        return (sumSquaredDeviations / (values.size - 1)).toFloat()
    }
    
    /**
     * Handle missing client results by providing default values or interpolation.
     * 
     * When some expected clients fail to provide evaluation results, this method
     * can provide reasonable default values or interpolate based on historical data.
     * This ensures the aggregation process can continue even with partial participation.
     * 
     * @param partialResults Available client results
     * @param expectedClientCount Total number of expected clients
     * @return Augmented results with defaults for missing clients
     */
    public suspend fun handleMissingResults(
        partialResults: List<EvaluationResult>,
        expectedClientCount: Int
    ): List<EvaluationResult> {
        if (partialResults.size >= expectedClientCount) {
            return partialResults
        }
        
        val validResults = filterValidResults(partialResults)
        if (validResults.isEmpty()) {
            throw IllegalArgumentException("No valid results available for missing result handling")
        }
        
        // Compute default values based on available results
        val defaultAccuracy = validResults.map { it.accuracy }.average().toFloat()
        val defaultLoss = validResults.map { it.loss }.average().toFloat()
        val defaultSampleCount = validResults.map { it.sampleCount }.average().toInt()
        
        val augmentedResults = validResults.toMutableList()
        
        // Add default results for missing clients
        val missingCount = expectedClientCount - validResults.size
        repeat(missingCount) { index ->
            augmentedResults.add(
                EvaluationResult(
                    clientId = "missing_client_$index",
                    accuracy = defaultAccuracy,
                    loss = defaultLoss,
                    sampleCount = defaultSampleCount,
                    additionalMetrics = mapOf("is_default" to 1f)
                )
            )
        }
        
        return augmentedResults
    }
    
    /**
     * Validate aggregated metrics for reasonableness and consistency.
     * 
     * Performs sanity checks on the aggregated metrics to detect potential
     * issues in the aggregation process or underlying client data.
     * 
     * @param metrics Aggregated metrics to validate
     * @return True if metrics pass validation checks
     */
    public fun validateAggregatedMetrics(metrics: AggregatedMetrics): Boolean {
        // Check basic constraints
        if (metrics.meanAccuracy < 0f || metrics.meanAccuracy > 1f) return false
        if (metrics.meanLoss < 0f || !metrics.meanLoss.isFinite()) return false
        if (metrics.totalSamples <= 0) return false
        if (metrics.clientCount <= 0) return false
        
        // Check additional metrics for validity
        val invalidAdditionalMetrics = metrics.additionalMetrics.values.any { 
            !it.isFinite() 
        }
        if (invalidAdditionalMetrics) return false
        
        // Check consistency between client count and total samples
        if (metrics.totalSamples < metrics.clientCount) return false
        
        return true
    }
}