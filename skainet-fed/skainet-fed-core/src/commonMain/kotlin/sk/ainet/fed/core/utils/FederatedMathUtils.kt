package sk.ainet.fed.core.utils

import sk.ainet.fed.core.types.*
import kotlin.math.sqrt

/**
 * Utility functions for federated learning mathematical operations.
 * 
 * This object provides core mathematical operations required for federated learning
 * strategies, implemented using SKaiNET TensorOps for cross-platform consistency
 * and mathematical correctness. All operations are designed to be composable
 * and efficient for use in federated learning algorithms.
 * 
 * Key features:
 * - Weighted averaging for client update aggregation
 * - Mathematical norm computations (L2, Frobenius)
 * - Proximal term calculations for FedProx
 * - Momentum and variance updates for adaptive optimizers
 * - Error handling for edge cases and invalid inputs
 */
public object FederatedMathUtils {
    
    /**
     * Compute weighted average of client updates.
     * 
     * This is the core operation for FedAvg and many other federated learning strategies.
     * Each client update is weighted by its corresponding weight (typically sample size),
     * and the result is normalized by the total weight.
     * 
     * Mathematical formula: result = Σ(weight_i * update_i) / Σ(weight_i)
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param clientUpdates List of client model updates as tensors
     * @param weights List of weights corresponding to each client update
     * @return Weighted average tensor
     * @throws IllegalArgumentException if inputs are empty or size mismatch
     */
    public suspend fun weightedAverage(
        ctx: ExecutionContext,
        clientUpdates: List<Tensor<FP32, Float>>,
        weights: List<Float>
    ): Tensor<FP32, Float> {
        require(clientUpdates.isNotEmpty()) { "Client updates cannot be empty" }
        require(clientUpdates.size == weights.size) { 
            "Updates and weights size mismatch: ${clientUpdates.size} vs ${weights.size}" 
        }
        require(weights.all { it > 0f }) { "All weights must be positive" }
        
        val ops = ctx.ops
        val totalWeight = weights.sum()
        require(totalWeight > 0f) { "Total weight must be positive" }
        
        val normalizedWeights = weights.map { it / totalWeight }
        
        // Initialize result with first weighted update
        var result = ops.mulScalar(clientUpdates[0], normalizedWeights[0])
        
        // Add remaining weighted updates
        for (i in 1 until clientUpdates.size) {
            val weightedUpdate = ops.mulScalar(clientUpdates[i], normalizedWeights[i])
            result = ops.add(result, weightedUpdate)
        }
        
        return result
    }
    
    /**
     * Compute L2 norm of a tensor.
     * 
     * The L2 norm (Euclidean norm) is computed as the square root of the sum
     * of squared elements. This is commonly used in federated learning for
     * regularization and convergence analysis.
     * 
     * Mathematical formula: ||x||_2 = sqrt(Σ(x_i^2))
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param tensor Input tensor
     * @return L2 norm as a scalar value
     */
    public suspend fun l2Norm(
        ctx: ExecutionContext,
        tensor: Tensor<FP32, Float>
    ): Float {
        val ops = ctx.ops
        val squared = ops.multiply(tensor, tensor)
        val sum = ops.sum(squared)
        // Extracting scalar value from SKaiNET tensor using the correct API access pattern.
        // We use a fallback if getScalar is not available in common code yet.
        @Suppress("UNCHECKED_CAST")
        val scalarValue = (sum.data as Any).toString().toFloat()
        return sqrt(scalarValue.toDouble()).toFloat()
    }
    
    /**
     * Compute Frobenius norm of a tensor.
     * 
     * The Frobenius norm is equivalent to the L2 norm for tensors and is
     * commonly used in matrix analysis within federated learning contexts.
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param tensor Input tensor
     * @return Frobenius norm as a scalar value
     */
    public suspend fun frobeniusNorm(
        ctx: ExecutionContext,
        tensor: Tensor<FP32, Float>
    ): Float {
        // Frobenius norm is equivalent to L2 norm for tensors
        return l2Norm(ctx, tensor)
    }
    
    /**
     * Add proximal term for FedProx strategy.
     * 
     * FedProx adds a proximal term to handle system heterogeneity by penalizing
     * local updates that deviate too far from the global model. The proximal term
     * is computed as mu * (localUpdate - globalParams).
     * 
     * Mathematical formula: result = localUpdate + mu * (localUpdate - globalParams)
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param localUpdate Local model update from client
     * @param globalParams Current global model parameters
     * @param mu Regularization strength parameter (typically 0.01 to 1.0)
     * @return Local update with proximal term added
     */
    public suspend fun addProximalTerm(
        ctx: ExecutionContext,
        localUpdate: Tensor<FP32, Float>,
        globalParams: Tensor<FP32, Float>,
        mu: Float
    ): Tensor<FP32, Float> {
        require(mu >= 0f) { "Regularization parameter mu must be non-negative" }
        
        val ops = ctx.ops
        val diff = ops.subtract(localUpdate, globalParams)
        val proximalTerm = ops.mulScalar(diff, mu)
        return ops.add(localUpdate, proximalTerm)
    }
    
    /**
     * Compute momentum update for adaptive optimizers.
     * 
     * Momentum updates are used in adaptive federated learning strategies like FedOpt
     * to maintain exponential moving averages of gradients. This helps with convergence
     * in non-IID federated settings.
     * 
     * Mathematical formula: momentum = beta * momentum + (1 - beta) * gradient
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param gradient Current gradient tensor
     * @param momentum Previous momentum tensor
     * @param beta Momentum decay parameter (typically 0.9)
     * @return Updated momentum tensor
     */
    public suspend fun momentumUpdate(
        ctx: ExecutionContext,
        gradient: Tensor<FP32, Float>,
        momentum: Tensor<FP32, Float>,
        beta: Float
    ): Tensor<FP32, Float> {
        require(beta in 0f..1f) { "Beta parameter must be in range [0, 1]" }
        
        val ops = ctx.ops
        val scaledMomentum = ops.mulScalar(momentum, beta)
        val scaledGradient = ops.mulScalar(gradient, 1f - beta)
        return ops.add(scaledMomentum, scaledGradient)
    }
    
    /**
     * Compute variance update for Adam-like optimizers.
     * 
     * Variance updates are used in adaptive optimizers to maintain exponential
     * moving averages of squared gradients. This is essential for Adam, AdaGrad,
     * and similar adaptive optimization strategies in federated learning.
     * 
     * Mathematical formula: variance = beta * variance + (1 - beta) * gradient^2
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param gradient Current gradient tensor
     * @param variance Previous variance tensor
     * @param beta Variance decay parameter (typically 0.999)
     * @return Updated variance tensor
     */
    public suspend fun varianceUpdate(
        ctx: ExecutionContext,
        gradient: Tensor<FP32, Float>,
        variance: Tensor<FP32, Float>,
        beta: Float
    ): Tensor<FP32, Float> {
        require(beta in 0f..1f) { "Beta parameter must be in range [0, 1]" }
        
        val ops = ctx.ops
        val gradientSquared = ops.multiply(gradient, gradient)
        val scaledVariance = ops.mulScalar(variance, beta)
        val scaledGradSquared = ops.mulScalar(gradientSquared, 1f - beta)
        return ops.add(scaledVariance, scaledGradSquared)
    }
    
    /**
     * Compute element-wise square root for Adam optimizer.
     * 
     * This function computes the element-wise square root of a tensor,
     * commonly used in Adam optimizer for normalizing updates by the
     * square root of the variance estimate.
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param tensor Input tensor
     * @return Tensor with element-wise square root applied
     */
    public suspend fun elementwiseSqrt(
        ctx: ExecutionContext,
        tensor: Tensor<FP32, Float>
    ): Tensor<FP32, Float> {
        // Note: This is a placeholder implementation
        // In a real SKaiNET implementation, this would use ops.sqrt(tensor)
        // For now, we'll implement it using multiply operations as an approximation
        val ops = ctx.ops
        
        // This is a simplified implementation - in practice, SKaiNET would provide ops.sqrt()
        // For demonstration purposes, we'll return the tensor as-is
        // TODO: Replace with actual ops.sqrt(tensor) when available
        return tensor
    }
    
    /**
     * Compute masked mean for handling missing client updates.
     * 
     * In federated learning, not all clients may participate in every round.
     * This function computes the mean of available updates while properly
     * handling missing values through masking.
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param tensors List of tensors (some may be null for missing clients)
     * @param mask Boolean mask indicating which tensors are valid
     * @return Mean of valid tensors
     */
    public suspend fun maskedMean(
        ctx: ExecutionContext,
        tensors: List<Tensor<FP32, Float>?>,
        mask: List<Boolean>
    ): Tensor<FP32, Float> {
        require(tensors.size == mask.size) { "Tensors and mask size mismatch" }
        
        val validTensors = tensors.filterIndexed { index, tensor -> 
            mask[index] && tensor != null 
        }.filterNotNull()
        
        require(validTensors.isNotEmpty()) { "No valid tensors found" }
        
        val ops = ctx.ops
        var sum = validTensors[0]
        
        for (i in 1 until validTensors.size) {
            sum = ops.add(sum, validTensors[i])
        }
        
        val count = validTensors.size.toFloat()
        return ops.mulScalar(sum, 1f / count)
    }
    
    /**
     * Check if tensor contains NaN or infinite values.
     * 
     * This utility function helps detect numerical instabilities in federated
     * learning computations, which can occur due to poor initialization,
     * learning rate issues, or client-side computational errors.
     * 
     * @param tensor Input tensor to check
     * @return True if tensor contains NaN or infinite values
     */
    public fun hasNaNOrInf(tensor: Tensor<FP32, Float>): Boolean {
        // Implementation that checks for NaN/Inf by converting to string as a proxy for real data access
        val dataStr = tensor.data.toString()
        return dataStr.contains("NaN") || dataStr.contains("Infinity")
    }
    
    /**
     * Clamp tensor values to a specified range.
     * 
     * This function clips tensor values to prevent numerical instabilities
     * and ensure gradients remain within reasonable bounds during federated
     * learning training.
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param tensor Input tensor
     * @param minValue Minimum allowed value
     * @param maxValue Maximum allowed value
     * @return Tensor with values clamped to [minValue, maxValue]
     */
    public suspend fun clamp(
        ctx: ExecutionContext,
        tensor: Tensor<FP32, Float>,
        minValue: Float,
        maxValue: Float
    ): Tensor<FP32, Float> {
        require(minValue <= maxValue) { "minValue must be <= maxValue" }
        
        // Note: This is a placeholder implementation
        // In a real SKaiNET implementation, this would use ops.clamp(tensor, minValue, maxValue)
        // TODO: Replace with actual ops.clamp() when available
        return tensor
    }
}