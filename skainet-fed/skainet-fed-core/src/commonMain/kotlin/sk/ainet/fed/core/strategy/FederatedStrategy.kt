package sk.ainet.fed.core.strategy

import sk.ainet.fed.core.types.ExecutionContext
import sk.ainet.fed.core.types.Model
import sk.ainet.fed.core.types.FP32
import sk.ainet.fed.core.data.*

/**
 * Core interface defining mathematical operations for federated learning algorithms.
 * 
 * This interface provides the foundation for implementing various federated learning strategies
 * using SKaiNET's Tensor API. All implementations must use only SKaiNET TensorOps for computations
 * to ensure cross-platform consistency and mathematical correctness.
 * 
 * The interface follows a standard federated learning workflow:
 * 1. Initialize global parameters
 * 2. Prepare updates for clients
 * 3. Aggregate client results
 * 4. Evaluate global model performance
 */
public interface FederatedStrategy {
    
    /**
     * Initialize global model parameters from a SKaiNET Model instance.
     * 
     * This method creates the initial state of the federated model by instantiating
     * the provided SKaiNET Model and extracting its parameters. The Model provides
     * the complete neural network definition including architecture, parameter shapes,
     * and initialization logic.
     * 
     * @param ctx SKaiNET execution context for tensor operations and model instantiation
     * @param model SKaiNET Model instance defining the neural network architecture
     * @return Initial global parameters extracted from the instantiated model
     */
    public suspend fun initializeGlobalParameters(
        ctx: ExecutionContext,
        model: Model<FP32, Float, *, *>
    ): GlobalParameters
    
    /**
     * Prepare mathematical state for client update.
     * 
     * This method prepares the information that will be sent to clients for local training.
     * Different strategies may include additional data beyond the global weights
     * (e.g., control variates for SCAFFOLD, momentum terms for FedOpt).
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param round Current federated learning round number
     * @param globalParameters Current global model parameters
     * @return Strategy update containing data for client training
     */
    public suspend fun prepareClientUpdate(
        ctx: ExecutionContext,
        round: Int,
        globalParameters: GlobalParameters
    ): StrategyUpdate
    
    /**
     * Aggregate client updates using mathematical operations.
     * 
     * This is the core method where different federated learning strategies implement
     * their specific aggregation logic. All tensor operations must use SKaiNET TensorOps
     * to ensure mathematical correctness and cross-platform consistency.
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param round Current federated learning round number
     * @param currentGlobalParameters Current global model parameters
     * @param clientResults List of client training results to aggregate
     * @return Updated global parameters after aggregation
     */
    public suspend fun aggregateClientUpdates(
        ctx: ExecutionContext,
        round: Int,
        currentGlobalParameters: GlobalParameters,
        clientResults: List<ClientResult>
    ): GlobalParameters
    
    /**
     * Aggregate evaluation metrics from clients.
     * 
     * This method combines evaluation results from multiple clients to provide
     * a comprehensive assessment of the global model performance.
     * 
     * @param ctx SKaiNET execution context for tensor operations
     * @param round Current federated learning round number
     * @param globalParameters Current global model parameters
     * @param evaluationResults List of client evaluation results
     * @return Aggregated metrics summarizing global model performance
     */
    public suspend fun evaluateGlobalModel(
        ctx: ExecutionContext,
        round: Int,
        globalParameters: GlobalParameters,
        evaluationResults: List<EvaluationResult>
    ): AggregatedMetrics
}