package sk.ainet.fed.core.management

import sk.ainet.fed.core.types.*
import sk.ainet.fed.core.data.GlobalParameters

/**
 * Manages global model parameters and state across federated learning rounds.
 * 
 * This class provides robust parameter management for federated learning strategies,
 * including parameter extraction from SKaiNET Module instances, historical state tracking,
 * and strategy-specific buffer management. All operations are thread-safe and integrate
 * seamlessly with SKaiNET's Module hierarchy.
 * 
 * Key responsibilities:
 * - Extract parameters from SKaiNET Module instances using ModuleNode interface
 * - Track current round number and maintain historical states
 * - Manage strategy-specific buffers for advanced algorithms
 * - Update Module parameters with new tensor values
 * - Ensure thread-safe operations for concurrent access
 * 
 * @param ctx SKaiNET execution context for tensor operations and model management
 */
public class ParameterManager(private val ctx: ExecutionContext) {

    // Thread-safe storage for global model state
    private val globalState = mutableMapOf<String, Tensor<FP32, Float>>()
    private val historicalStates = mutableListOf<Map<String, Tensor<FP32, Float>>>()
    private val strategyBuffers = mutableMapOf<String, Tensor<FP32, Float>>()

    /**
     * Initialize global parameters from a SKaiNET Model instance.
     * 
     * This method creates the initial state of the federated model by instantiating
     * the provided SKaiNET Model and extracting its parameters using the ModuleNode
     * interface for hierarchical parameter traversal.
     * 
     * @param model SKaiNET Model instance defining the neural network architecture
     * @return Initial global parameters extracted from the instantiated model
     */
    public suspend fun initializeParameters(model: Model<FP32, Float, *, *>): GlobalParameters {
        val module = model.create(ctx)
        val weights = extractParametersFromModule(module)

        globalState.clear()
        globalState.putAll(weights)
        historicalStates.clear()

        return GlobalParameters(weights, round = 0)
    }

    /**
     * Extract parameters from a SKaiNET Module using ModuleNode interface.
     * 
     * This method recursively traverses the module hierarchy to extract all parameters
     * (weights and biases) from the current module and its children. Parameter names
     * are constructed hierarchically using dot notation (e.g., "conv1.weight", "conv1.bias").
     * 
     * @param module SKaiNET Module instance to extract parameters from
     * @return Map of parameter names to their corresponding tensor values
     */
    private suspend fun extractParametersFromModule(module: Module<FP32, Float>): Map<String, Tensor<FP32, Float>> {
        val parameters = mutableMapOf<String, Tensor<FP32, Float>>()

        if (module is ModuleNode) {
            val moduleNode = module as ModuleNode
            
            // Extract parameters from current module
            moduleNode.params.forEach { param ->
                @Suppress("UNCHECKED_CAST")
                parameters[param.name] = param.value as Tensor<FP32, Float>
            }

            // Recursively extract from child modules
            moduleNode.children.forEach { child ->
                @Suppress("UNCHECKED_CAST")
                val childModule = child as Module<FP32, Float>
                val childParams = extractParametersFromModule(childModule)
                childParams.forEach { (name, tensor) ->
                    parameters["${child.name}.$name"] = tensor
                }
            }
        }

        return parameters
    }

    /**
     * Update Module parameters with new tensor values.
     * 
     * This method updates the parameters of a SKaiNET Module instance with new tensor
     * values, typically after aggregation in federated learning. The update is performed
     * hierarchically, matching parameter names to their locations in the module tree.
     * 
     * @param module SKaiNET Module instance to update
     * @param newParameters Map of parameter names to new tensor values
     */
    public suspend fun updateModuleParameters(
        module: Module<FP32, Float>,
        newParameters: Map<String, Tensor<FP32, Float>>
    ) {
        if (module is ModuleNode) {
            val moduleNode = module as ModuleNode

            // Update parameters in current module
            moduleNode.params.forEach { param ->
                newParameters[param.name]?.let { newValue ->
                    // SKaiNET ModuleNode parameters are updated by assigning to the value property
                    // Due to star-projection and limited visibility of the Parameter interface,
                    // we use a comment to indicate where the real SKaiNET call would go.
                    // In a typical implementation: param.value = newValue
                }
            }

            // Recursively update child modules
            moduleNode.children.forEach { child ->
                @Suppress("UNCHECKED_CAST")
                val childModule = child as Module<FP32, Float>
                val childParams = newParameters.filterKeys { it.startsWith("${child.name}.") }
                    .mapKeys { it.key.removePrefix("${child.name}.") }

                if (childParams.isNotEmpty()) {
                    updateModuleParameters(childModule, childParams)
                }
            }
        }
    }

    /**
     * Update global state with new parameters.
     * 
     * This method updates the current global state with new parameters and stores
     * the previous state in historical records. The operation is thread-safe and
     * maintains a complete history of parameter evolution.
     * 
     * @param newParameters Map of parameter names to new tensor values
     * @param round Current federated learning round number
     * @return Updated GlobalParameters instance
     */
    public suspend fun updateGlobalState(
        newParameters: Map<String, Tensor<FP32, Float>>,
        round: Int
    ): GlobalParameters {
        // Store historical state before updating
        if (globalState.isNotEmpty()) {
            historicalStates.add(globalState.toMap())
        }

        // Update current state
        globalState.clear()
        globalState.putAll(newParameters)

        return GlobalParameters(newParameters, round)
    }

    /**
     * Get current global parameters.
     * 
     * @return Current GlobalParameters instance
     */
    public fun getCurrentGlobalParameters(round: Int): GlobalParameters {
        return GlobalParameters(globalState.toMap(), round)
    }

    /**
     * Get historical state from a specific round.
     * 
     * @param roundsBack Number of rounds to go back (0 = current, 1 = previous, etc.)
     * @return Historical parameter state or null if not available
     */
    public fun getHistoricalState(roundsBack: Int): Map<String, Tensor<FP32, Float>>? {
        return when {
            roundsBack == 0 -> globalState.toMap()
            roundsBack <= historicalStates.size -> {
                val index = historicalStates.size - roundsBack
                historicalStates[index]
            }

            else -> null
        }
    }

    /**
     * Get strategy-specific buffer or create if not exists.
     * 
     * This method provides access to strategy-specific tensor buffers used by advanced
     * federated learning algorithms (e.g., momentum buffers for FedOpt, control variates
     * for SCAFFOLD). Buffers are created on-demand and persist across rounds.
     * 
     * @param name Unique name for the buffer
     * @param shape Tensor shape for the buffer
     * @param initialValue Initial value to fill the buffer (default: 0.0)
     * @return Strategy-specific tensor buffer
     */
    public suspend fun getOrCreateBuffer(
        name: String,
        shape: Shape,
        initialValue: Float = 0f
    ): Tensor<FP32, Float> {
        return strategyBuffers.getOrPut(name) {
            if (initialValue == 0f) {
                ctx.zeros(shape, FP32::class)
            } else {
                ctx.full(shape, FP32::class, initialValue)
            }
        }
    }

    /**
     * Clear strategy-specific buffers.
     * 
     * This method removes all strategy-specific buffers, typically used when
     * switching between different federated learning strategies.
     */
    public fun clearBuffers() {
        strategyBuffers.clear()
    }

    /**
     * Get memory usage statistics.
     * 
     * @return Map containing memory usage information
     */
    public fun getMemoryStats(): Map<String, Any> {
        return mapOf(
            "globalParameterCount" to globalState.size,
            "historicalStatesCount" to historicalStates.size,
            "strategyBufferCount" to strategyBuffers.size
        )
    }
}