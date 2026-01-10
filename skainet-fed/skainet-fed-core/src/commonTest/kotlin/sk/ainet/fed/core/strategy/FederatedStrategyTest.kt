package sk.ainet.fed.core.strategy

import sk.ainet.fed.core.types.*
import sk.ainet.fed.core.data.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Test suite for FederatedStrategy interface compliance and basic functionality.
 */
class FederatedStrategyTest {
    
    /**
     * Create a mock ExecutionContext for testing purposes.
     */
    private fun createMockExecutionContext(): ExecutionContext {
        return object : ExecutionContext {}
    }
    
    /**
     * Create a mock Model for testing purposes.
     */
    private fun createMockModel(): Model<FP32, Float, *, *> {
        return object : Model<FP32, Float, Any, Any> {}
    }
    
    /**
     * Create a mock Tensor for testing purposes.
     */
    private fun createMockTensor(): Tensor<FP32, Float> {
        return object : Tensor<FP32, Float> {}
    }
    
    /**
     * Test that FederatedStrategy interface defines all required methods with correct signatures.
     * This test verifies the interface compliance with the new Model-based approach.
     */
    @Test
    fun testInterfaceCompliance() {
        // Verify that the interface has all required methods by checking we can reference them
        // This is a compile-time check that the interface has the expected structure
        
        val mockCtx = createMockExecutionContext()
        val mockModel = createMockModel()
        val mockTensor = createMockTensor()
        
        // Create a mock implementation to verify the interface structure
        val mockStrategy = object : FederatedStrategy {
            override suspend fun initializeGlobalParameters(
                ctx: ExecutionContext,
                model: Model<FP32, Float, *, *>
            ): GlobalParameters {
                // Test that the method accepts Model instead of ModelShape
                assertNotNull(model)
                return GlobalParameters(
                    weights = mapOf("layer1.weight" to mockTensor),
                    round = 0
                )
            }
            
            override suspend fun prepareClientUpdate(
                ctx: ExecutionContext,
                round: Int,
                globalParameters: GlobalParameters
            ): StrategyUpdate {
                return StrategyUpdate(
                    globalWeights = globalParameters.weights,
                    strategySpecificData = emptyMap(),
                    round = round
                )
            }
            
            override suspend fun aggregateClientUpdates(
                ctx: ExecutionContext,
                round: Int,
                currentGlobalParameters: GlobalParameters,
                clientResults: List<ClientResult>
            ): GlobalParameters {
                return currentGlobalParameters.copy(round = round + 1)
            }
            
            override suspend fun evaluateGlobalModel(
                ctx: ExecutionContext,
                round: Int,
                globalParameters: GlobalParameters,
                evaluationResults: List<EvaluationResult>
            ): AggregatedMetrics {
                return AggregatedMetrics(
                    meanAccuracy = 0.85f,
                    meanLoss = 0.3f,
                    totalSamples = evaluationResults.sumOf { it.sampleCount },
                    clientCount = evaluationResults.size
                )
            }
        }
        
        // If we can create the mock implementation, the interface is properly defined
        assertNotNull(mockStrategy)
    }
    
    /**
     * Test SKaiNET Model integration in the interface.
     * Verifies that the interface properly accepts Model instances.
     */
    @Test
    fun testSKaiNETModelIntegration() {
        val mockCtx = createMockExecutionContext()
        val mockModel = createMockModel()
        val mockTensor = createMockTensor()
        
        val strategy = object : FederatedStrategy {
            override suspend fun initializeGlobalParameters(
                ctx: ExecutionContext,
                model: Model<FP32, Float, *, *>
            ): GlobalParameters {
                // Verify that we receive a proper Model instance
                assertNotNull(model)
                
                // Return GlobalParameters with mock tensor data
                return GlobalParameters(
                    weights = mapOf(
                        "conv1.weight" to mockTensor,
                        "conv1.bias" to mockTensor,
                        "fc1.weight" to mockTensor,
                        "fc1.bias" to mockTensor
                    ),
                    round = 0,
                    metadata = mapOf("model_type" to "cnn")
                )
            }
            
            override suspend fun prepareClientUpdate(
                ctx: ExecutionContext,
                round: Int,
                globalParameters: GlobalParameters
            ): StrategyUpdate = StrategyUpdate(emptyMap(), emptyMap(), round)
            
            override suspend fun aggregateClientUpdates(
                ctx: ExecutionContext,
                round: Int,
                currentGlobalParameters: GlobalParameters,
                clientResults: List<ClientResult>
            ): GlobalParameters = currentGlobalParameters
            
            override suspend fun evaluateGlobalModel(
                ctx: ExecutionContext,
                round: Int,
                globalParameters: GlobalParameters,
                evaluationResults: List<EvaluationResult>
            ): AggregatedMetrics = AggregatedMetrics(0f, 0f, 0, 0)
        }
        
        // Test that the strategy can be created and interface methods are accessible
        assertNotNull(strategy)
        
        // Test that we can create the expected data structures
        val expectedGlobalParams = GlobalParameters(
            weights = mapOf(
                "conv1.weight" to mockTensor,
                "conv1.bias" to mockTensor,
                "fc1.weight" to mockTensor,
                "fc1.bias" to mockTensor
            ),
            round = 0,
            metadata = mapOf("model_type" to "cnn")
        )
        
        // Verify the GlobalParameters structure
        assertNotNull(expectedGlobalParams)
        assertEquals(0, expectedGlobalParams.round)
        assertEquals(4, expectedGlobalParams.weights.size)
        assertTrue(expectedGlobalParams.weights.containsKey("conv1.weight"))
        assertTrue(expectedGlobalParams.weights.containsKey("fc1.bias"))
        assertEquals("cnn", expectedGlobalParams.metadata["model_type"])
    }
    
    /**
     * Test that data classes have proper structure and can be instantiated.
     */
    @Test
    fun testDataClassInstantiation() {
        val mockTensor = createMockTensor()
        
        // Test that we can create GlobalParameters with tensor data
        val globalParams = GlobalParameters(
            weights = mapOf(
                "layer1.weight" to mockTensor,
                "layer1.bias" to mockTensor
            ),
            round = 0,
            metadata = mapOf("strategy" to "fedavg")
        )
        assertEquals(0, globalParams.round)
        assertEquals(2, globalParams.weights.size)
        assertEquals("fedavg", globalParams.metadata["strategy"])
        
        // Test StrategyUpdate creation
        val strategyUpdate = StrategyUpdate(
            globalWeights = mapOf("layer1.weight" to mockTensor),
            strategySpecificData = mapOf("momentum" to mockTensor),
            round = 1
        )
        assertEquals(1, strategyUpdate.round)
        assertEquals(1, strategyUpdate.globalWeights.size)
        assertEquals(1, strategyUpdate.strategySpecificData.size)
        
        // Test ClientResult creation
        val clientResult = ClientResult(
            clientId = "client1",
            weightUpdates = mapOf("layer1.weight" to mockTensor),
            sampleCount = 100,
            trainingLoss = 0.5f,
            strategySpecificData = mapOf("local_momentum" to mockTensor)
        )
        assertEquals("client1", clientResult.clientId)
        assertEquals(100, clientResult.sampleCount)
        assertEquals(0.5f, clientResult.trainingLoss)
        assertEquals(1, clientResult.weightUpdates.size)
        assertEquals(1, clientResult.strategySpecificData.size)
        
        // Test EvaluationResult creation
        val evaluationResult = EvaluationResult(
            clientId = "client1",
            accuracy = 0.85f,
            loss = 0.3f,
            sampleCount = 50,
            additionalMetrics = mapOf("precision" to 0.82f, "recall" to 0.88f)
        )
        assertEquals(0.85f, evaluationResult.accuracy)
        assertEquals(0.3f, evaluationResult.loss)
        assertEquals(50, evaluationResult.sampleCount)
        assertEquals(2, evaluationResult.additionalMetrics.size)
        
        // Test AggregatedMetrics creation
        val aggregatedMetrics = AggregatedMetrics(
            meanAccuracy = 0.82f,
            meanLoss = 0.35f,
            totalSamples = 200,
            clientCount = 4,
            additionalMetrics = mapOf("consensus_score" to 0.91f)
        )
        assertEquals(0.82f, aggregatedMetrics.meanAccuracy)
        assertEquals(0.35f, aggregatedMetrics.meanLoss)
        assertEquals(200, aggregatedMetrics.totalSamples)
        assertEquals(4, aggregatedMetrics.clientCount)
        assertEquals(0.91f, aggregatedMetrics.additionalMetrics["consensus_score"])
    }
    
    /**
     * Test that data classes support strategy-specific data fields.
     */
    @Test
    fun testStrategySpecificData() {
        val mockTensor = createMockTensor()
        
        val strategyUpdate = StrategyUpdate(
            globalWeights = mapOf("layer1.weight" to mockTensor),
            strategySpecificData = mapOf(
                "control_variate" to mockTensor,
                "server_momentum" to mockTensor
            ),
            round = 1
        )
        
        assertNotNull(strategyUpdate.strategySpecificData)
        assertEquals(2, strategyUpdate.strategySpecificData.size)
        assertTrue(strategyUpdate.strategySpecificData.containsKey("control_variate"))
        assertTrue(strategyUpdate.strategySpecificData.containsKey("server_momentum"))
        
        val clientResult = ClientResult(
            clientId = "client1",
            weightUpdates = mapOf("layer1.weight" to mockTensor),
            sampleCount = 100,
            trainingLoss = 0.5f,
            strategySpecificData = mapOf(
                "local_control_variate" to mockTensor,
                "client_drift" to mockTensor
            )
        )
        
        assertNotNull(clientResult.strategySpecificData)
        assertEquals(2, clientResult.strategySpecificData.size)
        assertTrue(clientResult.strategySpecificData.containsKey("local_control_variate"))
        assertTrue(clientResult.strategySpecificData.containsKey("client_drift"))
    }
    
    /**
     * Test ModelShape data class (still used for architecture definition).
     */
    @Test
    fun testModelShapeDataClass() {
        // Test LayerShape creation
        val layerShape = LayerShape(
            name = "dense1.weight",
            shape = Shape.of(10, 5),
            dtype = FP32::class
        )
        val modelShape = ModelShape(listOf(layerShape))
        
        assertEquals("dense1.weight", layerShape.name)
        assertEquals(Shape.of(10, 5), layerShape.shape)
        assertEquals(FP32::class, layerShape.dtype)
        assertEquals(1, modelShape.layers.size)
        
        // Test multiple layers
        val biasShape = LayerShape(
            name = "dense1.bias",
            shape = Shape.of(5),
            dtype = FP32::class
        )
        val multiLayerModel = ModelShape(listOf(layerShape, biasShape))
        assertEquals(2, multiLayerModel.layers.size)
        assertEquals("dense1.bias", multiLayerModel.layers[1].name)
    }
}