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
     * Test that FederatedStrategy interface defines all required methods with correct signatures.
     */
    @Test
    fun testInterfaceCompliance() {
        // Verify that the interface has all required methods by checking we can reference them
        // This is a compile-time check that the interface has the expected structure
        
        // Create a mock implementation to verify the interface structure
        val mockStrategy = object : FederatedStrategy {
            override suspend fun initializeGlobalParameters(
                ctx: ExecutionContext,
                modelShape: ModelShape
            ): GlobalParameters {
                return GlobalParameters(emptyMap(), 0)
            }
            
            override suspend fun prepareClientUpdate(
                ctx: ExecutionContext,
                round: Int,
                globalParameters: GlobalParameters
            ): StrategyUpdate {
                return StrategyUpdate(emptyMap(), emptyMap(), round)
            }
            
            override suspend fun aggregateClientUpdates(
                ctx: ExecutionContext,
                round: Int,
                currentGlobalParameters: GlobalParameters,
                clientResults: List<ClientResult>
            ): GlobalParameters {
                return currentGlobalParameters
            }
            
            override suspend fun evaluateGlobalModel(
                ctx: ExecutionContext,
                round: Int,
                globalParameters: GlobalParameters,
                evaluationResults: List<EvaluationResult>
            ): AggregatedMetrics {
                return AggregatedMetrics(0f, 0f, 0, 0)
            }
        }
        
        // If we can create the mock implementation, the interface is properly defined
        assertNotNull(mockStrategy)
    }
    
    /**
     * Test that data classes have proper structure and can be instantiated.
     */
    @Test
    fun testDataClassInstantiation() {
        // Test ModelShape creation
        val layerShape = LayerShape(
            name = "dense1.weight",
            shape = Shape.of(10, 5),
            dtype = FP32::class
        )
        val modelShape = ModelShape(listOf(layerShape))
        
        assertEquals("dense1.weight", layerShape.name)
        assertEquals(1, modelShape.layers.size)
        
        // Test that we can create empty data structures
        val globalParams = GlobalParameters(
            weights = emptyMap(),
            round = 0
        )
        assertEquals(0, globalParams.round)
        assertEquals(0, globalParams.weights.size)
        
        val strategyUpdate = StrategyUpdate(
            globalWeights = emptyMap(),
            round = 1
        )
        assertEquals(1, strategyUpdate.round)
        
        val clientResult = ClientResult(
            clientId = "client1",
            weightUpdates = emptyMap(),
            sampleCount = 100,
            trainingLoss = 0.5f
        )
        assertEquals("client1", clientResult.clientId)
        assertEquals(100, clientResult.sampleCount)
        
        val evaluationResult = EvaluationResult(
            clientId = "client1",
            accuracy = 0.85f,
            loss = 0.3f,
            sampleCount = 50
        )
        assertEquals(0.85f, evaluationResult.accuracy)
        
        val aggregatedMetrics = AggregatedMetrics(
            meanAccuracy = 0.82f,
            meanLoss = 0.35f,
            totalSamples = 200,
            clientCount = 4
        )
        assertEquals(4, aggregatedMetrics.clientCount)
    }
    
    /**
     * Test that data classes support strategy-specific data fields.
     */
    @Test
    fun testStrategySpecificData() {
        val strategyUpdate = StrategyUpdate(
            globalWeights = emptyMap(),
            strategySpecificData = emptyMap(),
            round = 1
        )
        
        assertNotNull(strategyUpdate.strategySpecificData)
        
        val clientResult = ClientResult(
            clientId = "client1",
            weightUpdates = emptyMap(),
            sampleCount = 100,
            trainingLoss = 0.5f,
            strategySpecificData = emptyMap()
        )
        
        assertNotNull(clientResult.strategySpecificData)
    }
}