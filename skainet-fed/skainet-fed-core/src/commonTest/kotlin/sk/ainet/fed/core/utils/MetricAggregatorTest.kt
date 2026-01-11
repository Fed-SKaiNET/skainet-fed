package sk.ainet.fed.core.utils

import sk.ainet.fed.core.data.EvaluationResult
import sk.ainet.fed.core.data.AggregatedMetrics
import sk.ainet.fed.core.types.*
import kotlinx.coroutines.test.runTest
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlin.test.*
import kotlin.math.abs

/**
 * Unit tests for MetricAggregator class.
 * 
 * Tests the core functionality of metric aggregation including:
 * - Basic aggregation of loss and accuracy metrics
 * - Statistical summary computation
 * - Handling of invalid and missing client results
 * - Strategy-specific metric calculation
 * - Edge cases and error conditions
 */
class MetricAggregatorTest {
    
    private lateinit var mockContext: ExecutionContext
    private lateinit var aggregator: MetricAggregator
    
    @BeforeTest
    fun setup() {
        // Create a mock execution context for testing
        // MetricAggregator doesn't use tensor operations, so we can provide minimal mocks
        mockContext = object : ExecutionContext {
            override val ops: TensorOps get() = TODO("Mock TensorOps not needed for MetricAggregator tests")
            override val executionStats: sk.ainet.context.ExecutionStats get() = TODO("Not needed for MetricAggregator tests")
            override val memoryInfo: sk.ainet.context.MemoryInfo get() = TODO("Not needed for MetricAggregator tests")
            override val phase: sk.ainet.context.Phase get() = TODO("Not needed for MetricAggregator tests")
            override val tensorDataFactory: sk.ainet.lang.tensor.data.TensorDataFactory get() = TODO("Not needed for MetricAggregator tests")
        }
        aggregator = MetricAggregator(mockContext)
    }
    
    @Test
    fun testBasicMetricAggregation() = runTest {
        // Arrange
        val evaluationResults = listOf(
            EvaluationResult(
                clientId = "client1",
                accuracy = 0.8f,
                loss = 0.2f,
                sampleCount = 100
            )
        )
        
        // Act
        val aggregatedMetrics = aggregator.aggregateMetrics(evaluationResults)
        
        // Assert basic functionality
        assertEquals(1, aggregatedMetrics.clientCount)
        assertEquals(100, aggregatedMetrics.totalSamples)
        assertEquals(0.8f, aggregatedMetrics.meanAccuracy, 0.001f)
        assertEquals(0.2f, aggregatedMetrics.meanLoss, 0.001f)
    }
    
    @Test
    fun testLossAggregation() = runTest {
        // Arrange
        val evaluationResults = listOf(
            EvaluationResult("client1", 0.8f, 0.2f, 100),
            EvaluationResult("client2", 0.9f, 0.1f, 200)
        )
        
        // Act
        val aggregatedLoss = aggregator.aggregateLoss(evaluationResults)
        
        // Assert
        // Expected: (0.2*100 + 0.1*200) / 300 = 0.1333
        assertEquals(0.1333f, aggregatedLoss, 0.001f)
    }
    
    @Test
    fun testAccuracyAggregation() = runTest {
        // Arrange
        val evaluationResults = listOf(
            EvaluationResult("client1", 0.8f, 0.2f, 100),
            EvaluationResult("client2", 0.9f, 0.1f, 200)
        )
        
        // Act
        val aggregatedAccuracy = aggregator.aggregateAccuracy(evaluationResults)
        
        // Assert
        // Expected: (0.8*100 + 0.9*200) / 300 = 0.8667
        assertEquals(0.8667f, aggregatedAccuracy, 0.001f)
    }
    
    @Test
    fun testStatisticalSummary() = runTest {
        // Arrange
        val evaluationResults = listOf(
            EvaluationResult("client1", 0.8f, 0.2f, 100),
            EvaluationResult("client2", 0.9f, 0.1f, 200),
            EvaluationResult("client3", 0.7f, 0.3f, 150)
        )
        
        // Act
        val summary = aggregator.computeStatisticalSummary(evaluationResults)
        
        // Assert
        assertEquals(0.8f, summary["accuracy_mean"]!!, 0.001f)
        assertEquals(0.2f, summary["loss_mean"]!!, 0.001f)
        
        // Check that variance and std are computed
        assertTrue(summary.containsKey("accuracy_variance"))
        assertTrue(summary.containsKey("accuracy_std"))
        assertTrue(summary.containsKey("loss_variance"))
        assertTrue(summary.containsKey("loss_std"))
        
        // Check min/max values
        assertEquals(0.7f, summary["accuracy_min"]!!)
        assertEquals(0.9f, summary["accuracy_max"]!!)
        assertEquals(0.1f, summary["loss_min"]!!)
        assertEquals(0.3f, summary["loss_max"]!!)
        
        // Check confidence intervals are present
        assertTrue(summary.containsKey("accuracy_ci_lower"))
        assertTrue(summary.containsKey("accuracy_ci_upper"))
        assertTrue(summary.containsKey("loss_ci_lower"))
        assertTrue(summary.containsKey("loss_ci_upper"))
    }
    
    @Test
    fun testStrategySpecificMetrics() = runTest {
        // Arrange
        val evaluationResults = listOf(
            EvaluationResult("client1", 0.8f, 0.2f, 100),
            EvaluationResult("client2", 0.9f, 0.1f, 200),
            EvaluationResult("client3", 0.7f, 0.3f, 150)
        )
        
        // Act
        val strategyMetrics = aggregator.computeStrategySpecificMetrics(evaluationResults)
        
        // Assert
        assertTrue(strategyMetrics.containsKey("consensus_magnitude"))
        assertTrue(strategyMetrics.containsKey("consensus_magnitude_accuracy"))
        assertTrue(strategyMetrics.containsKey("consensus_magnitude_loss"))
        assertTrue(strategyMetrics.containsKey("accuracy_spread"))
        assertTrue(strategyMetrics.containsKey("loss_spread"))
        assertTrue(strategyMetrics.containsKey("total_samples"))
        assertTrue(strategyMetrics.containsKey("mean_samples_per_client"))
        
        // Verify consensus magnitude is between 0 and 1
        val consensusMagnitude = strategyMetrics["consensus_magnitude"]!!
        assertTrue(consensusMagnitude >= 0f && consensusMagnitude <= 1f)
        
        // Verify spread calculations
        assertEquals(0.2f, strategyMetrics["accuracy_spread"]!!, 0.001f) // 0.9 - 0.7
        assertEquals(0.2f, strategyMetrics["loss_spread"]!!, 0.001f) // 0.3 - 0.1
        
        // Verify sample metrics
        assertEquals(450f, strategyMetrics["total_samples"]!!)
        assertEquals(150f, strategyMetrics["mean_samples_per_client"]!!)
    }
    
    @Test
    fun testInvalidResultFiltering() = runTest {
        // Arrange - mix of valid and invalid results
        val evaluationResults = listOf(
            EvaluationResult("client1", 0.8f, 0.2f, 100), // Valid
            EvaluationResult("client2", Float.NaN, 0.1f, 200), // Invalid accuracy
            EvaluationResult("client3", 0.7f, Float.NEGATIVE_INFINITY, 150), // Invalid loss
            EvaluationResult("client4", 1.5f, 0.1f, 50), // Invalid accuracy > 1
            EvaluationResult("client5", 0.9f, -0.1f, 75), // Invalid negative loss
            EvaluationResult("client6", 0.6f, 0.3f, -10), // Invalid negative sample count
            EvaluationResult("client7", 0.85f, 0.15f, 125) // Valid
        )
        
        // Act
        val aggregatedMetrics = aggregator.aggregateMetrics(evaluationResults)
        
        // Assert - only 2 valid results should be used
        assertEquals(2, aggregatedMetrics.clientCount)
        assertEquals(225, aggregatedMetrics.totalSamples) // 100 + 125
        
        // Verify weighted averages with only valid results
        // Expected weighted accuracy: (0.8*100 + 0.85*125) / 225 = 0.8278
        assertEquals(0.8278f, aggregatedMetrics.meanAccuracy, 0.001f)
    }
    
    @Test
    fun testEmptyResultsThrowsException() = runTest {
        // Arrange
        val emptyResults = emptyList<EvaluationResult>()
        
        // Act & Assert
        assertFailsWith<IllegalArgumentException> {
            aggregator.aggregateMetrics(emptyResults)
        }
    }
    
    @Test
    fun testAllInvalidResultsThrowsException() = runTest {
        // Arrange - all invalid results
        val invalidResults = listOf(
            EvaluationResult("client1", Float.NaN, 0.2f, 100),
            EvaluationResult("client2", 0.8f, Float.POSITIVE_INFINITY, 200)
        )
        
        // Act & Assert
        assertFailsWith<IllegalArgumentException> {
            aggregator.aggregateMetrics(invalidResults)
        }
    }
    
    @Test
    fun testHandleMissingResults() = runTest {
        // Arrange
        val partialResults = listOf(
            EvaluationResult("client1", 0.8f, 0.2f, 100),
            EvaluationResult("client2", 0.9f, 0.1f, 200)
        )
        val expectedClientCount = 5
        
        // Act
        val augmentedResults = aggregator.handleMissingResults(partialResults, expectedClientCount)
        
        // Assert
        assertEquals(expectedClientCount, augmentedResults.size)
        
        // Verify original results are preserved
        assertTrue(augmentedResults.contains(partialResults[0]))
        assertTrue(augmentedResults.contains(partialResults[1]))
        
        // Verify default results are added
        val defaultResults = augmentedResults.filter { it.clientId.startsWith("missing_client_") }
        assertEquals(3, defaultResults.size)
        
        // Verify default values are reasonable (based on available results)
        defaultResults.forEach { result ->
            assertTrue(result.accuracy > 0f && result.accuracy <= 1f)
            assertTrue(result.loss >= 0f)
            assertTrue(result.sampleCount > 0)
            assertTrue(result.additionalMetrics.containsKey("is_default"))
        }
    }
    
    @Test
    fun testValidateAggregatedMetrics() {
        // Test valid metrics
        val validMetrics = AggregatedMetrics(
            meanAccuracy = 0.8f,
            meanLoss = 0.2f,
            totalSamples = 1000,
            clientCount = 10,
            additionalMetrics = mapOf("consensus_magnitude" to 0.9f)
        )
        assertTrue(aggregator.validateAggregatedMetrics(validMetrics))
        
        // Test invalid accuracy (> 1)
        val invalidAccuracy = validMetrics.copy(meanAccuracy = 1.5f)
        assertFalse(aggregator.validateAggregatedMetrics(invalidAccuracy))
        
        // Test invalid loss (negative)
        val invalidLoss = validMetrics.copy(meanLoss = -0.1f)
        assertFalse(aggregator.validateAggregatedMetrics(invalidLoss))
        
        // Test invalid sample count
        val invalidSamples = validMetrics.copy(totalSamples = 0)
        assertFalse(aggregator.validateAggregatedMetrics(invalidSamples))
        
        // Test invalid client count
        val invalidClients = validMetrics.copy(clientCount = 0)
        assertFalse(aggregator.validateAggregatedMetrics(invalidClients))
        
        // Test inconsistent samples vs clients
        val inconsistentMetrics = validMetrics.copy(totalSamples = 5, clientCount = 10)
        assertFalse(aggregator.validateAggregatedMetrics(inconsistentMetrics))
    }
    
    @Test
    fun testSingleClientAggregation() = runTest {
        // Arrange
        val singleResult = listOf(
            EvaluationResult("client1", 0.8f, 0.2f, 100)
        )
        
        // Act
        val aggregatedMetrics = aggregator.aggregateMetrics(singleResult)
        
        // Assert
        assertEquals(1, aggregatedMetrics.clientCount)
        assertEquals(100, aggregatedMetrics.totalSamples)
        assertEquals(0.8f, aggregatedMetrics.meanAccuracy)
        assertEquals(0.2f, aggregatedMetrics.meanLoss)
        
        // Statistical summary should handle single value case
        val summary = aggregator.computeStatisticalSummary(singleResult)
        assertEquals(0f, summary["accuracy_variance"]!!) // Variance should be 0 for single value
        assertEquals(0f, summary["loss_variance"]!!)
    }
    
    @Test
    fun testZeroVarianceCase() = runTest {
        // Arrange - all clients have identical metrics
        val identicalResults = listOf(
            EvaluationResult("client1", 0.8f, 0.2f, 100),
            EvaluationResult("client2", 0.8f, 0.2f, 200),
            EvaluationResult("client3", 0.8f, 0.2f, 150)
        )
        
        // Act
        val summary = aggregator.computeStatisticalSummary(identicalResults)
        val strategyMetrics = aggregator.computeStrategySpecificMetrics(identicalResults)
        
        // Assert
        assertEquals(0f, summary["accuracy_variance"]!!, 0.001f)
        assertEquals(0f, summary["loss_variance"]!!, 0.001f)
        assertEquals(0f, summary["accuracy_std"]!!, 0.001f)
        assertEquals(0f, summary["loss_std"]!!, 0.001f)
        
        // Consensus magnitude should be high (close to 1) when all clients agree
        assertTrue(strategyMetrics["consensus_magnitude"]!! > 0.9f)
        
        // Spread should be zero
        assertEquals(0f, strategyMetrics["accuracy_spread"]!!)
        assertEquals(0f, strategyMetrics["loss_spread"]!!)
    }

    /**
     * Property test for metric aggregation robustness.
     * Feature: federated-learning-strategies, Property 12: Metric Aggregation Robustness
     * Validates: Requirements 7.4, 7.5
     * 
     * Property: For any set of client evaluation results (including missing or invalid entries),
     * metric aggregation should handle edge cases gracefully and produce valid statistical summaries.
     * 
     * Key robustness properties tested:
     * - Invalid results are filtered out correctly (NaN, Inf, out-of-range values)
     * - Statistical summaries are mathematically correct for valid results
     * - Edge cases (single client, identical results, extreme values) are handled
     * - Missing client handling produces reasonable defaults
     * - Aggregated metrics pass validation checks
     * - Weighted averaging respects sample size proportions
     */
    @Test
    fun testMetricAggregationRobustnessProperty() = runTest {
        // Property-based test with 100 iterations as specified in design document
        checkAll(100,
            Arb.list(Arb.float(0f, 1f), 1..10),        // Valid accuracy values [0, 1]
            Arb.list(Arb.float(0f, 5f), 1..10),        // Valid loss values [0, 5]
            Arb.list(Arb.int(1, 1000), 1..10),         // Valid sample counts [1, 1000]
            Arb.int(0, 5),                             // Number of invalid results to inject
            Arb.int(1, 3)                              // Number of missing clients to simulate
        ) { validAccuracies, validLosses, validSampleCounts, numInvalid, numMissing ->
            
            // Ensure all lists have the same size
            val minSize = minOf(validAccuracies.size, validLosses.size, validSampleCounts.size)
            val accuracies = validAccuracies.take(minSize)
            val losses = validLosses.take(minSize)
            val sampleCounts = validSampleCounts.take(minSize)
            
            // Create valid evaluation results
            val validResults = accuracies.zip(losses).zip(sampleCounts).mapIndexed { index, pair ->
                val (accLoss, samples) = pair
                val (acc, loss) = accLoss
                EvaluationResult(
                    clientId = "valid_client_$index",
                    accuracy = acc,
                    loss = loss,
                    sampleCount = samples
                )
            }.toMutableList()
            
            // Inject invalid results to test robustness
            repeat(numInvalid) { index ->
                val invalidResult = when (index % 6) {
                    0 -> EvaluationResult("invalid_$index", Float.NaN, 0.2f, 100)           // NaN accuracy
                    1 -> EvaluationResult("invalid_$index", 0.8f, Float.POSITIVE_INFINITY, 100) // Inf loss
                    2 -> EvaluationResult("invalid_$index", -0.1f, 0.2f, 100)              // Negative accuracy
                    3 -> EvaluationResult("invalid_$index", 1.5f, 0.2f, 100)               // Accuracy > 1
                    4 -> EvaluationResult("invalid_$index", 0.8f, -0.1f, 100)              // Negative loss
                    else -> EvaluationResult("invalid_$index", 0.8f, 0.2f, -10)            // Negative samples
                }
                validResults.add(invalidResult)
            }
            
            // Test basic aggregation robustness
            if (validResults.any { result ->
                result.accuracy.isFinite() && result.accuracy >= 0f && result.accuracy <= 1f &&
                result.loss.isFinite() && result.loss >= 0f && result.sampleCount > 0
            }) {
                
                // Property 1: Aggregation should succeed despite invalid results
                val aggregatedMetrics = try {
                    aggregator.aggregateMetrics(validResults)
                } catch (e: Exception) {
                    fail("Aggregation should handle invalid results gracefully, but threw: ${e.message}")
                }
                
                // Property 2: Aggregated metrics should pass validation
                assertTrue(
                    aggregator.validateAggregatedMetrics(aggregatedMetrics),
                    "Aggregated metrics should pass validation checks"
                )
                
                // Property 3: Only valid results should be counted
                val expectedValidCount = validResults.count { result ->
                    result.accuracy.isFinite() && result.accuracy >= 0f && result.accuracy <= 1f &&
                    result.loss.isFinite() && result.loss >= 0f && result.sampleCount > 0 &&
                    result.additionalMetrics.values.all { it.isFinite() }
                }
                
                assertEquals(
                    expectedValidCount, 
                    aggregatedMetrics.clientCount,
                    "Client count should match number of valid results"
                )
                
                // Property 4: Weighted averaging should respect sample proportions
                val validResultsFiltered = validResults.filter { result ->
                    result.accuracy.isFinite() && result.accuracy >= 0f && result.accuracy <= 1f &&
                    result.loss.isFinite() && result.loss >= 0f && result.sampleCount > 0 &&
                    result.additionalMetrics.values.all { it.isFinite() }
                }
                
                if (validResultsFiltered.isNotEmpty()) {
                    val totalSamples = validResultsFiltered.sumOf { it.sampleCount }
                    val expectedWeightedAccuracy = validResultsFiltered.sumOf { 
                        it.accuracy.toDouble() * it.sampleCount 
                    } / totalSamples
                    val expectedWeightedLoss = validResultsFiltered.sumOf { 
                        it.loss.toDouble() * it.sampleCount 
                    } / totalSamples
                    
                    val tolerance = 0.001f
                    assertTrue(
                        abs(aggregatedMetrics.meanAccuracy - expectedWeightedAccuracy.toFloat()) < tolerance,
                        "Weighted accuracy should be correct: expected $expectedWeightedAccuracy, got ${aggregatedMetrics.meanAccuracy}"
                    )
                    assertTrue(
                        abs(aggregatedMetrics.meanLoss - expectedWeightedLoss.toFloat()) < tolerance,
                        "Weighted loss should be correct: expected $expectedWeightedLoss, got ${aggregatedMetrics.meanLoss}"
                    )
                    
                    assertEquals(
                        totalSamples,
                        aggregatedMetrics.totalSamples,
                        "Total samples should match sum of valid sample counts"
                    )
                }
                
                // Property 5: Statistical summaries should be mathematically correct
                if (validResultsFiltered.size > 1) {
                    val summary = aggregator.computeStatisticalSummary(validResultsFiltered)
                    
                    // Check that statistical measures are reasonable
                    assertTrue(
                        summary["accuracy_variance"]!! >= 0f,
                        "Accuracy variance should be non-negative"
                    )
                    assertTrue(
                        summary["loss_variance"]!! >= 0f,
                        "Loss variance should be non-negative"
                    )
                    assertTrue(
                        summary["accuracy_std"]!! >= 0f,
                        "Accuracy standard deviation should be non-negative"
                    )
                    assertTrue(
                        summary["loss_std"]!! >= 0f,
                        "Loss standard deviation should be non-negative"
                    )
                    
                    // Check min/max bounds
                    val accuracyValues = validResultsFiltered.map { it.accuracy }
                    val lossValues = validResultsFiltered.map { it.loss }
                    
                    assertEquals(
                        accuracyValues.minOrNull() ?: 0f,
                        summary["accuracy_min"]!!,
                        0.001f,
                        "Accuracy minimum should be correct"
                    )
                    assertEquals(
                        accuracyValues.maxOrNull() ?: 0f,
                        summary["accuracy_max"]!!,
                        0.001f,
                        "Accuracy maximum should be correct"
                    )
                    assertEquals(
                        lossValues.minOrNull() ?: 0f,
                        summary["loss_min"]!!,
                        0.001f,
                        "Loss minimum should be correct"
                    )
                    assertEquals(
                        lossValues.maxOrNull() ?: 0f,
                        summary["loss_max"]!!,
                        0.001f,
                        "Loss maximum should be correct"
                    )
                    
                    // Check confidence intervals are present and reasonable
                    assertTrue(
                        summary.containsKey("accuracy_ci_lower"),
                        "Accuracy confidence interval lower bound should be present"
                    )
                    assertTrue(
                        summary.containsKey("accuracy_ci_upper"),
                        "Accuracy confidence interval upper bound should be present"
                    )
                    assertTrue(
                        summary["accuracy_ci_lower"]!! <= summary["accuracy_ci_upper"]!!,
                        "Accuracy confidence interval should be properly ordered"
                    )
                }
                
                // Property 6: Strategy-specific metrics should be reasonable
                if (validResultsFiltered.isNotEmpty()) {
                    val strategyMetrics = aggregator.computeStrategySpecificMetrics(validResultsFiltered)
                    
                    // Consensus magnitude should be in [0, 1] range
                    val consensusMagnitude = strategyMetrics["consensus_magnitude"]!!
                    assertTrue(
                        consensusMagnitude >= 0f && consensusMagnitude <= 1f,
                        "Consensus magnitude should be in [0, 1]: $consensusMagnitude"
                    )
                    
                    // Spread values should be non-negative
                    assertTrue(
                        strategyMetrics["accuracy_spread"]!! >= 0f,
                        "Accuracy spread should be non-negative"
                    )
                    assertTrue(
                        strategyMetrics["loss_spread"]!! >= 0f,
                        "Loss spread should be non-negative"
                    )
                    
                    // Total samples should match
                    assertEquals(
                        validResultsFiltered.sumOf { it.sampleCount }.toFloat(),
                        strategyMetrics["total_samples"]!!,
                        "Strategy metrics total samples should match"
                    )
                }
                
                // Property 7: Missing client handling should work
                if (numMissing > 0 && validResultsFiltered.isNotEmpty()) {
                    val expectedTotalClients = validResultsFiltered.size + numMissing
                    val augmentedResults = aggregator.handleMissingResults(
                        validResultsFiltered, 
                        expectedTotalClients
                    )
                    
                    assertEquals(
                        expectedTotalClients,
                        augmentedResults.size,
                        "Augmented results should have expected client count"
                    )
                    
                    // All original results should be preserved
                    validResultsFiltered.forEach { original ->
                        assertTrue(
                            augmentedResults.any { it.clientId == original.clientId },
                            "Original result should be preserved: ${original.clientId}"
                        )
                    }
                    
                    // Default results should be reasonable
                    val defaultResults = augmentedResults.filter { 
                        it.clientId.startsWith("missing_client_") 
                    }
                    assertEquals(
                        numMissing,
                        defaultResults.size,
                        "Should have correct number of default results"
                    )
                    
                    defaultResults.forEach { defaultResult ->
                        assertTrue(
                            defaultResult.accuracy >= 0f && defaultResult.accuracy <= 1f,
                            "Default accuracy should be valid: ${defaultResult.accuracy}"
                        )
                        assertTrue(
                            defaultResult.loss >= 0f,
                            "Default loss should be non-negative: ${defaultResult.loss}"
                        )
                        assertTrue(
                            defaultResult.sampleCount > 0,
                            "Default sample count should be positive: ${defaultResult.sampleCount}"
                        )
                        assertTrue(
                            defaultResult.additionalMetrics.containsKey("is_default"),
                            "Default result should be marked as default"
                        )
                    }
                }
            }
            
            // Property 8: Edge case - all invalid results should throw exception
            val allInvalidResults = listOf(
                EvaluationResult("invalid1", Float.NaN, 0.2f, 100),
                EvaluationResult("invalid2", 0.8f, Float.NEGATIVE_INFINITY, 100)
            )
            
            assertFailsWith<IllegalArgumentException>(
                "All invalid results should throw exception"
            ) {
                aggregator.aggregateMetrics(allInvalidResults)
            }
            
            // Property 9: Edge case - empty results should throw exception
            assertFailsWith<IllegalArgumentException>(
                "Empty results should throw exception"
            ) {
                aggregator.aggregateMetrics(emptyList())
            }
        }
    }
}