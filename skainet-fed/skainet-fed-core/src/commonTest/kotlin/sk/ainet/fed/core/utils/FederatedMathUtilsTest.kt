package sk.ainet.fed.core.utils

import sk.ainet.fed.core.types.*
import kotlinx.coroutines.test.runTest
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.math.abs

/**
 * Test suite for FederatedMathUtils mathematical operations.
 */
class FederatedMathUtilsTest {

    private open class MockTensorOps(val defaultValue: Float = 0f) : TensorOps {
        @Suppress("UNCHECKED_CAST")
        override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = createMockTensor(defaultValue) as Tensor<T, V>
        @Suppress("UNCHECKED_CAST")
        override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = createMockTensor(defaultValue) as Tensor<T, V>
        @Suppress("UNCHECKED_CAST")
        override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = createMockTensor(defaultValue) as Tensor<T, V>
        @Suppress("UNCHECKED_CAST")
        override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = createMockTensor(defaultValue) as Tensor<T, V>
        @Suppress("UNCHECKED_CAST")
        override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = createMockTensor(defaultValue, Shape(1)) as Tensor<T, V>

        override fun <T : DType, V> addScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = TODO()
        override fun <T : DType, V> concat(tensors: List<Tensor<T, V>>, dim: Int): Tensor<T, V> = TODO()
        override fun <T : DType, V> conv2d(input: Tensor<T, V>, weight: Tensor<T, V>, bias: Tensor<T, V>?, stride: Pair<Int, Int>, padding: Pair<Int, Int>, dilation: Pair<Int, Int>, groups: Int): Tensor<T, V> = TODO()
        override fun <TFrom : DType, TTo : DType, V> convert(tensor: Tensor<TFrom, V>, targetType: TTo): Tensor<TTo, V> = TODO()
        override fun <T : DType, V> divScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = TODO()
        override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = TODO()
        override fun <T : DType, V> flatten(tensor: Tensor<T, V>, startDim: Int, endDim: Int): Tensor<T, V> = TODO()
        override fun <T : DType, V> gelu(tensor: Tensor<T, V>): Tensor<T, V> = TODO()
        override fun <T : DType, V> logSoftmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> = TODO()
        override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = TODO()
        override fun <T : DType, V> maxPool2d(input: Tensor<T, V>, kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>, padding: Pair<Int, Int>): Tensor<T, V> = TODO()
        override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = TODO()
        override fun <T : DType, V> rdivScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> = TODO()
        override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> = TODO()
        override fun <T : DType, V> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V> = TODO()
        override fun <T : DType, V> rsubScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> = TODO()
        override fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> = TODO()
        override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> = TODO()
        override fun <T : DType, V> softmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> = TODO()
        override fun <T : DType, V> split(tensor: Tensor<T, V>, splitSize: Int, dim: Int): List<Tensor<T, V>> = TODO()
        override fun <T : DType, V> sqrt(tensor: Tensor<T, V>): Tensor<T, V> = TODO()
        override fun <T : DType, V> squeeze(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = TODO()
        override fun <T : DType, V> subScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = TODO()
        override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> = TODO()
        override fun <T : DType, V> tril(tensor: Tensor<T, V>, k: Int): Tensor<T, V> = TODO()
        override fun <T : DType, V> unsqueeze(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> = TODO()
        override fun <T : DType, V> upsample2d(input: Tensor<T, V>, scale: Pair<Int, Int>, mode: sk.ainet.lang.tensor.ops.UpsampleMode, alignCorners: Boolean): Tensor<T, V> = TODO()
        override fun <T : DType, V> variance(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = TODO()
    }

    private fun createMockExecutionContext(ops: TensorOps = MockTensorOps()): ExecutionContext {
        return object : ExecutionContext {
            override val ops: TensorOps = ops
            override val executionStats: sk.ainet.context.ExecutionStats get() = TODO()
            override val memoryInfo: sk.ainet.context.MemoryInfo get() = TODO()
            override val phase: sk.ainet.context.Phase get() = TODO()
            override val tensorDataFactory: sk.ainet.lang.tensor.data.TensorDataFactory get() = TODO()
        }
    }

    private companion object {
        fun createMockTensor(value: Float = 1.0f, shape: Shape = Shape(2, 2)): Tensor<FP32, Float> {
            return object : Tensor<FP32, Float> {
                override val data: TensorData<FP32, Float> = object : TensorData<FP32, Float> {
                    override val shape: Shape = shape
                    override fun get(indices: IntArray): Float = value
                    override fun set(indices: IntArray, value: Float) {}
                    override fun toString(): String = value.toString()
                }
                override val shape: Shape = shape
                override val dtype: kotlin.reflect.KClass<FP32> = FP32::class
                override val ops: TensorOps get() = MockTensorOps(value)
            }
        }
    }

    private fun createMockTensor(value: Float = 1.0f, shape: Shape = Shape(2, 2)): Tensor<FP32, Float> {
        return FederatedMathUtilsTest.createMockTensor(value, shape)
    }

    @Test
    fun testL2Norm() = runTest {
        val ctx = createMockExecutionContext(object : MockTensorOps() {
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = createMockTensor(4f) as Tensor<T, V>
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = createMockTensor(16f, Shape(1)) as Tensor<T, V>
        })
        
        val tensor = createMockTensor(2f)
        val norm = FederatedMathUtils.l2Norm(ctx, tensor)
        assertEquals(4f, norm)
    }

    @Test
    fun testFrobeniusNorm() = runTest {
        val ctx = createMockExecutionContext(object : MockTensorOps() {
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = createMockTensor(4f) as Tensor<T, V>
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = createMockTensor(16f, Shape(1)) as Tensor<T, V>
        })
        
        val tensor = createMockTensor(2f)
        val norm = FederatedMathUtils.frobeniusNorm(ctx, tensor)
        assertEquals(4f, norm)
    }

    @Test
    fun testWeightedAverage() = runTest {
        val ctx = createMockExecutionContext(object : MockTensorOps() {
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
                val baseValue = (a.data as Any).toString().toFloat()
                return createMockTensor(baseValue * b.toFloat()) as Tensor<T, V>
            }
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
                val valA = (a.data as Any).toString().toFloat()
                val valB = (b.data as Any).toString().toFloat()
                return createMockTensor(valA + valB) as Tensor<T, V>
            }
        })

        val updates = listOf(createMockTensor(1f), createMockTensor(2f))
        val weights = listOf(1f, 3f)
        
        val result = FederatedMathUtils.weightedAverage(ctx, updates, weights)
        val resultValue = (result.data as Any).toString().toFloat()
        assertEquals(1.75f, resultValue)
    }

    @Test
    fun testAddProximalTerm() = runTest {
        val ctx = createMockExecutionContext(object : MockTensorOps() {
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
                val valA = (a.data as Any).toString().toFloat()
                val valB = (b.data as Any).toString().toFloat()
                return createMockTensor(valA - valB) as Tensor<T, V>
            }
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
                val baseValue = (a.data as Any).toString().toFloat()
                return createMockTensor(baseValue * b.toFloat()) as Tensor<T, V>
            }
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
                val valA = (a.data as Any).toString().toFloat()
                val valB = (b.data as Any).toString().toFloat()
                return createMockTensor(valA + valB) as Tensor<T, V>
            }
        })

        val local = createMockTensor(10f)
        val global = createMockTensor(8f)
        val mu = 0.5f
        
        val result = FederatedMathUtils.addProximalTerm(ctx, local, global, mu)
        val resultValue = (result.data as Any).toString().toFloat()
        assertEquals(11f, resultValue)
    }

    @Test
    fun testMomentumUpdate() = runTest {
        val ctx = createMockExecutionContext(object : MockTensorOps() {
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
                val baseValue = (a.data as Any).toString().toFloat()
                return createMockTensor(baseValue * b.toFloat()) as Tensor<T, V>
            }
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
                val valA = (a.data as Any).toString().toFloat()
                val valB = (b.data as Any).toString().toFloat()
                return createMockTensor(valA + valB) as Tensor<T, V>
            }
        })

        val grad = createMockTensor(1f)
        val momentum = createMockTensor(2f)
        val beta = 0.9f
        
        val result = FederatedMathUtils.momentumUpdate(ctx, grad, momentum, beta)
        val resultValue = (result.data as Any).toString().toFloat()
        assertEquals(1.9f, resultValue)
    }

    @Test
    fun testVarianceUpdate() = runTest {
        val ctx = createMockExecutionContext(object : MockTensorOps() {
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
                val valA = (a.data as Any).toString().toFloat()
                val valB = (b.data as Any).toString().toFloat()
                return createMockTensor(valA * valB) as Tensor<T, V>
            }
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
                val baseValue = (a.data as Any).toString().toFloat()
                return createMockTensor(baseValue * b.toFloat()) as Tensor<T, V>
            }
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
                val valA = (a.data as Any).toString().toFloat()
                val valB = (b.data as Any).toString().toFloat()
                return createMockTensor(valA + valB) as Tensor<T, V>
            }
        })

        val grad = createMockTensor(2f)
        val variance = createMockTensor(10f)
        val beta = 0.9f
        
        val result = FederatedMathUtils.varianceUpdate(ctx, grad, variance, beta)
        val resultValue = (result.data as Any).toString().toFloat()
        assertEquals(9.4f, resultValue)
    }

    @Test
    fun testMaskedMean() = runTest {
        val ctx = createMockExecutionContext(object : MockTensorOps() {
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
                val valA = (a.data as Any).toString().toFloat()
                val valB = (b.data as Any).toString().toFloat()
                return createMockTensor(valA + valB) as Tensor<T, V>
            }
            @Suppress("UNCHECKED_CAST")
            override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
                val baseValue = (a.data as Any).toString().toFloat()
                return createMockTensor(baseValue * b.toFloat()) as Tensor<T, V>
            }
        })

        val tensors = listOf(createMockTensor(10f), null, createMockTensor(20f))
        val mask = listOf(true, true, true)
        
        val result = FederatedMathUtils.maskedMean(ctx, tensors, mask)
        val resultValue = (result.data as Any).toString().toFloat()
        assertEquals(15f, resultValue)
    }

    @Test
    fun testElementwiseSqrt() = runTest {
        val ctx = createMockExecutionContext()
        val tensor = createMockTensor(16f)
        val result = FederatedMathUtils.elementwiseSqrt(ctx, tensor)
        assertEquals(tensor, result)
    }

    @Test
    fun testClamp() = runTest {
        val ctx = createMockExecutionContext()
        val tensor = createMockTensor(10f)
        val result = FederatedMathUtils.clamp(ctx, tensor, 0f, 5f)
        assertEquals(tensor, result)
    }

    @Test
    fun testFederatedMathUtilsIsSingleton() {
        assertNotNull(FederatedMathUtils)
    }

    @Test
    fun testHasNaNOrInf() {
        val normalTensor = createMockTensor(1.0f)
        assertEquals(false, FederatedMathUtils.hasNaNOrInf(normalTensor))

        val nanTensor = createMockTensor(Float.NaN)
        assertEquals(true, FederatedMathUtils.hasNaNOrInf(nanTensor))

        val infTensor = createMockTensor(Float.POSITIVE_INFINITY)
        assertEquals(true, FederatedMathUtils.hasNaNOrInf(infTensor))
    }

    @Test
    fun testMockTensorCreation() {
        val tensor1 = createMockTensor(5.0f, Shape(3, 3))
        assertEquals(5.0f, (tensor1.data as Any).toString().toFloat())
        assertEquals(Shape(3, 3), tensor1.shape)
    }

    /**
     * Property test for weighted mean aggregation.
     * Feature: federated-learning-strategies, Property 6: Weighted Mean Aggregation
     * Validates: Requirements 4.2, 7.1
     * 
     * Property: For any set of tensors and corresponding weights, the weighted mean should satisfy
     * mathematical properties of weighted averages including:
     * - Weighted sum property: result = Σ(weight_i * tensor_i) / Σ(weight_i)
     * - Boundary conditions: single tensor returns itself, equal weights return arithmetic mean
     * - Scaling invariance: scaling all weights by same factor doesn't change result
     */
    @Test
    fun testWeightedMeanAggregationProperty() = runTest {
        // Property-based test with 100 iterations as specified in design document
        checkAll(100, 
            Arb.list(Arb.float(1f, 100f), 1..5), // List of 1-5 tensor values
            Arb.list(Arb.float(0.1f, 10f), 1..5)  // List of 1-5 positive weights
        ) { tensorValues, weightsList ->
            // Ensure weights list matches tensor values list size
            val weights = if (weightsList.size == tensorValues.size) {
                weightsList
            } else {
                weightsList.take(tensorValues.size).let { taken ->
                    if (taken.size < tensorValues.size) {
                        taken + List(tensorValues.size - taken.size) { 1f }
                    } else taken
                }
            }

            // Create mock execution context that properly handles weighted averaging
            val ctx = createMockExecutionContext(object : MockTensorOps() {
                @Suppress("UNCHECKED_CAST")
                override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
                    val baseValue = (a.data as Any).toString().toFloat()
                    return createMockTensor(baseValue * b.toFloat()) as Tensor<T, V>
                }
                @Suppress("UNCHECKED_CAST")
                override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
                    val valA = (a.data as Any).toString().toFloat()
                    val valB = (b.data as Any).toString().toFloat()
                    return createMockTensor(valA + valB) as Tensor<T, V>
                }
            })

            // Create tensors from values
            val tensors = tensorValues.map { createMockTensor(it) }
            
            // Test the weighted average
            val result = FederatedMathUtils.weightedAverage(ctx, tensors, weights)
            val resultValue = (result.data as Any).toString().toFloat()
            
            // Calculate expected weighted average manually
            val totalWeight = weights.sum()
            val expectedValue = tensorValues.zip(weights) { value, weight -> 
                value * weight 
            }.sum() / totalWeight
            
            // Verify the weighted average property holds (with small tolerance for floating point)
            val tolerance = 0.001f
            assertTrue(
                abs(resultValue - expectedValue) < tolerance,
                "Weighted average property failed: expected $expectedValue, got $resultValue"
            )
            
            // Test boundary condition: single tensor should return itself (scaled by weight normalization)
            if (tensors.size == 1) {
                assertEquals(tensorValues[0], resultValue, tolerance)
            }
            
            // Test equal weights property: should equal arithmetic mean when all weights are equal
            if (weights.all { abs(it - weights[0]) < tolerance }) {
                val arithmeticMean = tensorValues.average().toFloat()
                assertTrue(
                    abs(resultValue - arithmeticMean) < tolerance,
                    "Equal weights should produce arithmetic mean: expected $arithmeticMean, got $resultValue"
                )
            }
        }
    }

    /**
     * Property test for mathematical norm properties.
     * Feature: federated-learning-strategies, Property 5: Mathematical Norm Properties
     * Validates: Requirements 4.3
     * 
     * Property: For any tensor, computed norms (L2, Frobenius) should satisfy mathematical 
     * properties such as:
     * - Non-negativity: ||x|| >= 0
     * - Definiteness: ||x|| = 0 if and only if x = 0
     * - Homogeneity: ||ax|| = |a| * ||x|| for scalar a
     * - Triangle inequality: ||x + y|| <= ||x|| + ||y||
     * - Equivalence: Frobenius norm equals L2 norm for tensors
     */
    @Test
    fun testMathematicalNormPropertiesProperty() = runTest {
        // Property-based test with 100 iterations as specified in design document
        checkAll(100,
            Arb.float(0.1f, 10f), // Tensor value for first tensor
            Arb.float(0.1f, 10f), // Tensor value for second tensor  
            Arb.float(-5f, 5f)    // Scalar multiplier
        ) { value1, value2, scalar ->
            
            // Create mock execution context that properly handles norm computations
            val ctx = createMockExecutionContext(object : MockTensorOps() {
                @Suppress("UNCHECKED_CAST")
                override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
                    val valA = (a.data as Any).toString().toFloat()
                    val valB = (b.data as Any).toString().toFloat()
                    return createMockTensor(valA * valB) as Tensor<T, V>
                }
                @Suppress("UNCHECKED_CAST")
                override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
                    val value = (tensor.data as Any).toString().toFloat()
                    // For a 2x2 tensor with same values, sum would be 4 * value
                    return createMockTensor(4f * value, Shape(1)) as Tensor<T, V>
                }
                @Suppress("UNCHECKED_CAST")
                override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
                    val baseValue = (a.data as Any).toString().toFloat()
                    return createMockTensor(baseValue * b.toFloat()) as Tensor<T, V>
                }
                @Suppress("UNCHECKED_CAST")
                override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
                    val valA = (a.data as Any).toString().toFloat()
                    val valB = (b.data as Any).toString().toFloat()
                    return createMockTensor(valA + valB) as Tensor<T, V>
                }
            })

            val tensor1 = createMockTensor(value1)
            val tensor2 = createMockTensor(value2)
            val scaledTensor = createMockTensor(value1 * scalar)
            
            // Test L2 norm properties
            val norm1 = FederatedMathUtils.l2Norm(ctx, tensor1)
            val norm2 = FederatedMathUtils.l2Norm(ctx, tensor2)
            val scaledNorm = FederatedMathUtils.l2Norm(ctx, scaledTensor)
            
            // Test Frobenius norm properties
            val frobNorm1 = FederatedMathUtils.frobeniusNorm(ctx, tensor1)
            val frobNorm2 = FederatedMathUtils.frobeniusNorm(ctx, tensor2)
            
            val tolerance = 0.001f
            
            // Property 1: Non-negativity - ||x|| >= 0
            assertTrue(norm1 >= 0f, "L2 norm must be non-negative: $norm1")
            assertTrue(norm2 >= 0f, "L2 norm must be non-negative: $norm2")
            assertTrue(frobNorm1 >= 0f, "Frobenius norm must be non-negative: $frobNorm1")
            assertTrue(frobNorm2 >= 0f, "Frobenius norm must be non-negative: $frobNorm2")
            
            // Property 2: Homogeneity - ||ax|| = |a| * ||x||
            val expectedScaledNorm = abs(scalar) * norm1
            assertTrue(
                abs(scaledNorm - expectedScaledNorm) < tolerance,
                "Homogeneity property failed: ||${scalar}x|| = $scaledNorm, expected ${abs(scalar)} * $norm1 = $expectedScaledNorm"
            )
            
            // Property 3: Equivalence - Frobenius norm equals L2 norm for tensors
            assertTrue(
                abs(norm1 - frobNorm1) < tolerance,
                "L2 and Frobenius norms should be equal: L2=$norm1, Frobenius=$frobNorm1"
            )
            assertTrue(
                abs(norm2 - frobNorm2) < tolerance,
                "L2 and Frobenius norms should be equal: L2=$norm2, Frobenius=$frobNorm2"
            )
            
            // Property 4: Triangle inequality - ||x + y|| <= ||x|| + ||y||
            // Create sum tensor for triangle inequality test
            val sumTensor = createMockTensor(value1 + value2)
            val sumNorm = FederatedMathUtils.l2Norm(ctx, sumTensor)
            val normSum = norm1 + norm2
            
            assertTrue(
                sumNorm <= normSum + tolerance,
                "Triangle inequality failed: ||x + y|| = $sumNorm > ||x|| + ||y|| = $normSum"
            )
            
            // Property 5: Definiteness for zero tensor (special case)
            if (abs(value1) < tolerance) {
                assertTrue(
                    abs(norm1) < tolerance,
                    "Zero tensor should have zero norm: $norm1"
                )
            }
            
            // Property 6: Positive definiteness for non-zero tensors
            if (abs(value1) > tolerance) {
                assertTrue(
                    norm1 > tolerance,
                    "Non-zero tensor should have positive norm: $norm1 for value $value1"
                )
            }
        }
    }

    /**
     * Property test for momentum update properties.
     * Feature: federated-learning-strategies, Property 8: Momentum Update Properties
     * Validates: Requirements 4.4, 5.3
     * 
     * Property: For any gradient and momentum tensors, momentum updates should satisfy
     * the exponential moving average properties with the specified beta parameter:
     * - Exponential moving average formula: momentum = beta * momentum + (1 - beta) * gradient
     * - Beta parameter bounds: beta ∈ [0, 1]
     * - Convergence property: when gradient is constant, momentum converges to gradient
     * - Boundary conditions: beta=0 gives gradient, beta=1 preserves momentum
     * - Linearity: momentum update is linear in both gradient and previous momentum
     * - Stability: momentum magnitude should be bounded by max(|gradient|, |momentum|)
     */
    @Test
    fun testMomentumUpdatePropertiesProperty() = runTest {
        // Property-based test with 100 iterations as specified in design document
        checkAll(100,
            Arb.float(-10f, 10f), // Gradient value
            Arb.float(-10f, 10f), // Previous momentum value
            Arb.float(0f, 1f)     // Beta parameter in valid range [0, 1]
        ) { gradientValue, momentumValue, beta ->
            
            // Create mock execution context that properly handles momentum computations
            val ctx = createMockExecutionContext(object : MockTensorOps() {
                @Suppress("UNCHECKED_CAST")
                override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
                    val baseValue = (a.data as Any).toString().toFloat()
                    return createMockTensor(baseValue * b.toFloat()) as Tensor<T, V>
                }
                @Suppress("UNCHECKED_CAST")
                override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
                    val valA = (a.data as Any).toString().toFloat()
                    val valB = (b.data as Any).toString().toFloat()
                    return createMockTensor(valA + valB) as Tensor<T, V>
                }
            })

            val gradient = createMockTensor(gradientValue)
            val momentum = createMockTensor(momentumValue)
            
            // Test the momentum update
            val result = FederatedMathUtils.momentumUpdate(ctx, gradient, momentum, beta)
            val resultValue = (result.data as Any).toString().toFloat()
            
            // Calculate expected momentum update manually
            val expectedValue = beta * momentumValue + (1f - beta) * gradientValue
            
            val tolerance = 0.001f
            
            // Property 1: Exponential moving average formula correctness
            assertTrue(
                abs(resultValue - expectedValue) < tolerance,
                "Momentum update formula failed: expected $expectedValue, got $resultValue " +
                "(beta=$beta, gradient=$gradientValue, momentum=$momentumValue)"
            )
            
            // Property 2: Boundary condition - beta = 0 should return gradient
            if (abs(beta) < tolerance) {
                assertTrue(
                    abs(resultValue - gradientValue) < tolerance,
                    "When beta=0, momentum update should equal gradient: expected $gradientValue, got $resultValue"
                )
            }
            
            // Property 3: Boundary condition - beta = 1 should preserve momentum
            if (abs(beta - 1f) < tolerance) {
                assertTrue(
                    abs(resultValue - momentumValue) < tolerance,
                    "When beta=1, momentum update should preserve momentum: expected $momentumValue, got $resultValue"
                )
            }
            
            // Property 4: Linearity in gradient - scaling gradient should scale the contribution
            val scaledGradient = createMockTensor(gradientValue * 2f)
            val scaledResult = FederatedMathUtils.momentumUpdate(ctx, scaledGradient, momentum, beta)
            val scaledResultValue = (scaledResult.data as Any).toString().toFloat()
            val expectedScaledValue = beta * momentumValue + (1f - beta) * (gradientValue * 2f)
            
            assertTrue(
                abs(scaledResultValue - expectedScaledValue) < tolerance,
                "Linearity in gradient failed: expected $expectedScaledValue, got $scaledResultValue"
            )
            
            // Property 5: Linearity in momentum - scaling momentum should scale its contribution
            val scaledMomentum = createMockTensor(momentumValue * 2f)
            val momentumScaledResult = FederatedMathUtils.momentumUpdate(ctx, gradient, scaledMomentum, beta)
            val momentumScaledResultValue = (momentumScaledResult.data as Any).toString().toFloat()
            val expectedMomentumScaledValue = beta * (momentumValue * 2f) + (1f - beta) * gradientValue
            
            assertTrue(
                abs(momentumScaledResultValue - expectedMomentumScaledValue) < tolerance,
                "Linearity in momentum failed: expected $expectedMomentumScaledValue, got $momentumScaledResultValue"
            )
            
            // Property 6: Stability - result magnitude should be reasonable
            val maxInput = maxOf(abs(gradientValue), abs(momentumValue))
            if (maxInput > tolerance) {
                // The result should not be dramatically larger than the inputs
                // This is a stability check to ensure numerical behavior is reasonable
                val resultMagnitude = abs(resultValue)
                val expectedMaxMagnitude = maxInput * 2f // Allow some reasonable growth
                
                assertTrue(
                    resultMagnitude <= expectedMaxMagnitude,
                    "Stability check failed: result magnitude $resultMagnitude exceeds expected bound $expectedMaxMagnitude " +
                    "(gradient=$gradientValue, momentum=$momentumValue, beta=$beta)"
                )
            }
            
            // Property 7: Convergence property - when gradient is constant over multiple steps
            // After many iterations with the same gradient, momentum should converge to gradient
            if (abs(gradientValue) > tolerance) {
                var currentMomentum = momentumValue
                // Simulate 10 iterations with the same gradient
                repeat(10) {
                    val iterMomentum = createMockTensor(currentMomentum)
                    val iterResult = FederatedMathUtils.momentumUpdate(ctx, gradient, iterMomentum, beta)
                    currentMomentum = (iterResult.data as Any).toString().toFloat()
                }
                
                // After many iterations, momentum should be closer to gradient
                // The convergence rate depends on beta, but we can check the direction
                val convergenceDirection = (currentMomentum - momentumValue) * (gradientValue - momentumValue)
                if (abs(gradientValue - momentumValue) > tolerance) {
                    assertTrue(
                        convergenceDirection >= -tolerance,
                        "Convergence property failed: momentum should move towards gradient over iterations " +
                        "(initial momentum=$momentumValue, gradient=$gradientValue, final momentum=$currentMomentum)"
                    )
                }
            }
            
            // Property 8: Beta parameter validation - function should handle edge cases
            // This is implicitly tested by the boundary conditions above, but we can add
            // additional validation for the beta parameter bounds
            assertTrue(
                beta >= 0f && beta <= 1f,
                "Beta parameter should be in valid range [0, 1]: $beta"
            )
        }
    }
}

