package sk.ainet.fed.core.types

import kotlin.reflect.KClass

/**
 * Temporary placeholder types until we resolve the correct SKaiNET imports.
 * These will be replaced with actual SKaiNET types once the import paths are corrected.
 */

// Placeholder for SKaiNET ExecutionContext
public interface ExecutionContext

// Placeholder for SKaiNET Tensor
public interface Tensor<T : DType, V>

// Placeholder for SKaiNET DType
public interface DType

// Placeholder for SKaiNET FP32
public class FP32 : DType

// Placeholder for SKaiNET Shape
public data class Shape(val dimensions: IntArray) {
    public companion object {
        public fun of(vararg dims: Int): Shape = Shape(dims)
    }
}