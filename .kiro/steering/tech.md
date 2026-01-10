# Technology Stack & Build System

## Core Technologies

### Programming Languages & Frameworks
- **Kotlin Multiplatform**: Primary development language with cross-platform code sharing
- **Kotlin**: Version 2.2.21 with explicit API mode enabled
- **SKaiNET Framework**: On-device ML framework for tensor operations and model execution

### Target Platforms
- **Android**: Min SDK 24, Compile SDK 36, JVM Target 11
- **iOS**: ARM64 and Simulator ARM64
- **macOS**: ARM64
- **Linux**: x64 and ARM64
- **JVM**: For server-side components
- **JavaScript/WASM**: Browser support

### Build System
- **Gradle**: Primary build system with Kotlin DSL
- **JVM Toolchain**: Java 21 enforced across all modules
- **KSP**: Kotlin Symbol Processing for code generation

## Key Dependencies

### SKaiNET Core Libraries
- `skainet-lang-core`: Core tensor and execution context APIs
- `skainet-lang-models`: Pre-built model definitions
- `skainet-backend-cpu`: CPU execution backend
- `skainet-data-api`: Data handling abstractions
- `skainet-io-*`: Model format support (GGUF, ONNX)

### Networking & Serialization
- **Ktor**: HTTP client for federated communication (v3.3.3)
- **kotlinx-serialization**: JSON serialization (v1.9.0)
- **kotlinx-coroutines**: Async programming (v1.10.2)

### Development Tools
- **Kover**: Code coverage reporting
- **Binary Compatibility Validator**: API stability
- **Dokka**: Documentation generation
- **KotlinPoet**: Code generation utilities

## Common Build Commands

```bash
# Build all modules
./gradlew build

# Run tests
./gradlew test

# Generate documentation
./gradlew generateOperatorDocs

# Code coverage report
./gradlew koverHtmlReport

# Clean build
./gradlew clean build

# Publish to local repository
./gradlew publishToMavenLocal
```

## Architecture Patterns

### Tensor Operations
- Use `ExecutionContext` for all tensor creation and operations
- Leverage `TensorOps` interface for mathematical computations
- Prefer `sliceView` over `sliceCopy` for memory efficiency
- Always specify explicit `DType` classes (e.g., `FP32::class`)

### Multiplatform Structure
- **commonMain**: Shared business logic and mathematical operations
- **Platform-specific**: Only for platform-specific integrations
- **Explicit API**: All public APIs must be explicitly declared

### Code Generation
- KSP-based operator documentation generation
- Custom documentation plugin for API reference
- Incremental build support with proper input/output configuration