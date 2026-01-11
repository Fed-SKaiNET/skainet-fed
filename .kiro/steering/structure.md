# Project Structure & Organization

## Root Level Structure

```
SKaiNET-FED/
├── skainet-fed/                    # Main federated learning modules
│   └── skainet-fed-core/          # Core FL implementation
├── build-logic/                   # Custom Gradle build logic
├── docs/                          # Documentation
├── gradle/                        # Gradle wrapper and version catalog
└── .kiro/                         # Kiro IDE configuration
```

## Core Module: `skainet-fed-core`

The primary implementation module with Kotlin Multiplatform structure:

```
skainet-fed-core/
├── src/
│   ├── commonMain/kotlin/          # Shared cross-platform code
│   │   └── sk/ainet/fed/core/
│   │       └── strategy/           # FL strategy implementations
│   ├── commonTest/kotlin/          # Shared tests
│   ├── androidMain/kotlin/         # Android-specific code
│   ├── iosMain/kotlin/             # iOS-specific code
│   └── jvmMain/kotlin/             # JVM-specific code
└── build.gradle.kts               # Module build configuration
```

## Package Organization

### Core Package Structure
- `sk.ainet.fed.core`: Root package for federated learning core
- `sk.ainet.fed.core.strategy`: Federated learning strategy implementations

### Strategy Implementation Pattern
Each FL strategy follows a consistent structure:
- **Base Strategy**: `base.py` - Foundation FedAvg implementation
- **Specialized Strategies**: Individual files for each algorithm (FedProx, FedTrend, etc.)
- **Mathematical Focus**: Pure mathematical implementations using SKaiNET Tensor API

## File Naming Conventions

### Kotlin Files
- Use PascalCase for class names: `FederatedStrategy.kt`
- Use camelCase for function and property names
- Interface names should be descriptive: `StrategyAggregator`

### Python Strategy Files (Legacy/Reference)
- Use snake_case: `fed_prox.py`, `fed_trend.py`
- Include strategy name in filename for clarity
- These serve as mathematical reference implementations

## Configuration Files

### Build Configuration
- `build.gradle.kts`: Root project configuration
- `settings.gradle.kts`: Project structure definition
- `gradle/libs.versions.toml`: Centralized dependency management
- `gradle.properties`: Global Gradle properties

### Module Configuration
- Each module has its own `build.gradle.kts`
- Module-specific `gradle.properties` for local settings
- Explicit API mode enabled for all modules

## Documentation Structure

### Technical Documentation
- `docs/skainet-core-tech.md`: SKaiNET Tensor API guide
- `README.md`: Project overview and getting started
- Generated docs via Dokka and custom documentation plugin

### Specifications
- `.kiro/specs/`: Feature specifications and design documents
- Structured as requirements, design, and tasks

## Development Patterns

### Source Set Organization
- **commonMain**: All mathematical FL logic and core interfaces
- **Platform-specific**: Only for platform integration, not business logic
- **Test organization**: Mirror main source structure in test directories

### Dependency Management
- All dependencies declared in `libs.versions.toml`
- SKaiNET libraries as primary dependencies
- Minimal external dependencies to reduce complexity

### Code Generation
- KSP-generated documentation from operator definitions
- Custom plugins for specialized documentation needs
- Incremental build support with proper caching