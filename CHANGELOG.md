# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-01-11

### Added
- **Federated Core API**: Initial implementation of the core federated learning concepts.
- **FedAvg Strategy**: Foundational Federated Averaging algorithm for weighted model aggregation.
- **Parameter Management**: `ParameterManager` for handling global model states and weight extractions.
- **Federated Math Utilities**: Optimized tensor operations for weighted averaging and metric aggregation.
- **Architecture Documentation**: Comprehensive overview of the 3-layer stack (Core, Transport, Coordinator) and dataflow.
- **On-Device Focus**: Support for local training pipelines with a focus on privacy and data anonymization.
- **Kotlin Multiplatform Support**: Base structure for JVM and Android targets.
- Refined `README.md` to reflect the Federated Learning vision and SKaiNET integration.
- Initial project structure and build configuration for KMP.
