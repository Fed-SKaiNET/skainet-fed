[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![Maven Central](https://img.shields.io/maven-central/v/sk.ainet.fed/skainet-fed-core.svg)](https://central.sonatype.com/artifact/sk.ainet.core/skainet-lang-core)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-View%20Docs-blue?logo=readthedocs&logoColor=white)](https://deepwiki.com/Fed-SKaiNET/skainet-fed)


# Federated Learning with SKaiNET



SKaiNET is a Kotlin Multiplatform deep learning framework designed with a **device-first philosophy** and efficient, portable execution across JVM, Android and other targets. :contentReference[oaicite:0]{index=0}  
This makes it a natural fit for **on-device AI** scenarios where models run close to the data instead of in a central cloud.

Federated learning extends this vision by allowing many devices to **train a shared model collaboratively** while **keeping raw data local and private**.

---

## Why federated learning?

Traditional training:

- Collects user data on a central server
- Trains a model in one place
- Is often not acceptable for privacy-sensitive use cases

Federated learning:

- Runs training **on each device** using SKaiNET’s normal training APIs
- Sends only **model updates (weights/gradients)** to a coordinator
- Aggregates updates (e.g. FedAvg) into a new global model
- Never sends raw user data off-device

This matches SKaiNET’s goals of:

- On-device / edge AI
- Distributed computing and training capabilities :contentReference[oaicite:1]{index=1}
- Decentralized training with privacy-preserving data handling

---

## Documentation

For a deeper dive into how SKaiNET handles federated learning, check out the [Architecture Documentation](docs/architecture.md). It covers:

- The three-layer stack (Core, Transport, Coordinator)
- Dataflow and training pipelines
- On-device privacy and anonymization strategies

---

## Current Status (v0.0.1-alpha)

We are currently in the initial development phase. The current focus is on building the **Federated Core** and standardizing aggregation strategies.

### What's implemented:
- **Foundational Core API**: Basic interfaces for `FederatedStrategy` and model management.
- **FedAvg Implementation**: The primary Federated Averaging algorithm is ready for use.
- **KMP Support**: Core logic is cross-platform (JVM/Android).
- **Architecture**: A clear roadmap for Transport and Coordinator layers.

### Next steps:
- Implementation of Ktor-based Transport layer.
- Reference JVM Coordinator.
- Integration with Android-specific background training tasks.

