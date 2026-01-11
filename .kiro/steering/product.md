# Product Overview

**FED-SKaiNET** is an open-source Kotlin Multiplatform library that enables privacy-preserving **Federated Learning on mobile devices** (Android and iOS).

## Core Value Proposition

- **Privacy by Design**: Models train locally on devices; only anonymized model updates are shared
- **Mobile-First**: Native integration for Android and iOS apps
- **Developer-Friendly**: Designed for developers without prior ML/AI expertise
- **Cross-Platform**: Extensive code sharing via Kotlin Multiplatform

## Key Features

- Federated Learning directly on mobile devices
- Full data protection and digital sovereignty
- Platform-independent development
- Integration with existing FL solutions (OpenFL, Flower AI, FedCast)
- Built on the SKaiNET on-device ML framework

## Architecture

The system consists of:
- **Mobile SDK**: FED-SKaiNET SDK with SKaiNET Runtime
- **Shared Core**: Kotlin Multiplatform shared FL logic and client networking
- **Federated Hub**: Orchestrator and aggregation services for global model management

## Current Focus

The project is implementing "The Federated Calculator" - an MVP focused on robust, bit-perfect implementation of federated learning strategies, starting with **FedAvg** using SKaiNET's Tensor API.