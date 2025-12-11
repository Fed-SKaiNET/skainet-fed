# FED-SKaiNET

**FED-SKaiNET** is an open-source library that enables privacy-preserving **Federated Learning on mobile devices**.  
It allows developers — regardless of prior ML/AI expertise — to integrate decentralized AI training directly into native mobile apps for **Android and iOS**.

Instead of sending sensitive user data to cloud servers, models are trained **locally on each device**.  
Only **anonymized model updates** are exchanged to create a shared global model, ensuring that **private data never leaves the device**.

FED-SKaiNET is built using **Kotlin Multiplatform**, allowing extensive **code sharing across platforms** and enabling seamless integration into existing native mobile projects.

The project builds upon the on-device ML framework **[SKaiNET](https://github.com/sk-ai-net/SKaiNET)** and aims to deliver a production-ready federated learning solution with a strong focus on:

- 🔒 **Data protection and digital sovereignty**  
- 🌍 **Decentralized AI training**  
- 📱 **Mobile-first design**  
- 🧩 **Easy developer integration**

---

## 🚀 Core Features

- Federated Learning **directly on mobile devices**
- Full **privacy by design** (no raw data upload)
- **Platform-independent** development via Kotlin Multiplatform
- Native integration for **Android and iOS**
- Open-source, industry-friendly licensing
- Contribution to the broader **SKaiNET ecosystem**

---

## 🛠️ Technology Stack

FED-SKaiNET is built on modern, open technologies:

### Programming & Frameworks
- **Kotlin**
- **Kotlin Multiplatform** for cross-platform shared code
- **SKaiNET Framework** for on-device machine learning

### Federated Learning & Aggregation
- Integration with existing open FL solutions:
  - **OpenFL**
  - **Flower AI**
  - **FedCast**

### Infrastructure & Tooling
- **GitHub** for open development
- **GitHub Actions** for CI/CD automation
- **Docker** for containerized training hubs
- Federated Hub services for aggregation and orchestration

### Platforms
- **Android**
- **iOS**

---

## 🧩 Architecture Overview

```mermaid
flowchart LR
    subgraph Devices["Mobile Devices"]
        subgraph Android["Android"]
            AApp[Android App]
            AFED[FED-SKaiNET SDK]
            ASK[SKaiNET Runtime]
        end
        subgraph iOS["iOS"]
            IApp[iOS App]
            IFED[FED-SKaiNET SDK]
            ISK[SKaiNET Runtime]
        end
    end

    subgraph SharedCore["Kotlin Multiplatform Shared Core"]
        Core["FL logic & client\n(networking, training pipeline)"]
    end

    subgraph Hub["Federated Learning Hub"]
        Orchestrator["Orchestrator\n(Flower / OpenFL / FedCast)"]
        Aggregator["Aggregation Service\n(FedAvg, etc.)"]
        GlobalModel[Global Model]
    end

    AApp --> AFED --> ASK
    IApp --> IFED --> ISK

    AFED --- Core
    IFED --- Core

    ASK -->|local training\non-device data| U1[(Model Update)]
    ISK -->|local training\non-device data| U2[(Model Update)]

    U1 -->|secure, anonymized| Orchestrator
    U2 -->|secure, anonymized| Orchestrator

    Orchestrator --> Aggregator --> GlobalModel
    GlobalModel -->|updated weights| AFED
    GlobalModel -->|updated weights| IFED
