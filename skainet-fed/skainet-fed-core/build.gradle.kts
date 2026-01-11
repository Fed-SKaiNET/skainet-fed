import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.vanniktech.mavenPublish)
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.dokka)
}

kotlin {
    explicitApi()

    androidTarget {
        @OptIn(ExperimentalKotlinGradlePluginApi::class)
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosArm64()
    iosSimulatorArm64()
    macosArm64 ()
    linuxX64 ()
    linuxArm64 ()

    jvm()

    js {
        browser()
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    sourceSets {
        commonMain.dependencies {
            implementation(libs.skainet.lang.core)
            implementation(libs.skainet.lang.models)
            implementation(libs.skainet.model.yolo)
            implementation(libs.skainet.lang.kan)
            implementation(libs.skainet.compile.core)
            implementation(libs.skainet.backend.cpu)
            implementation(libs.skainet.data.api)
            implementation(libs.skainet.data.simple)
            implementation(libs.skainet.io.core)
            implementation(libs.skainet.io.gguf)
            implementation(libs.skainet.io.onnx)
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
            implementation(libs.kotlinx.coroutines.test)
//            implementation(libs.kotest.assertions.core)
            //implementation(libs.kotest.property)
        }
    }
}

android {
    namespace = "sk.ainet.fed.core"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        minSdk = libs.versions.android.minSdk.get().toInt()
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}