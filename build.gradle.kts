plugins {
    alias(libs.plugins.androidLibrary) apply false
    alias(libs.plugins.kotlinMultiplatform) apply  false
    alias(libs.plugins.jetbrainsKotlinJvm) apply false
    alias(libs.plugins.vanniktech.mavenPublish) apply false
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator) apply false
    alias(libs.plugins.ksp) apply false
    alias(libs.plugins.asciidoctorJvm) apply false
    alias(libs.plugins.dokka) apply false
    alias(libs.plugins.skainet.docs)
}

allprojects {
    group = "sk.ainet"
}

// Enforce a consistent JVM toolchain across all Kotlin modules to avoid bytecode mismatches
subprojects {
    // Kotlin Multiplatform projects
    plugins.withId("org.jetbrains.kotlin.multiplatform") {
        extensions.findByType(org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension::class.java)?.apply {
            jvmToolchain(21)
        }
    }
    // Kotlin/JVM projects
    plugins.withId("org.jetbrains.kotlin.jvm") {
        extensions.findByType(org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension::class.java)?.apply {
            jvmToolchain(21)
        }
    }
}

kover {
    reports {
        total {
            html {
                onCheck = true
            }
            xml {
                onCheck = true
            }
        }
    }
}