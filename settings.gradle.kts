pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}




rootProject.name = "SKaiNET-FED"

includeBuild("build-logic")

include("skainet-fed:skainet-fed-core")
