plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.minimap"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.example.minimap"
        minSdk = 26
        targetSdk = 35
        versionCode = 2
        versionName = "0.2.0-research"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { buildConfig = true }
}

dependencies {
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation(project(":vision-core"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
}
