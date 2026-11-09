# Archived local Android inference path

These sources are outside `app/src` and are not compiled into the HUD APK. They
preserve the previous on-device MediaProjection/minimap implementation for
reference. The app has no ONNX Runtime dependency. Do not re-enable this path as
the target architecture: PC performs all analysis, Android only renders HUD.

The `vision-core` JVM module is retained separately for existing geometry tests;
it is no longer an APK dependency.
