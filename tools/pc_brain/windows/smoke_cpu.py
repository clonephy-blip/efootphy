"""Check native Windows dependencies without capture or private model inference."""
import ctypes
import cv2
import mss
import numpy
import PIL
import onnxruntime as ort

ort.disable_telemetry_events()
assert 'CPUExecutionProvider' in ort.get_available_providers()
assert hasattr(ctypes, 'windll'), 'This check is for Windows'
print('Windows CPU imports OK:', cv2.__version__, numpy.__version__, PIL.__version__, ort.__version__)
print('Providers:', ort.get_available_providers())
