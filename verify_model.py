import onnx
import onnxruntime as ort

model_name = "data/ResNet50.onnx"

# Load the ONNX model
onnx_model = onnx.load(model_name)

# Check the model
onnx.checker.check_model(onnx_model)

# Print a human-readable representation of the graph
print(onnx.helper.printable_graph(onnx_model.graph))

# Run a basic inference to check if everything is working
ort_session = ort.InferenceSession(model_name)
print("ONNX Model loaded and verified successfully.")
