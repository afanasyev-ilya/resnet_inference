**1. input & install**

use ResNet50.onnx file from samples/data/resnet50

```
sudo apt-get install libopencv-dev
python3 -m pip install onnx
python3 -m pip install onnxruntime
```

**2. We verify model**

```
python3 ./verify_model.py
```

**3. Create TensorRT engine**

```
cd resnet
../bin/trtexec --onnx=data/ResNet50.onnx --saveEngine=data/ResNet50_fp32.engine
../bin/trtexec --onnx=data/ResNet50.onnx --saveEngine=ResNet50_fp16.engine --fp16
```

**4. compile**

Compile using makefile:

```
make
```

or manually:

```
g++ -o resnet50_inference resnet50_inference.cpp -I/usr/local/cuda/include -I/usr/local/TensorRT-10.2.0/include -I/usr/include/opencv4 -L/usr/local/cuda/lib64 -L/usr/local/TensorRT-10.2.0/lib -L/usr/lib/x86_64-linux-gnu -lnvinfer -lcudart -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -std=c++17
```
