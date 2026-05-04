import onnx
from onnx import TensorProto
import numpy as np
import os, sys

def embed_weights(src_onnx, src_data, dst_onnx):
    model = onnx.load(src_onnx, load_external_data=False)
    
    with open(src_data, 'rb') as f:
        ext_data = f.read()
    
    SIZE_MAP = {1: 4, 7: 4, 8: 8, 11: 8}  # FLOAT, INT32, INT64, DOUBLE
    
    offset = 0
    for init in model.graph.initializer:
        num_elems = 1
        for d in init.dims:
            num_elems *= d
        elem_size = SIZE_MAP.get(init.data_type, 4)
        expected_bytes = num_elems * elem_size
        
        init.raw_data = ext_data[offset:offset + expected_bytes]
        init.data_location = TensorProto.DEFAULT
        offset += expected_bytes
    
    onnx.save(model, dst_onnx)

embed_weights('models/temp_real.onnx', 'models/temp_real.onnx.data', 'demo/retail_classifier.onnx')

size = os.path.getsize('demo/retail_classifier.onnx')
print(f"Embedded ONNX: {size / 1024:.1f} KB ({size / 1024 / 1024:.2f} MB)")

if os.path.exists('demo/retail_classifier.onnx.data'):
    os.remove('demo/retail_classifier.onnx.data')

import onnxruntime as ort
sess = ort.InferenceSession('demo/retail_classifier.onnx')
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = sess.run(None, {'input': x})
print(f"Test OK! Output: shape={result[0].shape}, sum={result[0].sum():.4f}")