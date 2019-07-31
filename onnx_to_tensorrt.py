import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


onnx_model_dir = os.path.abspath(os.path.join(__file__, "..", "pytorchnet", "torch_onnx_model"))
onnx_file_path = os.path.join(onnx_model_dir, "torch_onnx_model.onnx")
onnx_trt_model_dir = os.path.abspath(os.path.join(__file__, "..", "onnx_trt_model", "pytorchnet", "test_model"))
if not os.path.exists(onnx_trt_model_dir):
    os.makedirs(onnx_trt_model_dir)
engine_file_path = os.path.join(onnx_trt_model_dir, "test_model.trt")

if os.path.exists(engine_file_path):
    os.system("rm %s" % engine_file_path)

TRT_LOGGER = trt.Logger()
max_batch_size = 10


def printable_graph(Network):
    """
    Print the architecture of model in onnx
    :param Network: input builder.create_network() object
    :return: None
    """
    print("=======================================")
    for layer_idx in range(Network.num_layers):
        print(Network.get_layer(layer_idx).name)
    print("=======================================")


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30  # 1GB
            builder.max_batch_size = max_batch_size
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print("==========Parsing Error==========")
                    print(parser.get_error(0))
                    print("=================================")
            # network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
            print('Completed parsing of ONNX file')
            printable_graph(network)
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            try:
                print("Model saving .. ", end='')
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
                print("ok")
            except Exception as e:
                print("error")
                if os.path.exists(engine_file_path):
                    os.system("rm %s" % engine_file_path)
                print(e)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


# model parameter
input_shape = (3, 100, 100)
num_of_data = 10
batch_size = 2
# num_class = 30000
num_class = 5

# preparing data
seed = 19
np.random.seed(seed)
x = np.random.randn(num_of_data, *input_shape)
# x = np.random.randint(1, 5, (num_of_data, input_shape)) * 1.
x = np.asarray(x, dtype=np.float32)
# x = np.zeros((num_of_data, *input_shape))

# Create a Network Definition
with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0))*batch_size, dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1))*batch_size, dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    iter = int(num_of_data/batch_size) + (1 if (num_of_data % batch_size) != 0 else 0)
    for i in range(iter):
        cuda.memcpy_htod_async(d_input, x[i*batch_size:i*batch_size+batch_size], stream)
        # cuda.memcpy_htod(d_input, x[i*batch_size:i*batch_size+batch_size])

        context.execute_async(batch_size, [int(d_input), int(d_output)], stream.handle, None)
        # context.execute(batch_size, bindings=[int(d_input), int(d_output)])

        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # cuda.memcpy_dtoh(h_output, d_output)

        stream.synchronize()

        for j in range(batch_size):
            print("x_%d: " % j, x[i*batch_size + j, 0, :2, :2])
            print("output_%d: " % j, h_output[j])
            print("==========================================================")
        # print("pred: ", np.max(output, axis=0))
