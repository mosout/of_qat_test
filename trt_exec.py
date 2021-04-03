import common
import argparse
import numpy as np
import tensorrt as trt


def build_engine_onnx(model_file, verbose=False):
    if verbose:
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    else:
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network_flags = network_flags | (
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    )

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        flags=network_flags
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.flags = config.flags | 1 << int(trt.BuilderFlag.INT8)
        return builder.build_engine(network, config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="lenet.onnx", type=str)
    parser.add_argument("--batch-size", default=100, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with build_engine_onnx(args.onnx) as engine:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            test_case = np.ones((args.batch_size, 1, 28, 28)).reshape(-1)
            np.copyto(inputs[0].host, test_case)
            trt_outputs = common.do_inference_v2(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
            data = trt_outputs[0]
            print(data.reshape(args.batch_size, -1))
