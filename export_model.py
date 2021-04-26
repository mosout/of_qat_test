from model import Lenet
import oneflow as flow
import oneflow.typing as tp
import numpy as np

func_config = flow.FunctionConfig()
func_config.enable_qat(True)
func_config.qat.symmetric(True)
func_config.qat.per_channel_weight_quantization(False)
func_config.qat.moving_min_max_stop_update_after_iters(1000)


def make_repvgg_infer_func():
    input_lbns = {}
    output_lbns = {}

    @flow.global_function("predict", function_config=func_config)
    def repvgg_inference(image: tp.Numpy.Placeholder(shape=(1, 1, 28, 28))) -> tp.Numpy:
        input_lbns["image"] = image.logical_blob_name
        output = Lenet(image)
        output_lbns["output"] = output.logical_blob_name
        return output

    return repvgg_inference, input_lbns, output_lbns


if __name__ == "__main__":
    repvgg_infer, input_lbns, output_lbns = make_repvgg_infer_func()
    flow.load_variables(flow.checkpoint.get("./lenet_models"))

    x = np.ones(shape=(1, 1, 28, 28))
    original_out = repvgg_infer(x)
    model_builder = flow.saved_model.ModelBuilder("./output")
    signature_builder = (
        model_builder.ModelName("Lenet")
        .Version(1)
        .AddFunction(repvgg_infer)
        .AddSignature("regress")
    )
    for input_name, lbn in input_lbns.items():
        signature_builder.Input(input_name, lbn)
    for output_name, lbn in output_lbns.items():
        signature_builder.Output(output_name, lbn)
    model_builder.Save(False)
