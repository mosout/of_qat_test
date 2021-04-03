import argparse
import numpy as np
import oneflow as flow
from model import get_job_function

flow.config.enable_legacy_model_io(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--save-name", default="lenet", type=str)
    parser.add_argument("--disable-qat", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # precit
    data = np.ones((args.batch_size, 1, 28, 28))
    predict_job = get_job_function("predict", not args.disable_qat, args.batch_size)
    flow.load_variables(flow.checkpoint.get("./" + args.save_name + "_models"))
    print(predict_job(data))
    # export
    flow.onnx.export(
        predict_job,
        args.save_name + "_models",
        args.save_name + ".onnx",
        opset=10,
        external_data=False,
    )
    print("onnx saved.")
