# lenet_eval.py
import numpy as np
import oneflow as flow
from typing import Tuple
import oneflow.typing as tp

BATCH_SIZE = 100
flow.config.enable_legacy_model_io(False)

qat_func_config = flow.FunctionConfig()
qat_func_config.enable_qat(True)
qat_func_config.qat.symmetric(True)
qat_func_config.qat.per_channel_weight_quantization(False)
qat_func_config.qat.moving_min_max_stop_update_after_iters(1000)


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(
        data,
        32,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv1",
        kernel_initializer=initializer,
        use_bias=False,
    )
    pool1 = flow.nn.max_pool2d(
        conv1, ksize=2, strides=2, padding="SAME", name="pool1", data_format="NCHW"
    )
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv2",
        kernel_initializer=initializer,
        use_bias=False,
    )
    pool2 = flow.nn.max_pool2d(
        conv2, ksize=2, strides=2, padding="SAME", name="pool2", data_format="NCHW"
    )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="dense1",
        use_bias=False
    )
    if train:
        hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="dense2",use_bias=False)

@flow.global_function(type="predict", function_config=qat_func_config)
def eval_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)

    return logits


if __name__ == "__main__":
    flow.load_variables(flow.checkpoint.get("./lenet_models_1"))
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )

    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            # labels, logits = eval_job(images, labels)
            logits = eval_job(images)
            break
            # acc(labels, logits)
        break
    flow.onnx.export(
        eval_job, "./lenet_models_1", "lenet.onnx", opset=10, external_data=False,
    )
