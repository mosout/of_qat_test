import oneflow as flow
from oneflow.python.test.onnx.util import convert_to_onnx_and_check
from model import Lenet

@flow.global_function()
def lenet():
    with flow.scope.placement("cpu", "0:0"):
        x = flow.get_variable(
                name="x1",
                shape=(100,1, 28, 28),
                dtype=flow.float,
                initializer=flow.constant_initializer(1),
            )
        return Lenet(x)

if __name__ == "__main__":
    convert_to_onnx_and_check(lenet)
