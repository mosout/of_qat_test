import argparse
import oneflow as flow
from model import get_job_function

flow.config.enable_legacy_model_io(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--save-name", default="lenet", type=str)
    parser.add_argument("--disable-qat", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        args.batch_size, args.batch_size
    )
    # train
    train_job = get_job_function("train", not args.disable_qat, args.batch_size)
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())
    flow.checkpoint.save(args.save_name + "_models")
    print("model saved.")
