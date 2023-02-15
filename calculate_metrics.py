from typing import Tuple

import chainer
import chainer.cuda
import numpy
import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
import yaml

import tgan2
from tgan2 import UCF101Dataset
from tgan2.evaluations import inception_score
from tgan2.utils import make_instance


def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(
        np.dot(sigma_gen, sigma_real), disp=False
    )  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]

    return mu, sigma


@torch.no_grad()
def FVD(videos_fake: np.ndarray, videos_real: np.ndarray, device: str = "cuda"):
    detector_kwargs = dict(
        rescale=False, resize=False, return_features=True
    )  # Return raw features before the softmax layer.

    detector = torch.jit.load("i3d_torchscript.pt").eval().to(device)

    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)

    feats_fake = detector(videos_fake, **detector_kwargs).cpu().numpy()
    feats_real = detector(videos_real, **detector_kwargs).cpu().numpy()

    return compute_fvd(feats_fake, feats_real)


# len(dset) * n_loops == 9537 * 10 == 5610 * 17
def get_dataset_samples(dataset, batchsize, n_iterations):
    ys = []
    it = chainer.iterators.MultiprocessIterator(
        dataset, batch_size=batchsize, shuffle=False, repeat=True, n_processes=8
    )
    for i, batch in enumerate(it):
        if i == n_iterations:
            break
        print("Compute {} / {}".format(i + 1, n_iterations))

        batch = chainer.dataset.concat_examples(batch)
        ys.append(batch)
    return numpy.concatenate(ys)


def interpolate(videos):

    videos = torch.from_numpy(videos).permute(0, 2, 3, 4, 1)
    target_shape = (224, 224, 3)
    videos = F.interpolate(videos, target_shape).numpy()

    return videos


def main():
    num_videos = 2  # 2048
    video_len = 16

    batchsize = 2
    device = "cpu"

    ucf101_h5path_train = "datasets/ucf101_192x256/train.h5"
    ucf101_config_train = "datasets/ucf101_192x256/train.json"
    snapshot_path = "datasets/models/balance_trained.npz"
    config_path = "results/full-bs-128/config.yml"

    config = yaml.load(open(config_path))
    print(yaml.dump(config, default_flow_style=False))

    gen = make_instance(tgan2, config["gen"], args={"out_channels": 3})
    chainer.serializers.load_npz(snapshot_path, gen)

    dataset = UCF101Dataset(
        n_frames=video_len,
        h5path=ucf101_h5path_train,
        config_path=ucf101_config_train,
        img_size=192,
        stride=1,
    )

    videos_real = get_dataset_samples(dataset, batchsize, num_videos // batchsize)
    videos_real = interpolate(videos_real)


    videos_fake = inception_score.make_samples(
        gen, batchsize=batchsize, n_samples=num_videos, n_frames=video_len
    )
    videos_fake = interpolate(videos_fake)

    print("Computing our FVD...")
    our_fvd_result = FVD(videos_fake, videos_real, device)

    print(f"[FVD scores]: {our_fvd_result}")


if __name__ == "__main__":
    main()
