import json

import torch

from models.utils.utils import loadmodule, getLastCheckPoint, getVal, getNameAndPackage, parse_state_name
from models.frame_seed_generator import FrameSeedGenerator


if __name__ == "__main__":
    # 1. generate input seed
    fsg = FrameSeedGenerator()

    seed = torch.rand([2047])
    t = 5
    x = torch.hstack([torch.tensor([t]), seed])
    x = x.unsqueeze(0)   # add batch size = 1

    input = fsg(x)
    input = input.to('cuda')

    # 2. load freezed and pre-trained model from checkpoint
    name = 'jelito3d_batchsize8'
    checkPointDir = '/home/z1143165/pytorch_GAN_zoo_2/output_networks/jelito3d_batchsize8'
    checkpointData = getLastCheckPoint(checkPointDir, name, scale=None, iter=None)

    modelConfig, pathModel, _ = checkpointData
    _, scale, _ = parse_state_name(pathModel)

    module = 'PGAN'
    packageStr, modelTypeStr = getNameAndPackage(module)
    modelType = loadmodule(packageStr, modelTypeStr)

    # visualizer = GANVisualizer(pathModel, modelConfig, modelType, None)

    with open(modelConfig, 'rb') as file:
        config = json.load(file)

    model = modelType(useGPU=True, storeAVG=True, **config)
    model.load(pathModel)

    # get generator and forward input
    getAvG = True   # ???
    if getAvG:
        output = model.avgG(input)
    else:
        output = model.netG(input)

    # sanity check
    for param in model.avgG.parameters():
        assert not (param.requires_grad)

    print('STH')
