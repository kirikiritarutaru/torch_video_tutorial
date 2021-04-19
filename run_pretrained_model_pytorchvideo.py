import json
from pathlib import Path

import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (ApplyTransformToKey, ShortSideScale,
                                     UniformTemporalSubsample)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (CenterCropVideo,
                                                      NormalizeVideo)


# 事前準備
# Pytorchvideoのリポジトリからhubconf.pyをダウンロード

# コードを動かす前にデータ・セットのラベルをダウンロード
# wget https://dl.fbaipublicfiles.com/pyslowfast/
# dataset/class_names/kinetics_classnames.json

# コードを動かす前にサンプルの動画をダウンロード
# wget https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4

if __name__ == '__main__':
    # Load Model
    device = 'cuda:0'
    model_name = 'slow_r50'
    path = Path.cwd()

    model = torch.hub.load(
        path,
        source='local',
        model=model_name,
        pretrained=True
    )

    model = model.eval()
    model = model.to(device)

    # Setup Labels
    with open('kinetics_classnames.json', 'r')as f:
        kinetics_classnames = json.load(f)

    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")

    # Input Transform
    # slow_r50モデルに固有なパラメータであることに注意！！
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8
    sampling_rate = 8
    frames_per_second = 30

    transform = ApplyTransformToKey(
        key='video',
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x:x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ]
        )
    )

    clip_duration = (num_frames*sampling_rate)/frames_per_second

    # Load Video
    video_path = 'archery.mp4'
    start_sec = 0
    end_sec = start_sec+clip_duration

    video = EncodedVideo.from_path(video_path)
    video_data = video.get_clip(
        start_sec=start_sec,
        end_sec=end_sec
    )

    video_data = transform(video_data)

    inputs = video_data['video']
    inputs = inputs.to(device)

    # Predict
    preds = model(inputs[None, ...])
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices

    pred_class_names = [
        kinetics_id_to_classname[int(i)] for i in pred_classes[0]
    ]
    print(f'Predicted labels: {pred_class_names}')
