import io
import copy
import torch
from PIL import Image
from videodataset import VideoDataset

class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        
        self.image_name_formatter = image_name_formatter
        
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader

    def __call__(self, video_path, frame_indices):
        
        video = []
        
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            if image_path.exists():
                video.append(self.image_loader(image_path))

        return video

class VideoDatasetMultiClips(VideoDataset):

    def __loading(self, path, video_frame_indices):
        clips = []
        segments = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clips.append(torch.stack(clip, 0).permute(1, 0, 2, 3))
            segments.append(
                [min(clip_frame_indices),
                 max(clip_frame_indices) + 1])

        return clips, segments

    def __getitem__(self, index):
        path = self.data[index]['video']

        video_frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            video_frame_indices = self.temporal_transform(video_frame_indices)

        clips, segments = self.__loading(path, video_frame_indices)

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        if 'segment' in self.target_type:
            if isinstance(self.target_type, list):
                segment_index = self.target_type.index('segment')
                targets = []
                for s in segments:
                    targets.append(copy.deepcopy(target))
                    targets[-1][segment_index] = s
            else:
                targets = segments
        else:
            targets = [target for _ in range(len(segments))]

        return clips, targets