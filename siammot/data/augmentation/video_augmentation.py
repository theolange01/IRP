from typing import Optional, Tuple, Dict

from torch import Tensor
import random
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter as ImageColorJitter
from torchvision.models.detection.transform import resize_boxes

from .image_augmentation import ImageResize


class VideoTransformer(object):
    def __init__(self, transform_fn=None):
        if transform_fn is None:
            raise KeyError('Transform function should not be None.')
        self.transform_fn = transform_fn

    def __call__(self, video, target=None):
        """
        A data transformation wrapper for video
        :param video: a list of images
        :param target: a list of BoxList (per image)
        """
        if not isinstance(video, (list, tuple)):
            return self.transform_fn(video, target)

        new_video = []
        new_target = []
        for (image, image_target) in zip(video, target):
            (image, image_target) = self.transform_fn(image, image_target)
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target




class SiamVideoResize(ImageResize):
    def __init__(self, min_size, max_size, size_divisibility):
        super(SiamVideoResize, self).__init__(min_size, max_size, size_divisibility)

    def __call__(self, video, target=None):

        if not isinstance(video, (list, tuple)):
            return super(SiamVideoResize, self).__call__(video, target)

        assert len(video) >= 1
        new_size = self.get_size(video[0].size)

        new_video = []
        new_target = []
        for (image, image_target) in zip(video, target):
            (image, image_target) = self._resize(image, new_size, image_target)
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target

    def resize(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        if self.training:
            if self._skip_resize:
                return image, target
            size = self.torch_choice(self.min_size)
        else:
            size = self.min_size[-1]
        image, target = _resize_image_and_masks(image, size, self.max_size, target, self.fixed_size)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image, target
    


class SiamVideoRandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, video, target=None):

        if not isinstance(video, (list, tuple)):
            return video, target

        new_video = []
        new_target = []
        # All frames should have the same flipping operation
        if random.random() < self.prob:
            for (image, image_target) in zip(video, target):
                new_video.append(F.hflip(image))
                new_target.append(image_target[[1,0,3,2]])
        else:
            new_video = video
            new_target = target
        return new_video, new_target


class SiamVideoColorJitter(ImageColorJitter):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None):
        super(SiamVideoColorJitter, self).__init__(brightness, contrast, saturation, hue)

    def __call__(self, video, target=None):
        # Color jitter only applies for Siamese Training
        if not isinstance(video, (list, tuple)):
            return video, target

        idx = random.choice((0, 1))
        # all frames in the video should go through the same transformation
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        new_video = []
        new_target = []
        for i, (image, image_target) in enumerate(zip(video, target)):
            if i == idx:
                image = transform(image)
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target