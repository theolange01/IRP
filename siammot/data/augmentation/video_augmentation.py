# IRP SiamMOT Tracker

import random
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter as ImageColorJitter, GaussianBlur


class SiamVideoBlur(GaussianBlur):
    """Apply Gaussian Blur to a video clip."""
    def __init__(self,
                 kernel_size=3,
                 sigma=(0.1, 2.0)):
        super(SiamVideoBlur, self).__init__(kernel_size, sigma=sigma)
        
    def __call__(self, video, target):
        if not isinstance(video, (list, tuple)):
            return video, target
        
        sigma = self.get_params(self.sigma[0], self.sigma[1])
    
        
        new_video = []
        new_target = []
        for (image, image_target) in zip(video, target):
            new_video.append(F.gaussian_blur(image, self.kernel_size, [sigma, sigma])) # Apply the gaussian Blur to each frame of the clip
            new_target.append(image_target)
        
        return new_video, new_target
          

class SiamVideoRandomHorizontalFlip(object):
    """Apply a Random Horozontal Flip to a video clip."""
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
                _, height, width = image.shape
                new_video.append(F.hflip(image))
                box = image_target["boxes"]
                box[:, 0], box[:, 2] = width - box[:, 2], width - box[:, 0]
                new_target.append({"boxes": box, "labels": image_target["labels"]})
        else:
            new_video = video
            new_target = target
        
        return new_video, new_target


class SiamVideoColorJitter(ImageColorJitter):
    """Apply a ColorJitter to a video clip."""
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
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        new_video = []
        new_target = []
        # Apply the same transformation to each frame of the video clip
        for i, (image, image_target) in enumerate(zip(video, target)):
            if i == idx:
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        image = F.adjust_brightness(image, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        image = F.adjust_contrast(image, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        image = F.adjust_saturation(image, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        image = F.adjust_hue(image, hue_factor)
                
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target