# IRP SiamMOT Tracker

from .video_augmentation import SiamVideoBlur, \
    SiamVideoColorJitter, SiamVideoRandomHorizontalFlip


class Compose(object):
    """Class to compose multiple transformations."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
        

def build_siam_augmentation(cfg, is_train=True):
    """Create the composed transformation to apply to the dataset. No transformations are applied during validating or testing."""
    
    if is_train:
        flip_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
        
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
        
        kernel = cfg.INPUT.BLUR_KERNEL
        sigma = cfg.INPUT.BLUR_SIGMA
  
        video_color_jitter = SiamVideoColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
    
        transform = Compose(
            [
                video_color_jitter,
                SiamVideoRandomHorizontalFlip(prob=flip_prob),
                SiamVideoBlur(kernel, sigma)
            ]
        )
      
    else:
        transform = None
        
    return transform