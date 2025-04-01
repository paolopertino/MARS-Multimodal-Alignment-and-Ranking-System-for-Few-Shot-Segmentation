import torch
import PIL

from torchvision import transforms

class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)
    
class MaybeToPILImage(transforms.ToPILImage):
    """
    Convert a ``torch.Tensor`` or ``numpy.ndarray`` to PIL Image, or keep as is if already a PIL Image.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to PIL Image.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, PIL.Image.Image):
            return pic
        return super().__call__(pic)