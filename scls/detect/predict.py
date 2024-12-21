import cv2
import torch
from PIL import Image

from scls.engine.predictor import BasePredictor
from scls.engine.results import Results
from scls.utils import DEFAULT_CFG, ops


class DetecitonPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a Deteciton model.

    Notes:
        - Torchvision Deteciton models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from scls.utils import ASSETS
        from scls.models.scl.classify import DetecitonPredictor

        args = dict(model="scl-cls.pt", source=ASSETS)
        predictor = DetecitonPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes DetecitonPredictor setting the task to 'classify'."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"
        self._legacy_transform_name = "scls.scl.data.augment.ToTensor"

    def preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )
            if is_legacy_transform:  # to handle legacy transforms
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                img = torch.stack(
                    [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
                )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]