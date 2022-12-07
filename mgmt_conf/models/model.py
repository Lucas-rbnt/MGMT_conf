# Standard libraries
from typing import Tuple

# Third-party libraries
import monai
import torch
import torch.nn as nn

# Local dependencies
from .fusion import EmbraceWithAttentionNet

__all__ = ["ResNet10Wrapper", "MultimodalModel"]


class ResNet10Wrapper(nn.Module):
    """
    The nn.Module class, it is just a wrapper around the original ResNet10-3D from MONAI https://github.com/Project-MONAI/MONAI.

    Attributes:
        n_input_channels (int) : the number of channels, in our case the number of modality to be used.
        n_classes (int) : the class number of the problem. Default to `2`.
        confidence_branch (bool) : whether or not to use a derived confidence branch.
        embracenet (bool) : whether or not to declare a model that will be plugged into embracenet framework.
    """

    def __init__(
        self,
        n_input_channels: int,
        n_classes: int = 2,
        confidence_branch: bool = False,
        embracenet: bool = False,
    ) -> None:
        """
        Class constructor.

        Args:
            n_input_channels (int) : the number of channels, in our case the number of modality to be used.
            n_classes (int) : the class number of the problem. Default to `2`.
            confidence_branch (bool) : whether or not to use a derived confidence branch. Default to `False`.
            embracenet (bool) : whether or not to declare a model that will be plugged into embracenet framework. Default to `False`.
        """
        super(ResNet10Wrapper, self).__init__()
        self.model = monai.networks.nets.resnet10(
            spatial_dims=3, n_input_channels=n_input_channels, n_classes=1
        )
        self.model.fc = nn.Identity()
        self.confidence_branch = confidence_branch
        self.embracenet = embracenet
        if self.confidence_branch:
            self.confidence = nn.Linear(512, 1)
        if not self.embracenet:
            self.classifier = nn.Linear(512, n_classes)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Usual forward class for a nn.Module.
        """
        x = self.model(x)
        if not self.embracenet:
            pred = self.classifier(x)
            if self.confidence_branch:
                confidence = self.confidence(x)
                return pred, confidence
            else:
                return pred
        else:
            return x


class MultimodalModel(nn.Module):
    """
    Class for using different modalities and the EmbraceNet architecture. https://github.com/idearibosome/embracenet

    Attributes:
        base_model (nn.Module) : the base model used as backbone.
        modalities (Tuple[str, ...]) : MRIs modalities used.
        device (torch.device) : device on which operations and weights will be computed and stored.
        n_classes (int) : the number of class we want to predict. Default to `2`.
        confidence_branch (bool) : whether or not to use a derived confidence branch.
        embracement_size (int) : the size of the embracement layer conformly to  https://github.com/idearibosome/embracenet. Default to `256`.
        use_attention (bool) : whether or not to use an attention layer between embeddings from different modalities. Default to `True`.
        pre_output_size (int) : initial dimension of the outputted embedding. Here we are using a ResNet10. Default to `512`.
        dropout (bool) : if set to `True`, applies dropout to modality during training to enhance regularization and model's performances. Default to `True`.
    """

    def __init__(
        self,
        base_model: nn.Module,
        modalities: Tuple[str, ...],
        device: torch.device,
        n_classes: int = 2,
        confidence_branch: bool = False,
        embracement_size: int = 256,
        use_attention: bool = True,
        pre_output_size: int = 512,
        dropout: bool = True,
    ) -> None:
        """
        Class constructor.

        Args:
            base_model (nn.Module) : the base model used as backbone.
            modalities (Tuple[str, ...]) : MRIs modalities used.
            device (torch.device) : device on which operations and weights will be computed and stored.
            n_classes (int) : the number of class we want to predict. Default to `2`.
            confidence_branch (bool) : whether or not to use a derived confidence branch.
            embracement_size (int) : the size of the embracement layer conformly to  https://github.com/idearibosome/embracenet. Default to `256`.
            use_attention (bool) : whether or not to use an attention layer between embeddings from different modalities. Default to `True`.
            pre_output_size (int) : initial dimension of the outputted embedding. Here we are using a ResNet10. Default to `512`.
            dropout (bool) : if set to `True`, applies dropout to modality during training to enhance regularization and model's performances. Default to `True`.
        """
        super(MultimodalModel, self).__init__()
        self.modalities = modalities
        self._modalities_sanity_check()
        self.device = device
        self.dropout = dropout
        self.pre_output_size = pre_output_size
        self.embrace = EmbraceWithAttentionNet(
            device,
            modalities_size={
                modality: self.pre_output_size for modality in self.modalities
            },
            embracement_size=embracement_size,
            use_attention=use_attention,
        ).to(device)
        self.post = nn.Linear(embracement_size, n_classes).to(device)

        for modality in self.modalities:
            setattr(self, f"model_{modality}", base_model)

        self.confidence_branch = confidence_branch
        if self.confidence_branch:
            self.confidence = nn.Linear(embracement_size, 1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Regular nn.Module's forward method.
        """
        item = {}
        for key in sorted(x):
            item[key] = getattr(self, f"model_{key}")(x[key].to(self.device))

        availabilities = None
        if self.dropout:
            dropout_prob = torch.rand(1, device=self.device)[0]
            if dropout_prob >= 0.5:
                target_modalities = torch.randint(
                    0, len(x), (list(x.values())[0].size(0),), device=self.device
                ).to(torch.int64)
                availabilities = torch.logical_not(
                    nn.functional.one_hot(target_modalities, num_classes=len(x))
                ).float()
        x_embrace = self.embrace(item, availabilities=availabilities)
        pred = self.post(x_embrace)
        if self.confidence_branch:
            confidence = self.confidence(x_embrace)
            return pred, confidence
        else:
            return pred

    def _modalities_sanity_check(
        self, expected_modalities: Tuple[str, ...] = ("FLAIR", "T1wCE", "T1w", "T2w")
    ) -> None:
        """
        Checks if modalities are correctly employed here.
        """
        assert all(
            modality in expected_modalities for modality in self.modalities
        ), f"Accepted input data must be in: {expected_modalities}."

        assert len(self.modalities) > 0, "At least one input must be provided."
