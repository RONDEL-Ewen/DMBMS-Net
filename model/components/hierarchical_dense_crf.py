import torch
import torch.nn as nn
import torch.nn.functional as F



# ===============================[ HierarchicalDenseCRF ]===============================

class HierarchicalDenseCRF(nn.Module):

    def __init__(
        self, 
        num_classes:        int = 8, 
        num_iterations:     int = 2, 
        spatial_ksize:      int = 3, 
        bilateral_ksize:    int = 3, 
        num_encoder_blocks: int = 5, 
        base_dim:           int = 64, 
        skip_channels           = None
    ):
        """
        Hierarchical Dense CRF that incorporates skip features from the Frame Encoder.
        
        Args:
            num_classes (int): Number of segmentation classes.
            num_iterations (int): Number of mean-field iterations.
            spatial_ksize (int): Kernel size for spatial filtering.
            bilateral_ksize (int): Kernel size for bilateral filtering.
            num_encoder_blocks (int, optional): Number of encoder blocks used in the Frame Encoder.
            base_dim (int, optional): Base channel dimension used in the Frame Encoder.
            skip_channels (list, optional): List of channel numbers for each skip feature.
                If not provided, they will be dynamically computed as 
                [base_dim * (2**i) for i in range(num_encoder_blocks - 1)].
        """
        super(HierarchicalDenseCRF, self).__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations

        # Spatial filtering: depthwise convolution per class.
        self.spatial_conv = nn.Conv2d(
            num_classes, num_classes,
            kernel_size=spatial_ksize,
            padding=spatial_ksize // 2,
            groups=num_classes,
            bias=False
        )
        # Bilateral filtering: takes concatenated Q and guidance.
        # Guidance is the concatenation of the input image (3 channels) and the fused skip features (projected to 3 channels).
        self.bilateral_conv = nn.Conv2d(
            num_classes + 6, num_classes,
            kernel_size=bilateral_ksize,
            padding=bilateral_ksize // 2,
            bias=False
        )
        # Compatibility transformation: initialized to -I.
        self.compat_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False)
        nn.init.constant_(self.compat_conv.weight, 0)
        with torch.no_grad():
            self.compat_conv.weight += -torch.eye(num_classes).view(num_classes, num_classes, 1, 1)

        # Determine skip channels dynamically if not provided.
        if skip_channels is None:
            if num_encoder_blocks is None or base_dim is None:
                raise ValueError("Either provide skip_channels or both num_encoder_blocks and base_dim.")
            # For an Frame Encoder producing (num_encoder_blocks - 1) skip connections,
            # the channels are assumed to be [base_dim * (2**0), base_dim * (2**1), ..., base_dim * (2**(num_encoder_blocks-2))]
            skip_channels = [base_dim * (2 ** i) for i in range(num_encoder_blocks - 1)]
        self.skip_channels = skip_channels

        # Create a projection layer for each skip feature to reduce it to 3 channels.
        self.skip_proj = nn.ModuleList([nn.Conv2d(ch, 3, kernel_size=1) for ch in self.skip_channels])

    def fuse_skip_features(
        self,
        skip_features,
        target_size
    ):
        """
        Fuse skip features by projecting each to 3 channels, upsampling them to the target size,
        and averaging the results.
        
        Args:
            skip_features (list of Tensor): List of skip feature maps.
            target_size (tuple): (H, W) target spatial dimensions.
        
        Returns:
            Tensor: Fused skip feature map of shape (N, 3, H, W).
        """
        if len(skip_features) != len(self.skip_proj):
            raise ValueError(f"Expected {len(self.skip_proj)} skip features, but got {len(skip_features)}.")
        
        proj_feats = []
        for feat, proj in zip(skip_features, self.skip_proj):
            feat_proj = proj(feat)  # Project to 3 channels.
            feat_proj = F.interpolate(feat_proj, size=target_size, mode='bilinear', align_corners=True)
            proj_feats.append(feat_proj)
        # Average all projected features.
        fused = torch.stack(proj_feats, dim=0).mean(dim=0)
        return fused

    def mean_field_update(
        self,
        unary,
        guidance
    ):
        """
        Perform mean-field iterations combining spatial and bilateral messages.
        """
        Q = F.softmax(unary, dim=1)
        for _ in range(self.num_iterations):
            spatial_msg = self.spatial_conv(Q)
            bilateral_input = torch.cat([Q, guidance], dim=1)
            bilateral_msg = self.bilateral_conv(bilateral_input)
            message = spatial_msg + bilateral_msg
            message = self.compat_conv(message)
            Q = F.softmax(unary - message, dim=1)
        return Q

    def forward(
        self,
        logits,
        image,
        skip_features = None
    ):
        """
        Forward pass of the Hierarchical Dense CRF.
        
        Args:
            logits (Tensor): Unrefined segmentation logits from the Mask Decoder (N, num_classes, H, W).
            image (Tensor): Input image (N, 3, H, W).
            skip_features (list of Tensor, optional): List of skip feature maps from the ImageEncoder.
                These are expected to match the skip connections produced by the encoder.
        
        Returns:
            Tensor: Refined segmentation probabilities.
        """
        target_size = logits.shape[2:]  # (H, W)
        if skip_features is not None:
            fused_skip = self.fuse_skip_features(skip_features, target_size)
        else:
            # If no skip features are provided, use a zero tensor.
            fused_skip = torch.zeros((image.shape[0], 3, *target_size), device=image.device)
        
        # Concatenate the input image (3 channels) with the fused skip features (3 channels).
        guidance = torch.cat([image, fused_skip], dim=1)  # (N, 6, H, W)
        
        # Full-resolution mean-field update.
        Q_full = self.mean_field_update(logits, guidance)
        
        # Also compute a two-level (downsampled) version for improved global consistency.
        logits_down = F.interpolate(logits, scale_factor=0.5, mode='bilinear', align_corners=True)
        guidance_down = F.interpolate(guidance, scale_factor=0.5, mode='bilinear', align_corners=True)
        Q_down = self.mean_field_update(logits_down, guidance_down)
        Q_down_up = F.interpolate(Q_down, size=target_size, mode='bilinear', align_corners=True)
        
        # Combine both scales.
        Q_combined = (Q_full + Q_down_up) / 2
        return Q_combined