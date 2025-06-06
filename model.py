"""
Contains the class Model for the cross-view user identification
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from typing import Tuple

class CVmodel(nn.Module):
    """
    Cross-view matching model
    
    This model processes both vehicle-mounted and UAV-mounted camera images to enable
    cross-view matching between ground and aerial perspectives. It uses EfficientNet-based
    feature extractors combined with attention mechanisms for robust matching.
    
    Args:
        device: Device to run the model on (GPU/CPU)
        Height: Vehicle camera image height (default: 256)
        Width: Vehicle camera image width (default: 1280)
        Length: UAV camera image dimensions (default: 512)
        veh_hiddenUnits: Number of hidden units in vehicle attention module (default: 256)
        uav_hiddenUnits: Number of hidden units in UAV attention module (default: 256)
    """
    def __init__(self, device, Height=256, Width=1280, Length=512, veh_hiddenUnits=256, uav_hiddenUnits=256):
        super().__init__()
        self.device = device
        
        # Image dimensions
        self.Height = Height          # Vehicle camera height
        self.Width = Width           # Vehicle camera width
        self.Length = Length         # UAV camera dimensions (square)

        # Feature extractors
        self.veh_efficientnet = EfficientNet.from_pretrained('efficientnet-b0')  # Vehicle view feature extractor
        self.uav_efficientnet = EfficientNet.from_pretrained('efficientnet-b0')  # UAV view feature extractor

        # Attention modules
        self.veh_attentionModule = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=veh_hiddenUnits, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=veh_hiddenUnits, out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )  # Vehicle self-attention module
        
        self.uav_attentionModule = nn.Sequential(
            nn.Conv2d(in_channels=1280+1, out_channels=uav_hiddenUnits, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=uav_hiddenUnits, out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )  # UAV cross-view attention module with similarity channel

        # Vectorization layers
        self.veh_vectorizationLayer = nn.AvgPool2d(
            kernel_size=(round(Height/32), round(Width/(4*32))),
            stride=(1, round(Width/(4*32)))
        )  # Convert vehicle feature maps to descriptors
        
        self.uav_vectorizationLayer = nn.AvgPool2d(kernel_size=round(Length/32))  # Convert UAV feature maps to descriptors
        
        def maskGenerator(compassTheta: torch.tensor,
                          newImageCoordinate: torch.tensor,
                          baseMask: torch.tensor) -> torch.tensor:
            """
            Generates masks based on orientation and location
            
            Args:
                compassTheta: Vehicle compass angle relative to North
                newImageCoordinate: Target coordinates for mask placement
                baseMask: Base mask template
                
            Returns:
                a tensor representing all needed masks
            """

            batchSize = baseMask.shape[0]
            device = baseMask.device
            numVeh = newImageCoordinate.shape[1]
            
            # Calculate feature map center
            numRows = baseMask.shape[2]
            numCols = baseMask.shape[3]
            center = torch.tensor([numCols/2-1, numRows/2-1],device=device)
            featureMapSize = torch.tensor([numRows/3, numCols/3],device=device).unsqueeze(0) # [1, 1, 2]
                
            # Compute rotation angles for 4 views
            angles = (-compassTheta).unsqueeze(1).unsqueeze(1).unsqueeze(0) - (torch.arange(4, device=device)*np.pi/2).view(4, 1, 1, 1, 1)# [4, B, 1, 1, 1]
            cos = torch.cos(angles)
            sin = torch.sin(angles)
        
            # Create rotation matrices
            zeros = torch.zeros_like(cos)
            row0 = torch.cat([cos, -sin, zeros], dim=-1)  # [4, B, 1, 1, 3]
            row1 = torch.cat([sin, cos, zeros], dim=-1)  # [4, B, 1, 1, 3]
            rot_mats = torch.cat([row0, row1], dim=3).repeat(1, 1, numVeh, 1, 1)  # [4, B, V, 2, 3]
        
            # Apply rotation
            rot_mats = rot_mats.view(-1, 2, 3)  # [4*B*V, 2, 3]
            masks = baseMask.unsqueeze(1).unsqueeze(0).repeat(4, 1, numVeh, 1, 1, 1).view(-1,1,numRows,numCols)  # [4*B*V, 1, 3L', 3L']
            rotated = F.grid_sample(masks,
                                    F.affine_grid(rot_mats, masks.size(), align_corners=True),
                                    align_corners=True)
        
            # Compute translation parameters
            scale = torch.tensor([numRows/3/1024, numCols/3/1024], device=device)
            newCoord = newImageCoordinate * scale
            offset = (center - (newCoord + featureMapSize - 1)) / (numRows/2) # [B, V, 2]
            offset = offset.unsqueeze(-1).unsqueeze(0) # [1 B, V, 2, 1]
            
            # Create translation matrices
            eye = torch.eye(2, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2, 2]
            eye = eye.repeat(4, batchSize, numVeh, 1, 1)  # [4, B, V, 2, 2]
            offset = offset.repeat(4, 1, 1, 1, 1)  # [4, B, V, 2, 1]
            trans_mats = torch.cat([eye, offset], dim=4)  # [4, B, V, 2, 3]
        
            # Apply translation
            trans_mats = trans_mats.view(-1, 2, 3)  # [4*B*V , 2, 3]
            translated = F.grid_sample(rotated,
                                       F.affine_grid(trans_mats, rotated.size(), align_corners=True),
                                       align_corners=True)
        
            # Crop and organize results
            crop_h = slice(int(numRows/3)-1, 2*int(numRows/3)-1)
            crop_w = slice(int(numCols/3)-1, 2*int(numCols/3)-1)
            translated = translated.view(4, batchSize, numVeh, 1, numRows, numCols) # [4, B, V, 1, 3L', 3L']
            return translated[:, :, :, :, crop_h, crop_w] # [4, B, V, 1, L', L']
        self.maskGenerator = maskGenerator
        
    def forward(self, vehImage, uavImage, compassTheta, BBoxCenters, baseMask):
        """
        Forward pass through the cross-view matching network
        
        Args:
            vehImage: Vehicle camera images (concatenated)
            uavImage: UAV camera image
            compassTheta: Vehicle orientation angle
            BBoxCenters: Bounding box centers for detected vehicles by YOLO
            baseMask: Base mask template
            
        Returns:
            Tuple containing:
                uavView_descriptors: Normalized UAV view descriptors
                vehView_descriptor: Normalized vehicle view descriptor
        """
        numVeh = BBoxCenters.shape[1]
        # Extract vehicle view features
        vehView_featureMap = self.veh_efficientnet.extract_features(vehImage)
        
        # Self-attention module
        vehView_weights = self.veh_attentionModule(vehView_featureMap)
        
        # Vectorization module (compute vehicle view descriptor)
        vehView_descriptor = nn.functional.normalize(
            self.veh_vectorizationLayer(vehView_featureMap * vehView_weights),
            p=2,
            dim=1) # [B, C, 1, 4]
        vehView_descriptor_new = vehView_descriptor.permute(3, 0, 1, 2).unsqueeze(-1) # [4, B, C, 1, 1]
    
        # Extract UAV view features
        uavView_featureMap = self.uav_efficientnet.extract_features(uavImage) # [B, C, L', L']
        featureMap_shape = uavView_featureMap.size()
        uavView_featureMap = uavView_featureMap.unsqueeze(0).repeat(4, 1, 1, 1, 1) # [4, B, C, L', L']
        norm_uavView_featureMap = nn.functional.normalize(uavView_featureMap, p=2, dim=2)

        # Compute similarity score maps
        similarities  = torch.sum(norm_uavView_featureMap*vehView_descriptor_new,
                  keepdim=True, dim=2) # [4, B, 1, L', L']
    
        # Cross-view attention module
        uavViewFeatureMap = self.uav_attentionModule(
            torch.cat((uavView_featureMap, similarities), dim=2).view(-1, featureMap_shape[1]+1, featureMap_shape[2], featureMap_shape[3])
        ).view(4, featureMap_shape[0], 1, featureMap_shape[2], featureMap_shape[3]) * uavView_featureMap
    
        # Compute UAV-view descriptors for detected vehicles
        # Orientation disambiguation
        masks = self.maskGenerator(
            compassTheta,
            BBoxCenters,
            baseMask
        ) # [4, B, V, 1, L', L']
        numerator = torch.pow(torch.tensor(round(self.Length/32)), 2)
        denominator = torch.sum(torch.sum(masks, keepdim=True, dim=5), keepdim=True, dim=4) + 1e-8
        maskFactor = numerator / denominator # [4, B, V, 1, 1, 1]

        # Vectorization
        masks = masks.view(-1, 1, featureMap_shape[2], featureMap_shape[3])# [4*B*V, 1, L', L']
        uavViewFeatureMap = uavViewFeatureMap.unsqueeze(2).repeat(1, 1, numVeh, 1, 1, 1).view(-1, featureMap_shape[1], featureMap_shape[2], featureMap_shape[3]) # [4*B*V, C, 1, 1]
        uavView_descriptors = self.uav_vectorizationLayer(uavViewFeatureMap * masks)
        uavView_descriptors = uavView_descriptors.view(4, featureMap_shape[0], numVeh, featureMap_shape[1], 1, 1) # [4, B, V, C, 1, 1]
        uavView_descriptors = maskFactor * uavView_descriptors
        uavView_descriptors = nn.functional.normalize(uavView_descriptors, p=2, dim=3).squeeze(-1).squeeze(-1).permute(1, 3, 2, 0) # [B, C, V, 4]
        return uavView_descriptors, vehView_descriptor
