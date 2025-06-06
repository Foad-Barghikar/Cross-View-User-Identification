
import torch
import torch.nn as nn

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=None, alpha=None):
        super(InfoNCELoss, self).__init__()
        # Set default temperature of 0.1 if none provided
        if (temperature==None):
            self.temperature = 0.1
        else:
            self.temperature = temperature
        # Store alpha parameter for later use
        self.alpha = alpha
        
        def infoNCELoss(uavView_descriptors, vehView_descriptor, temperature, alpha):
            """
            Contrastive loss function based on the InfoNCE loss formula from:
            https://arxiv.org/pdf/2004.11362.pdf Eq.2
            """
            # Get batch size from vehicle view descriptor
            BATCH_SIZE = vehView_descriptor.shape[0]
            # If alpha is not provided, use the number of UAV view descriptors
            if alpha==None:
                alpha = uavView_descriptors.shape[2]
            
            # Calculate exponential similarity between descriptors
            exp_similarity = torch.exp(torch.sum(uavView_descriptors*vehView_descriptor, keepdim=True, dim=1) / temperature)
            
            # Extract positive similarity (first element)
            exp_similarity_positive = exp_similarity[:,0,0,:]
            
            # Calculate the average of negative similarities (all except the first element)
            exp_similarity_negative = torch.mean(exp_similarity[:,0,1:,:], dim=1)
            
            # Compute final InfoNCE loss
            loss = torch.mean(torch.mean(-torch.log(exp_similarity_positive/((alpha*exp_similarity_negative) + exp_similarity_positive)), dim=1))
            
            return loss
        
        # Store the loss function as an instance method
        self.infoNCELoss = infoNCELoss

    def forward(self, uavView_descriptors, vehView_descriptor):
        # Call the stored loss function with current parameters
        return self.infoNCELoss(uavView_descriptors, vehView_descriptor, self.temperature, self.alpha)
