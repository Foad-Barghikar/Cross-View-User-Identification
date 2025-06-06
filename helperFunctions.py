
import torch
import numpy as np
from tqdm.auto import tqdm
import os
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               baseMask: torch.tensor,
               batchSize: int,
               Length: int,
               lambdaReg: float) -> Tuple[float, float]:
    """
    Performs a single training epoch on the cross-view matching model.

    This function implements the complete training loop for one epoch, including:
    - Model training mode setup
    - Forward pass through the network
    - Loss calculation with optional L2 regularization
    - Backward pass and optimization
    - Accuracy calculation using cosine similarity

    Args:
        model: PyTorch model to be trained
        dataloader: DataLoader containing training data
        loss_fn: Loss function to minimize
        optimizer: Optimizer for parameter updates
        device: Target device (GPU/CPU) for computations
        baseMask: Base mask tensor
        batchSize: Size of each training batch
        Length: Dimension size for mask generation
        lambdaReg: L2 regularization strength (0 to disable)

    Returns:
        Tuple of (train_loss, train_acc):
            - train_loss: Average loss across all batches
            - train_acc: Average accuracy across all batches
    """
    print("train step")
    # Set model to training mode
    model.train()
    
    # Initialize metrics accumulators
    train_loss, train_acc = 0, 0
    
    # Process each batch in the dataloader
    for batch, (vehImage, uavImage, compassTheta, BBoxCenters) in enumerate(dataloader):
        # Move all tensors to the target device
        vehImage, uavImage, baseMask = vehImage.to(device), uavImage.to(device), baseMask.to(device)
        compassTheta, BBoxCenters = compassTheta.to(device), BBoxCenters.to(device)

        # Modify baseMask shape based on batch size
        currentBatchSize = len(compassTheta)
        if (batchSize!=currentBatchSize):
            baseMask = baseMaskGenerator(Length, device, currentBatchSize).to(device)
            
        # Forward pass through the network
        uavView_descriptors, vehView_descriptor = model(vehImage, uavImage, compassTheta, BBoxCenters, baseMask)
        
        # Calculate base loss
        loss = loss_fn(uavView_descriptors, vehView_descriptor)
        
        # Add L2 regularization if enabled
        if lambdaReg:
            # Calculate L2 norm for both EfficientNet models
            l2_norm = sum(p.pow(2).sum() for p in model.veh_efficientnet.parameters())
            l2_norm = l2_norm + sum(p.pow(2).sum() for p in model.uav_efficientnet.parameters())
            loss += lambdaReg * l2_norm

        # Accumulate loss
        train_loss += loss.item()
        
        # Backpropagation and optimization
        optimizer.zero_grad()    # Clear gradients
        loss.backward()          # Compute gradients
        optimizer.step()         # Update parameters
        
        # Calculate accuracy
        # Filter out-of-bounds bounding boxes
        maskIdxFilter = torch.logical_or(
            torch.logical_or(BBoxCenters[:,:30,0]<4, BBoxCenters[:,:30,0]>1020),
            torch.logical_or(BBoxCenters[:,:30,1]<4, BBoxCenters[:,:30,1]>1020)
        )
        
        # Prepare descriptors for similarity calculation
        # Reshape and normalize the vehicle view descriptor
        vehView_descriptor = torch.nn.functional.normalize(
            vehView_descriptor.permute(0,3,2,1).reshape(currentBatchSize, 1, vehView_descriptor.shape[1]*4),
            p=2, dim=2
        )
        
        # Reshape and normalize the UAV view descriptors
        uavView_descriptors = torch.nn.functional.normalize(
            uavView_descriptors.permute(0,2,3,1).reshape(currentBatchSize,uavView_descriptors.shape[2],uavView_descriptors.shape[1]*4),
            p=2, dim=2
        )
        
        # Calculate cosine similarity between views
        similarity = torch.sum(uavView_descriptors[:,:30,:]*vehView_descriptor, dim=2)
        
        # Mark out-of-FOV vehicles
        similarity[maskIdxFilter] = -2
        
        # Calculate batch accuracy (target vehicle is at the first entry)
        train_acc += (torch.argmax(similarity, dim=1) == 0).sum().item()/len(similarity)
        
        # Print progress every 10 batches
        if batch % 10 == 0:
            print(f"batch:{batch}, acc:{train_acc/(batch+1): .4f}, loss:{train_loss/(batch+1): .4f}")
    
    # Calculate average metrics across all batches
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc

def save_checkpoint(saving_dir: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    device: torch.device) -> None:
    """
    Saves a training checkpoint containing model and optimizer state.

    This function creates a checkpoint file containing:
    - Current training epoch
    - Model parameters (moved to CPU before saving)
    - Optimizer state dictionary
    
    The checkpoint is saved in the specified directory with a filename
    indicating the current epoch number.

    Args:
        saving_dir: Directory path where the checkpoint will be saved
        model: PyTorch model to save
        optimizer: Optimizer instance to save
        epoch: Current training epoch number
        device: Target device (GPU/CPU) for the model after saving

    Returns:
        None
    """
    print("save model")
    
    # Create checkpoint state dictionary
    state = {
        'epoch': epoch,
        'model_state_dict': model.cpu().state_dict(),  # Move model to CPU before saving
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    # Ensure the saving directory exists
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    # Construct checkpoint filename with epoch number
    checkpoint_path = os.path.join(saving_dir, f'checkpoint_{epoch}.pth')
    
    # Save checkpoint
    torch.save(state, checkpoint_path)
    
    # Move model back to specified device for continued training
    model.to(device)
    
def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              baseMask: torch.tensor,
              batchSize: int,
              Length: int) -> Tuple[float, float]:
    """
    Evaluates a PyTorch model on a test dataset for a single epoch.

    This function puts the model in evaluation mode and performs:
    - Forward passes through the test dataset
    - Loss calculations
    - Accuracy measurements
    - Batch-wise progress tracking

    Args:
        model: PyTorch model to be evaluated
        dataloader: DataLoader containing test data
        loss_fn: Loss function for evaluation
        device: Target device (GPU/CPU) for computations
        baseMask: Base mask tensor
        batchSize: Size of each test batch
        Length: Dimension size for mask generation

    Returns:
        Tuple of (test_loss, test_acc):
            - test_loss: Average loss across all test batches
            - test_acc: Average accuracy across all test batches
    """
    print("test step")
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize test metrics
    test_loss, test_acc = 0, 0
    
    # Use inference mode for better performance during testing
    with torch.inference_mode():
        # Process each batch in the test dataloader
        for batch, (vehImage, uavImage, compassTheta, BBoxCenters) in enumerate(dataloader):
            # Move all tensors to the target device
            vehImage, uavImage, baseMask = vehImage.to(device), uavImage.to(device), baseMask.to(device)
            compassTheta, BBoxCenters = compassTheta.to(device), BBoxCenters.to(device)

            # Modify baseMask shape based on batch size
            currentBatchSize = len(compassTheta)
            if (batchSize!=currentBatchSize):
                baseMask = baseMaskGenerator(Length, device, currentBatchSize).to(device)
            
            # Forward pass through the network
            uavView_descriptors, vehView_descriptor = model(vehImage, uavImage, compassTheta, BBoxCenters, baseMask)
            
            # Calculate loss for this batch
            loss = loss_fn(uavView_descriptors, vehView_descriptor)
             
            # Accumulate loss
            test_loss += loss.item()
            
            # Calculate batch accuracy
            # Filter out-of-bounds bounding boxes
            maskIdxFilter = torch.logical_or(
                torch.logical_or(BBoxCenters[:,:,0]<4, BBoxCenters[:,:,0]>1020),
                torch.logical_or(BBoxCenters[:,:,1]<4, BBoxCenters[:,:,1]>1020)
            )
            
            # Prepare descriptors for similarity calculation
            # Reshape and normalize the vehicle view descriptor
            vehView_descriptor = torch.nn.functional.normalize(
                vehView_descriptor.permute(0,3,2,1).reshape(currentBatchSize, 1, vehView_descriptor.shape[1]*4),
                p=2, dim=2
            )
            
            # Reshape and normalize the UAV view descriptors
            uavView_descriptors = torch.nn.functional.normalize(
                uavView_descriptors.permute(0,2,3,1).reshape(currentBatchSize,uavView_descriptors.shape[2],uavView_descriptors.shape[1]*4),
                p=2, dim=2
            )
            
            # Calculate cosine similarity between views
            similarity = torch.sum(uavView_descriptors*vehView_descriptor, dim=2)
            
            # Mark out-of-FOV vehicles
            similarity[maskIdxFilter] = -2
            
            # Calculate batch accuracy (target vehicle is at the first entry)
            test_acc += (torch.argmax(similarity, dim=1) == 0).sum().item()/len(similarity)
            
            # Print progress every 10 batches
            if batch % 10 == 0:
                print(f"batch:{batch}, acc:{test_acc/(batch+1): .4f}, loss:{test_loss/(batch+1): .4f}")
    
    # Calculate average metrics across all batches
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc

def f(x: np.float32, tangent: np.float32, x0: tuple((np.float32, np.float32))) -> np.float32:
    """
    Computes a point on a line given its equation parameters.

    This function implements the point-slope form of a line equation:
    y = mx + b
    where:
    - m is the slope (tangent)
    - b is the y-intercept (derived from x0)
    - x0 is a point on the line (x0[0], x0[1])

    Args:
        x: x-coordinate at which to evaluate the line
        tangent: Slope of the line
        x0: Tuple containing (x, y) coordinates of a point on the line

    Returns:
        np.float32: y-coordinate of the point on the line at x

    Example:
        >>> f(2.0, 3.0, (1.0, 4.0))
        10.0  # Because: (3 * 2) - (3 * 1) + 4 = 10
    """
    return (tangent * x) - (tangent * x0[0]) + x0[1]
    
def baseMaskGenerator(Length: int, device: torch.device, batchSize: int) -> torch.tensor:
    """
    Generates a base mask tensor for cross-view matching.

    Creates a triangular mask centered in a 3x3 expanded space relative to the input Length.
    The mask is then repeated for the specified batch size.
    Expanding is to make sure that the generated mask can completely cover the feature map,
    especially after rotation and translation.

    Args:
        Length: Size of the base dimension (will be expanded to 3xLength)
        device: Target device for tensor creation
        batchSize: Number of times to repeat the mask

    Returns:
        torch.tensor: Batch of base masks with shape (batchSize, 1, 3*Length, 3*Length)
    """
    # Define mask dimensions (3x expansion for rotation/translation space)
    numRows = 3 * Length
    numCols = 3 * Length
    
    # Initialize zero-filled mask
    space = torch.zeros((numRows, numCols), dtype=torch.float32, device=device)
    
    # Calculate center coordinates
    centerRow = round(numRows / 2)
    centerCol = round(numCols / 2)
    center = (centerCol - 1, centerRow - 1)
    
    # Define boundary tangents for the triangular mask faced North (based on North in the dataset)
    tangent1 = 1  # Upper boundary slope
    tangent2 = -1  # Lower boundary slope
    
    # Fill mask with 1s within triangular region
    for col in np.arange(numCols):
        if col > centerCol - 1:
            continue
        for row in np.arange(numRows):
            if f(col, tangent1, center) < row and f(col, tangent2, center) > row:
                space[row, col] = 1
    
    # Create the batch dimension and return
    baseMask = torch.repeat_interleave(
        space.unsqueeze(dim=0).unsqueeze(dim=0), 
        batchSize, 
        dim=0
    )
    return baseMask
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          Length: int,
          batchSize: int,
          device: torch.device,
          saving_dir: str,
          lambdaReg: float) -> Dict[str, List]:
    """
    Orchestrates the complete training and testing process for a PyTorch model.

    This function manages the entire training lifecycle, including:
    - Training and testing loops
    - Model checkpointing
    - Metric tracking
    - Progress monitoring

    Args:
        model: PyTorch model to be trained and tested
        train_dataloader: DataLoader for training data
        test_dataloader: DataLoader for testing data
        optimizer: Optimizer instance for parameter updates
        loss_fn: Loss function for both training and testing
        epochs: Number of training epochs
        Length: Dimension size for base mask generation
        batchSize: Size of each training batch
        device: Target device (GPU/CPU) for computations
        saving_dir: Directory path for saving checkpoints
        lambdaReg: L2 regularization strength (0 to disable)

    Returns:
        Dict[str, List]: Dictionary containing training and testing metrics
            - train_loss: List of training losses per epoch
            - train_acc: List of training accuracies per epoch
            - test_loss: List of testing losses per epoch
            - test_acc: List of testing accuracies per epoch

    Example return format (2 epochs):
        {
            'train_loss': [2.0616, 1.0537],
            'train_acc': [0.3945, 0.3945],
            'test_loss': [1.2641, 1.5706],
            'test_acc': [0.3400, 0.2973]
        }
    """
    print("main loop")
    
    # Initialize results tracking dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # Generate base mask for cross-view matching
    baseMask = baseMaskGenerator(Length, device, batchSize)
    
    # Train for the specified number of epochs with the progress bar
    for epoch in tqdm(range(epochs)):
        # Perform training step
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            baseMask=baseMask,
            batchSize=batchSize,
            Length = Length,
            lambdaReg=lambdaReg
        )
        
        # Save current checkpoint
        save_checkpoint(
            saving_dir=saving_dir,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            device=device
        )
        
        # Perform testing step
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            baseMask=baseMask,
            batchSize=batchSize,
            Length = Length
        )
        
        # Print current epoch metrics
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    return results
    
def predict(model: torch.nn.Module,
            vehImage: torch.tensor,
            uavImage: torch.tensor,
            compassTheta: torch.tensor,
            BBoxCenters: torch.tensor,
            Length: int,
            device: torch.device,
            batchSize: int) -> Tuple[torch.tensor, torch.tensor]:
    """
    Makes predictions using a trained cross-view matching model.

    This function performs inference using a trained model.
    It handles device management, model evaluation mode, and similarity calculations.

    Args:
        model: Trained PyTorch model for cross-view matching
        vehImage: Vehicle camera image tensor
        uavImage: UAV camera image tensor
        compassTheta: Vehicle orientation angle tensor
        BBoxCenters: Bounding box centers tensor
        Length: Dimension size for base mask generation
        device: Target device (GPU/CPU) for computations
        batchSize: Batch size for inference

    Returns:
        Tuple of:
            - BBoxCenters[0,predictedIdx,:]: Coordinates of the matched vehicle
            - predictedIdx: Index of the matched vehicle (0-based)

    Example:
        >>> predicted_box, idx = predict(model, veh_image, uav_image, theta, boxes, 640, device, 1)
        >>> print(predicted_box.shape)  # Should be torch.Size([2]) - [x, y] coordinates
    """
    # Generate base mask for cross-view matching
    baseMask = baseMaskGenerator(Length, device, batchSize)
    # Modify baseMask shape based on batch size
    currentBatchSize = len(compassTheta)
    if (batchSize!=currentBatchSize):
        baseMask = baseMaskGenerator(Length, device, currentBatchSize)
        batchSize = currentBatchSize
    
    # Set model to evaluation mode for inference
    model.eval()
    
    # Use inference mode for better performance during prediction
    with torch.inference_mode():
        # Move all tensors to the target device
        vehImage, uavImage, baseMask = vehImage.to(device), uavImage.to(device), baseMask.to(device)
        compassTheta, BBoxCenters = compassTheta.to(device), BBoxCenters.to(device)
        
        # Forward pass through the network
        uavView_descriptors, vehView_descriptor, veh_Time, uav_extract_Time, uav_encoder_Time = model(vehImage, uavImage, compassTheta, BBoxCenters, baseMask)
        
        # Prepare descriptors for similarity calculation
        # Reshape and normalize the vehicle view descriptor
        vehView_descriptor = torch.nn.functional.normalize(
            vehView_descriptor.permute(0,3,2,1).reshape(batchSize, 1, vehView_descriptor.shape[1]*4),
            p=2, dim=2
        )
        
        # Reshape and normalize the UAV view descriptors
        uavView_descriptors = torch.nn.functional.normalize(
            uavView_descriptors.permute(0,2,3,1).reshape(batchSize,uavView_descriptors.shape[2],uavView_descriptors.shape[1]*4),
            p=2, dim=2
        )
        
        # Calculate cosine similarity between views
        maskIdxFilter = torch.logical_or(
            torch.logical_or(BBoxCenters[:,:,0]<4, BBoxCenters[:,:,0]>1020),
            torch.logical_or(BBoxCenters[:,:,1]<4, BBoxCenters[:,:,1]>1020)
        )
        similarity = torch.sum(uavView_descriptors*vehView_descriptor, dim=2)
        
        # Mark out-of-FOV vehicles
        similarity[maskIdxFilter] = -2
        
        # Get the index of the highest similarity (predicted match)
        predictedIdx = torch.argmax(similarity, dim=1)
    
    # Return the matched vehicle's coordinates and its index
    return BBoxCenters[0,predictedIdx,:], predictedIdx, veh_Time, uav_extract_Time, uav_encoder_Time
