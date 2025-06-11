
# Citing This Paper
Please cite the following paper if you intend to use this code for your research.<br>
Under review
# Cross View User Identification
In the cross-view user identification approach, the objective is to identify the target vehicle in the image captured by the UAV among other objects and vehicles while the image of the target vehicle is also available.  
You can train the cross-view user identification model through cross_view_model_training.ipynb file.  
dataset.py is used to load and preprocess the dataset.  
In model.py, the model has been implemented.  
model_cuda_timer.py is identical to model.py but with cuda timers added for latency evaluation.  
The loss function and some helper functions have been implemented in loss.py and helperFunctions.py, respectively.  
## Cross View Dataset
The dataset contains six different locations including FiveWays, FourWays, Park, roundabout, straight road, and T-junction.  
Each location has three different lighting conditions including daylight, dusk, and nighttime that are indicated by _0, _1, and _2, respectively.  
Folders and file structures for FiveWays location in daylight conditions have been shown below:
<pre>  
.
└── FiveWays_0/  
    ├── UAV/  
    │   ├── sensor.camera.rgb/  
    │   │   └── CamID_289/  
    │   │       ├── 0.png  
    │   │       ├── :  
    │   │       └── 52.png  
    │   └── CamID_289.csv (UAV camera data)  
    └── Vehicles/  
        ├── VehicleID_109/  
        │   ├── concatenated/  
        │   │   ├── 0.png (concatenated images of four vehicle cameras)  
        │   │   ├── :  
        │   │   └── 52.png (concatenated images of four vehicle cameras)  
        │   ├── CamID_139.csv (vehicle front camera data)  
        │   ├── CamID_140.csv (vehicle right camera data)  
        │   ├── CamID_141.csv (vehicle rear camera data)  
        │   └── CamID_142.csv (vehicle left camera data)  
        ├── VehicleID_109.csv (Vehicle data)  
        ├── :  
        ├── VehicleID_138/  
        │   ├── concatenated/  
        │   │   ├── 0.png (concatenated images of four vehicle cameras)  
        │   │   ├── :  
        │   │   └── 52.png (concatenated images of four vehicle cameras)  
        │   ├── CamID_255.csv (vehicle front camera data)  
        │   ├── CamID_256.csv (vehicle right camera data)  
        │   ├── CamID_257.csv (vehicle rear camera data)  
        │   └── CamID_258.csv (vehicle left camera data)  
        └── VehicleID_138.csv (Vehicle data)  
</pre>
The folder structures are the same for other locations.  
## Introducing of Cameras' CSV File
column name: description  
frame: "frame index" that is identical for all data captured at the same frame.  
location_X: X coordinate of the camera in the global Cartesian coordinate system of CARLA.  
location_Y: Y coordinate of the camera in the global Cartesian coordinate system of CARLA.  
location_Z: Z coordinate of the camera in the global Cartesian coordinate system of CARLA.  
w2c_ij: entry at (row=i,column=j) of world-to-camera transformation matrix converting world coordinates to camera coordinates.  
forwardVector_X:  X element of the vector pointing forward according to the rotation of the camera.  
forwardVector_Y:  Y element of the vector pointing forward according to the rotation of the camera.  
forwardVector_Z:  Z element of the vector pointing forward according to the rotation of the camera.  
rotation_pitch: pitch angle of the camera according to CARLA's axis system.  
rotation_yaw: yaw angle of the camera according to CARLA's axis system.  
rotation_roll: roll angle of the camera according to CARLA's axis system.  
## Introducing of Vehicles' CSV File
column name: description  
frame: "frame index" that is identical for all data captured at the same frame.  
theta: The heading angle of the vehicle relative to North.  
location_X: X coordinate of the vehicle in the global Cartesian coordinate system of CARLA.  
location_Y: Y coordinate of the vehicle in the global Cartesian coordinate system of CARLA.  
location_Z: Z coordinate of the vehicle in the global Cartesian coordinate system of CARLA.  
ImageBBoxCenter_X: X-coordinate of the vehicle bounding box center on the 1024x1024 image captured by the UAV camera in pixels.  
ImageBBoxCenter_Y: Y-coordinate of the vehicle bounding box center on the 1024x1024 image captured by the UAV camera in pixels.  
ImageBBoxVertex0_X: X-coordinate of the vehicle bounding box vertex0 on the 1024x1024 image captured by the UAV camera in pixels. (Vehicle 3D bounding box projected on 2D the UAV image)  
ImageBBoxVertex0_Y: Y-coordinate of the vehicle bounding box vertex0 on the 1024x1024 image captured by the UAV camera in pixels. (Vehicle 3D bounding box projected on 2D the UAV image)  
ImageBBoxVertex7_X: X-coordinate of the vehicle bounding box vertex7 on the 1024x1024 image captured by the UAV camera in pixels. (Vehicle 3D bounding box projected on 2D the UAV image)  
ImageBBoxVertex7_Y: Y-coordinate of the vehicle bounding box vertex7 on the 1024x1024 image captured by the UAV camera in pixels. (Vehicle 3D bounding box projected on 2D the UAV image)  
BBoxCenter_X = X-coordinate of the vehicle bounding box center in the global Cartesian coordinate system of CARLA.  
BBoxCenter_Y = Y-coordinate of the vehicle bounding box center in the global Cartesian coordinate system of CARLA.  
BBoxCenter_Z = Z-coordinate of the vehicle bounding box center in the global Cartesian coordinate system of CARLA.  
BBoxVertex0_X = X-coordinate of the vehicle bounding box vertex0 in the global Cartesian coordinate system of CARLA. (3D bounding box provided by CARLA)  
BBoxVertex0_Y = Y-coordinate of the vehicle bounding box vertex0 in the global Cartesian coordinate system of CARLA. (3D bounding box provided by CARLA)  
BBoxVertex0_Z = Z-coordinate of the vehicle bounding box vertex0 in the global Cartesian coordinate system of CARLA. (3D bounding box provided by CARLA)  
BBoxVertex7_X = X-coordinate of the vehicle bounding box vertex7 in the global Cartesian coordinate system of CARLA. (3D bounding box provided by CARLA)  
BBoxVertex7_Y = Y-coordinate of the vehicle bounding box vertex7 in the global Cartesian coordinate system of CARLA. (3D bounding box provided by CARLA)  
BBoxVertex7_Z = Z-coordinate of the vehicle bounding box vertex7 in the global Cartesian coordinate system of CARLA. (3D bounding box provided by CARLA)  
forwardVector_X:  X element of the vector pointing forward according to the rotation of the vehicle.  
forwardVector_Y:  Y element of the vector pointing forward according to the rotation of the vehicle.  
forwardVector_Z:  Z element of the vector pointing forward according to the rotation of the vehicle.  
rotation_pitch: pitch angle of the vehicle according to CARLA's axis system.  
rotation_yaw: yaw angle of the vehicle according to CARLA's axis system.  
rotation_roll: roll angle of the vehicle according to CARLA's axis system.  
## CARLA Website
For more information about collected parameters by CARLA, please visit:
[CARLA](https://carla.org/)
# YOLOv11 for Oriented Bounding Boxes Object Detection
You can train the model through yolo_obb_model_training.ipynb file.  
## YOLO OBB Dataset
You might need to modify the path field in data.yalm based on the dataset directory on your system.  
data.yalm is provided with the dataset.  