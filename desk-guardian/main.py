import depthai as dai
import numpy as np
from depthai_nodes.node import ParsingNeuralNetwork, FrameCropper, GatherData
import time
from guardian_node import GuardianNode

# --- Configuration ---
DEVICE_IP = "169.254.1.212"
YOLO_MODEL = "luxonis/yolov6-nano:r2-coco-512x288"
REID_MODEL = "luxonis/osnet:imagenet-128x256"
FPS = 15

def create_pipeline(device, platform):
    pipeline = dai.Pipeline(device)
    
    # 1. Hardware Nodes
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput(size=(640, 400), type=dai.ImgFrame.Type.GRAY8, fps=FPS),
        right=right.requestOutput(size=(640, 400), type=dai.ImgFrame.Type.GRAY8, fps=FPS),
        presetMode=dai.node.StereoDepth.PresetMode.HIGH_DETAIL
    )
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    
    # 2. Spatial Detection (YOLO)
    yolo_model_desc = dai.NNModelDescription(YOLO_MODEL, platform=platform)
    yolo_archive = dai.NNArchive(dai.getModelFromZoo(yolo_model_desc))
    
    spatial_nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        input=cam,
        stereo=stereo,
        nnArchive=yolo_archive,
        fps=float(FPS)
    )
    
    # 3. Re-Identification Pipeline
    reid_model_desc = dai.NNModelDescription(REID_MODEL, platform=platform)
    reid_archive = dai.NNArchive(dai.getModelFromZoo(reid_model_desc))
    
    # Crop detections
    crop_node = pipeline.create(FrameCropper).fromImgDetections(
        inputImgDetections=spatial_nn.out,
        outputSize=(reid_archive.getInputWidth(), reid_archive.getInputHeight())
    ).build(
        inputImage=cam.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.BGR888i, fps=FPS)
    )
    
    # OSNet Node
    reid_nn = pipeline.create(dai.node.NeuralNetwork).build(
        input=crop_node.out,
        nnArchive=reid_archive
    )
    
    # 4. Synchronization and Logic
    gather_node = pipeline.create(GatherData).build(
        cameraFps=float(FPS),
        inputData=reid_nn.out,
        inputReference=spatial_nn.out
    )
    
    guardian_node = pipeline.create(GuardianNode).build(gather_node.out)
    
    # Returns the pipeline, the logic node, and the context video output
    video_out = cam.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.NV12, fps=FPS)
    
    return pipeline, guardian_node, video_out

def main():
    device_info = dai.DeviceInfo(DEVICE_IP)
    
    with dai.Device(device_info) as device:
        platform = device.getPlatformAsString()
        print(f"Connected to {DEVICE_IP} (Platform: {platform})")
        
        pipeline, guardian_node, video_out = create_pipeline(device, platform)
        
        visualizer = dai.RemoteConnection(httpPort=8082)
        visualizer.addTopic("Desk Guardian", guardian_node.out)
        visualizer.addTopic("Video", video_out)
        
        pipeline.start()
        visualizer.registerPipeline(pipeline)
        
        print("Desk Guardian running...")
        print("Visualizer available at http://localhost:8082")
        
        while pipeline.isRunning():
            key = visualizer.waitKey(1)
            if key == ord('q'):
                break
            time.sleep(0.01)

if __name__ == "__main__":
    main()
