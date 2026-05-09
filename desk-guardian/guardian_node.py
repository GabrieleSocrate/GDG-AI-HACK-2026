import numpy as np
import depthai as dai
import traceback

class GuardianNode(dai.node.HostNode):
    """
    Custom HostNode for Desk Guardian security logic.
    Handles enrollment, matching, and spatial proximity alerts.
    """
    def __init__(self) -> None:
        super().__init__()
        self.owner_embedding = None
        self.enrollment_buffer = []
        self.enrollment_size = 20
        self.cos_sim_threshold = 0.6
        self.touch_threshold_mm = 400.0
        
        # State: 0: ENROLLMENT, 1: AUTHORIZED, 2: ARMED, 3: ALARM
        self.state = 0
        self.state_labels = ["ENROLLMENT", "AUTHORIZED", "ARMED", "ALARM"]
        
        self.assets = {} # label -> (x, y, z)
        self.last_owner_seen = 0
        self.owner_absent_timeout = 5.0
        
        # COCO class mapping for YOLOv6 nano
        self.target_classes = {0: "person", 63: "laptop", 67: "cell phone"}

    def build(self, gather_data_msg) -> "GuardianNode":
        self.link_args(gather_data_msg)
        return self

    def _cos_sim(self, A, B):
        A = A.flatten()
        B = B.flatten()
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    def process(self, gather_data_msg) -> None:
        try:
            # 1. Extract Data
            spatial_dets: dai.SpatialImgDetections = gather_data_msg.reference_data
            rec_msg_list = gather_data_msg.items
            
            now_sec = dai.Clock.now().total_seconds()
            
            # 2. Map Assets (Laptop/Phone)
            current_people = []
            for det in spatial_dets.detections:
                if det.label == 0: # person
                    current_people.append(det)
                elif det.label in [63, 67]: # laptop, phone
                    label = self.target_classes[det.label]
                    self.assets[label] = (det.spatialCoordinates.x, det.spatialCoordinates.y, det.spatialCoordinates.z)
            
            # 3. Handle People & Re-ID
            owner_present = False
            reid_idx = 0
            
            for det in current_people:
                if reid_idx < len(rec_msg_list):
                    embedding = rec_msg_list[reid_idx].getTensor("output", dequantize=True)
                    reid_idx += 1
                    
                    if self.state == 0: # ENROLLMENT
                        self.enrollment_buffer.append(embedding)
                        det.labelName = "ENROLLING..."
                        if len(self.enrollment_buffer) >= self.enrollment_size:
                            self.owner_embedding = np.mean(self.enrollment_buffer, axis=0)
                            self.state = 1
                            self.last_owner_seen = now_sec
                            print("Enrollment Complete. Owner Registered.")
                    
                    elif self.owner_embedding is not None:
                        sim = self._cos_sim(embedding, self.owner_embedding)
                        if sim > self.cos_sim_threshold:
                            det.labelName = f"OWNER ({sim:.2f})"
                            owner_present = True
                            self.last_owner_seen = now_sec
                        else:
                            det.labelName = f"UNKNOWN ({sim:.2f})"
                            self._check_proximity(det)
                else:
                    det.labelName = "PERSON (NO REID)"

            # 4. State Machine Updates
            if self.state == 1 and not owner_present:
                if now_sec - self.last_owner_seen > self.owner_absent_timeout:
                    self.state = 2
                    print("System ARMED. Owner absent.")
            
            if self.state == 2 and owner_present:
                self.state = 1
                print("System DISARMED. Owner present.")

            # Final label cleanup and status overlay
            status_msg = f"STATE: {self.state_labels[self.state]}"
            for i, det in enumerate(spatial_dets.detections):
                base_label = getattr(det, 'labelName', self.target_classes.get(det.label, str(det.label)))
                if i == 0:
                    det.labelName = f"[{status_msg}] {base_label}"
                else:
                    det.labelName = base_label

            self.out.send(spatial_dets)
            
        except Exception as e:
            print(f"Error in GuardianNode: {e}")
            traceback.print_exc()

    def _check_proximity(self, person_det):
        if not self.assets:
            return
            
        p_pos = np.array([person_det.spatialCoordinates.x, person_det.spatialCoordinates.y, person_det.spatialCoordinates.z])
        for asset_name, a_pos in self.assets.items():
            dist = np.linalg.norm(p_pos - np.array(a_pos))
            if dist < self.touch_threshold_mm:
                if self.state == 2: # ARMED
                    self.state = 3 # ALARM
                    print(f"!!! ALARM !!! Unauthorized approach to {asset_name} (Dist: {dist:.0f}mm)")
                person_det.labelName = f"INTRUDER! Dist: {dist/10:.0f}cm to {asset_name}"
