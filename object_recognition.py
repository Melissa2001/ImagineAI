    import cv2
    import pyttsx3
    import numpy as np
    from pathlib import Path
    import sys
    import torch
    import time  # Import the time module

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Path to YOLOv5 weights
    weights_path = r"E:Trial\Chatgpt\yolov5\yolov5s.pt"  # Update with the path to the downloaded YOLOv5 weigwwwwwwwwwwwwhts

    # Load the pre-trained YOLOv5 object detection model
    sys.path.append(str(Path(weights_path).parents[0]))  # add yolov5/ to path
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device

    # Load YOLOv5 model
    device = select_device('')
    model = attempt_load(weights_path)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    # Track recognized objects and cooldown time
    recognized_objects = set()
    cooldown_time = 15  # in seconds
    last_recognition_time = time.time()

    # Function to detect objects in a frame and convert the results to speech
    def detect_objects(frame):
        global recognized_objects, last_recognition_time

        # Check if cooldown time has elapsed
        if time.time() - last_recognition_time >= cooldown_time:
            recognized_objects.clear()

        # Resize frame to the model input size
        img = cv2.resize(frame, (640, 640))
        img = img[..., ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Convert image to torch tensor
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Predict
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):
            # If detections exist
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

                # Loop through detections and draw bounding boxes
                for *xyxy, conf, cls in reversed(det):
                    class_name = names[int(cls)]
                    label = f'{class_name}'

                    # Check if the object has been recognized recently
                    if class_name not in recognized_objects:
                        recognized_objects.add(class_name)

                        # Convert class name to speech
                        engine.say(class_name)
                        engine.runAndWait()

                        # Update last recognition time
                        last_recognition_time = time.time()

                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Call the function with the webcam
    cap = cv2.VideoCapture(0)  # Use webcam source 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        detect_objects(frame)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()