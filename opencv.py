import cv2
import tqdm

def opencv_img(cvNet, img_path):
    img = cv2.imread(img_path)
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

    cv2.imwrite("result.png", img)

def opencv_video(cvNet, vid_path):
    cap = cv2.VideoCapture(vid_path)
    codec = cv2.VideoWriter_fourcc(*'MP4V')
    vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    vid_writer = cv2.VideoWriter('result.mp4', codec, vid_fps, vid_size)

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm.tqdm(total=frame_cnt)
    while True:
        hasFrame, img_frame = cap.read()
        if not hasFrame:
            break

        rows = img_frame.shape[0]
        cols = img_frame.shape[1]
        cvNet.setInput(cv2.dnn.blobFromImage(img_frame, size=(300, 300), swapRB=True, crop=False))
        cvOut = cvNet.forward()

        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > 0.3:
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                cv2.rectangle(img_frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

        vid_writer.write(img_frame)
        pbar.update(1)
    vid_writer.release()
    cap.release()

if __name__ == "__main__":
    pb_path = '../pretrained/faster_rcnn/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb'
    pbtxt_path = '../pretrained/faster_rcnn/faster_rcnn_resnet50_coco_2018_01_28.pbtxt'
    cvNet = cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)

    opencv_img(cvNet, 'example.jpg')
    opencv_video(cvNet, 'example.mp4')