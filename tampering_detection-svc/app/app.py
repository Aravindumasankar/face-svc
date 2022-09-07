from flask import Flask, render_template, Response
import cv2
import numpy as np


app = Flask(__name__)


def gen_frames():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    kernel = np.ones((5, 5), np.uint8)
    while (True):
        ret, frame = cap.read()
        if (frame is None):
            print("End of frame")
            break;
        else:
            a = 0
            bounding_rect = []
            fgmask = fgbg.apply(frame)
            fgmask = cv2.erode(fgmask, kernel, iterations=5)
            fgmask = cv2.dilate(fgmask, kernel, iterations=5)
            # _,contours,_ = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            for i in range(0, len(contours)):
                bounding_rect.append(cv2.boundingRect(contours[i]))
            for i in range(0, len(contours)):
                if bounding_rect[i][2] >= 40 or bounding_rect[i][3] >= 40:
                    a = a + (bounding_rect[i][2]) * bounding_rect[i][3]
                if (a >= int(frame.shape[0]) * int(frame.shape[1]) / 3):
                    cv2.putText(frame, "TAMPERING DETECTED", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
            ret, buffer = cv2.imencode('.jpg', frame)
            final_frame = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + final_frame + b'\r\n')



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=8000, debug=True)