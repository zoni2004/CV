import sys
import time
import cv2
import numpy as np

BOARD = (9, 6)
SQUARE = 1.0
NEEDED = 20
CALIB_FILE = "camera_calib.npz"


def calibrate_from_webcam():
    cap = cv2.VideoCapture(0)
    template = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
    template[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE
    object_points = []
    image_points = []
    last_save = 0
    size = None

    print("Show the chessboard from different angles. Collecting", NEEDED, "views. Press q to stop early.")
    while len(image_points) < NEEDED:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, BOARD)
        if found:
            cv2.drawChessboardCorners(frame, BOARD, corners, found)
            if time.time() - last_save > 1.5:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                object_points.append(template)
                image_points.append(corners)
                last_save = time.time()
        cv2.putText(frame, str(len(image_points)) + "/" + str(NEEDED), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

    if len(image_points) < 5:
        print("Not enough views. Try again.")
        return

    error, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, size, None, None)
    np.savez(CALIB_FILE, K=K, dist=dist)
    print("RMS error:", error)
    print("Camera matrix:\n", K)
    print("Distortion coefficients:\n", dist.ravel())


def show_live():
    saved = np.load(CALIB_FILE)
    K = saved["K"]
    dist = saved["dist"]
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0.5, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fixed = cv2.undistort(frame, K, dist, None, new_K)
        cv2.putText(frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(fixed, "Undistorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live", np.hstack([frame, fixed]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
    calibrate_from_webcam()
else:
    show_live()
