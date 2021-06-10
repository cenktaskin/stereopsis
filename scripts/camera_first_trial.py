import cv2
from pathlib import Path

# %%
img_path = Path.cwd().joinpath('images')

index = 2
arr = []
while True:
    cap = cv2.VideoCapture(index)
    if not cap.read()[0]:
        break
    else:
        arr.append(index)
    cap.release()
    index += 1
print(arr)

# %%

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports

list_ports()

# %%
video = cv2.VideoCapture(2)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if video.isOpened():
    for i in range(int(1e12)):
        check, frame = video.read()
        if check:
            cv2.imshow("Result", frame)
            #cv2.imwrite(str(img_path.joinpath(str(i) + '.png')), frame)
            if cv2.waitKey(1) & 0xFF == ord('e'):
                cv2.imwrite(str(img_path.joinpath(str(i)+'.png')), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('Frame not available')
            print(video.isOpened())
else:
    print("video not opened")