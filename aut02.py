import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def measure_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_count = cv2.countNonZero(edges)
    return edges, edge_count

# リアルタイムプロット
plt.ion()
fig, ax = plt.subplots()
x_data = deque(maxlen=100)
y_data = deque(maxlen=100)
line, = ax.plot(x_data, y_data)
ax.set_ylim(0, 10000) 
ax.set_title("edge count to frame")
ax.set_xlabel("frame")
ax.set_ylabel("Edge Count")

# カメラの初期化
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 0)

# エッジがMAXのもの
ret, first_img = cap.read()

# 最初の画像のエッジ
first_edges, first_edge_value = measure_edge(first_img)
print("First Edge Value:", first_edge_value)

frame_count = 0
edge_values = []  

while cap.isOpened():
    ret, img = cap.read()
    
    if not ret:
        break
    
    # エッジ計算
    edges, edge_value = measure_edge(img)
    # scaled_edge_value = edge_value / 10  # エッジ数を10分の1にスケール
    scaled_edge_value = edge_value   
    
    # エッジ検出値を表示
    #print("Frame:", frame_count, "Edge:", edge_value, "Scaled:", scaled_edge_value)
    
    # エッジ画像を表示
    #cv2.imshow('Edges', edges)
    
    # リアルタイムグラフの更新
    x_data.append(frame_count)
    y_data.append(scaled_edge_value)
    line.set_xdata(x_data)
    line.set_ydata(y_data)
    ax.set_xlim(0, max(100, frame_count))
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # エッジ数をリストに追加
    edge_values.append(edge_value)
    
    # # 100フレームごとに最大値を表示
    # if (frame_count + 1) % 100 == 0:
    #     max_edge_value = max(edge_values[-100:]) 
    #     print(f"Max Edge Value in last 100 frames: {max_edge_value}")
    
    if (frame_count + 1) % 50 == 0:
        max_edge_value = max(edge_values[-50:])
        print(f"Max edge  in last 50 frames: {max_edge_value}")
    
    frame_count += 1
    
    # 終了
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# オートフォーカスをONで終了
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cv2.destroyAllWindows()
cap.release()
plt.ioff()
plt.show()
