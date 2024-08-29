from fastapi import FastAPI, WebSocket
import json

import numpy as np 
from sklearn.decomposition import FastICA
from scipy.signal import butter, lfilter, find_peaks

fs = 30
low = 1 # 60bpm
high = 2 # 120bpm
bins = 1000
lw = 1
alpha = 0.8

g_avg = np.zeros((bins,),dtype=np.float32)
b_avg = np.zeros((bins,),dtype=np.float32)
r_avg = np.zeros((bins,),dtype=np.float32)

g_data, r_data, b_data = [],[],[]
s2 = [] #used to store data after hanning window convolution
#%%
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

ham = np.hamming(100)
arr = []
bpm_arr = np.zeros(2000)
ica = FastICA(n_components=3)
loop_count = 0
window_size = 100
hann_window = np.hanning(window_size)

def get_ica(data):
    ica = FastICA(n_components=3)
    s = ica.fit(data)
    return ica

peak_arr = np.zeros(2000)

# Create a Socket.IO instance
app = FastAPI()
rgb_arr = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    arr = []
    bpm_arr = np.zeros(500)
    peak_arr = np.zeros(500)
    s2 = []
    while True:
        data = await websocket.receive_text()
        try:
            data_dict = json.loads(data)
            arr.append(data_dict['data'])
            if len(arr) == 200:
                train_arr = arr[80:]
                ica = get_ica(train_arr)
            if len(arr) > 200 and len(arr) <= 1000:
                s = ica.transform(arr)
                print(s)
                s1 = butter_bandpass_filter(s[:,2],low,high,fs)
                s2.append(np.dot(s1[-100:],hann_window))
                bpm_arr[:-1] = bpm_arr[1:]
                bpm_arr[-1] = s2[-1]
                peaks,_ = find_peaks(bpm_arr,height=0)
                peak_arr[peaks] = bpm_arr[peaks]
                await websocket.send_text(json.dumps({"graph": bpm_arr.tolist(), "bpm": -1}))
            if len(arr) == 1000:
                s2 = np.array(s2)
                peaks,_ = find_peaks(s2,height=0)
                bpm = len(peaks) / (len(arr) / 30.0) * 60
                await websocket.send_text(json.dumps({"bpm": bpm}))
        except json.JSONDecodeError:
            await websocket.send_text("Error: Invalid JSON")

# FastAPI route as an example
@app.get("/")
async def root():
    return {"message": "Hello, world!"}
