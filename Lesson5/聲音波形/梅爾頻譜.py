import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 讀取音訊檔案
y, sr = librosa.load('01.wav', sr=None)

# 計算梅爾頻譜
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)

# 對梅爾頻譜取對數
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

# 繪製梅爾頻譜
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Mel Frequency')
plt.tight_layout()
plt.show()
