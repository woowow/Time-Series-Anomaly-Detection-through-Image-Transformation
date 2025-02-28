import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

df = pd.read_csv("test_z_11_ds_10.csv")

# 시계열 데이터가 들어있는 열 이름을 지정
time_column = 'time'
value_column = 'LIT101'

# 시계열 데이터를 numpy 배열로 변환
start_index = 7533
end_index=19581
subset_df = df.loc[start_index:end_index]

#time_series = df[value_column].values
time_series = subset_df[value_column].values

# 시계열 데이터의 길이
n = len(time_series)

# 샘플링 간격 (시간 간격, 여기서는 1로 가정, 필요시 조정)
T = 1.0

# 푸리에 변환 수행
yf = fft(time_series)
xf = fftfreq(n, T)[:n // 2]

# 주파수 스펙트럼의 절대값 계산
amplitude_spectrum = 2.0 / n * np.abs(yf[:n // 2])

# 주파수 0을 무시하고 그 다음으로 높은 피크 찾기
dominant_frequency = xf[1:][np.argmax(amplitude_spectrum[1:])]  # 0 제외
dominant_period = 1 / dominant_frequency

# 결과 출력
print(f"Dominant Frequency: {dominant_frequency} Hz")
print(f"Dominant Period: {dominant_period} time units")

# 주파수 스펙트럼 시각화
plt.figure(figsize=(10, 6))
plt.plot(xf, amplitude_spectrum)
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
plt.savefig("fourier.png")
