"""
==================================
Data set of Gramian angular fields
==================================

A Gramian angular field is an image obtained from a time series, representing
some kind of temporal correlation between each pair of values from the time
series. Two methods are available: Gramian angular summation field and Gramian
angular difference field.
It is implemented as :class:`pyts.image.GramianAngularField`.

In this example, we consider the training samples of the
`GunPoint dataset <http://timeseriesclassification.com/description.php?Dataset=GunPoint>`_,
consisting of 50 univariate time series of length 150.
The Gramian angular summation field of each time series is independently
computed and the 50 Gramian angular summation fields are plotted.
"""  # noqa:E501

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import os
import time

from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from PIL import Image
from random import *


class CustomGramianAngularField(GramianAngularField):
    def __init__(self, image_size=None, method='summation', overlapping=True, custom_min=None, custom_max=None):
        super().__init__(image_size=image_size, method=method, overlapping=overlapping)
        self.custom_min = custom_min
        self.custom_max = custom_max

    def transform(self, X):
        X = self._check_params(X)
        
        # Custom min and max for normalization
        if self.custom_min is not None and self.custom_max is not None:
            min_val = self.custom_min
            max_val = self.custom_max
        else:
            min_val = X.min(axis=1)[:, None]
            max_val = X.max(axis=1)[:, None]
        
        # Linear scaling
        X = (X - min_val) / (max_val - min_val)
        X = 2 * X - 1
        
        phi = np.arccos(X)

        if self.method == 'summation':
            gaf = np.cos(phi[:, :, None] + phi[:, None, :])
        elif self.method == 'difference':
            gaf = np.sin(phi[:, :, None] - phi[:, None, :])
        return gaf

    def _check_params(self, X):
        X = np.asarray(X)
        n_samples, n_timestamps = X.shape
        if self.image_size is not None and (self.image_size < 1 or self.image_size > n_timestamps):
            raise ValueError("`image_size` must be a positive integer less than or equal to the number of timestamps.")
        if self.method not in ['summation', 'difference']:
            raise ValueError("`method` must be either 'summation' or 'difference'.")
        return X

def main():
        dpi = 100    
        normal = pd.read_csv("time_series_file.csv")

        ab = normal
        window_length = 0
        ds = 10
        end = 44991
        for a in range(1):
                cnt = 1
                window_length = 100

                for j in range (1):
                        plt.cla()
                        i = 1
                        sensor = "Sensor name"

                        min_val = ab.iloc[:,i].min()
                        max_val = ab.iloc[:,i].max()
                        
                        n_ab = ab.iloc[:,i]

                        n_ab_front = pd.DataFrame()

                        
                        scope = (len(n_ab) // window_length)
                        quantity = window_length * scope

                        n_ab_front = n_ab.iloc[0:quantity]
                        n_ab_back = n_ab.iloc[quantity:end]
                        
                        length_n_ab_back = len(n_ab_back)
                                                
                        n_ab_arr = np.array(n_ab_front)
                        n_ab_back_arr = np.array(n_ab_back)
                        

                        n_ab_re_arr = np.reshape(n_ab_arr, (scope, window_length))
                        n_ab_re_back_arr = np.reshape(n_ab_back_arr, (1, length_n_ab_back))
                
                        n_ab = pd.DataFrame(n_ab_re_arr)
                        n_back = pd.DataFrame(n_ab_re_back_arr)                
                        
                        print(sensor)

                        custom_gaf = CustomGramianAngularField(image_size=window_length, method='summation', custom_min=min_val, custom_max=max_val)
                        X_gaf = custom_gaf.fit_transform(n_ab)
                        

                        path = "./"+sensor+"/GGAF"
                        if not os.path.exists(path):
                                os.mkdir(path)

                        for k in range(scope):
                                
                                plt.figure(figsize=(window_length / 100, window_length / 100), dpi=100)
                                plt.imshow(X_gaf[k], cmap='rainbow')
                                plt.axis('off')
                                plt.tight_layout(pad=0)
                                
                                filename = sensor+"_"+str(cnt)
                                
                                plt.savefig(path+'/'+filename, bbox_inches='tight', pad_inches=0)
                                plt.cla()
                                plt.clf()
                                plt.close()
                                
                                temp_filename = path+'/'+filename+'.png'
                                
                                with Image.open(temp_filename) as img:
                                        img = img.resize((window_length, window_length), Image.LANCZOS)
                                        filename = sensor+"_"+str(cnt) + ".png"
                                        img.save(os.path.join(path, filename))
                                cnt = cnt + 1


                        custom_gaf = CustomGramianAngularField(image_size=length_n_ab_back, method='summation', custom_min=min_val, custom_max=max_val)
                        X_gaf = custom_gaf.fit_transform(n_back)

                        
                        plt.figure(figsize=(length_n_ab_back / 100, length_n_ab_back / 100), dpi=100)
                        plt.imshow(X_gaf[0], cmap='rainbow')
                        plt.axis('off')
                        plt.tight_layout(pad=0)

                        filename = sensor+"_"+str(cnt)
                                
                        plt.savefig(path+'/'+filename, bbox_inches='tight', pad_inches=0)
                        plt.cla()
                        plt.clf()
                        plt.close()
                                
                        temp_filename = path+'/'+filename+'.png'
                                
                        with Image.open(temp_filename) as img:
                                img = img.resize((length_n_ab_back, length_n_ab_back), Image.LANCZOS)
                                filename = sensor+"_"+str(cnt) + ".png"
                                img.save(os.path.join(path, filename))
                        
                        del n_ab
                        del n_ab_front
                        del n_ab_back
                        print(cnt)
                        
                        
        
if __name__ == '__main__':
    main()