# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 10:31:17 2021

@author: GaliOMG
"""
import librosa
import librosa.display
import os
import pandas as pd
import numpy as np
import statistics
import scipy.io.wavfile
import math
from pydub import AudioSegment
import matplotlib.pyplot as plt


def paddingShortAudio(file_name, sizeVector):
    files_path = (r'C:\Users\User\PycharmProjects\pythonProject1\train')
    files_path = files_path + '\\' + file_name
    y, sr = librosa.load('train/' + file_name)
    duration = librosa.get_duration(y=y, sr=sr)
    subVectorSize = sizeVector - y.shape[0]
    if subVectorSize % 2 == 0:
        subVectorSize1 = math.floor(subVectorSize / 2)
        subVectorSize2 = math.floor(subVectorSize / 2)
    else:
        subVectorSize1 = math.floor(subVectorSize / 2)
        subVectorSize2 = math.ceil(subVectorSize / 2)
    y = np.pad(y, (subVectorSize1, subVectorSize2), 'constant', constant_values=(0, 0))
    scipy.io.wavfile.write("median/" + file_name + 'newSong.wav', sr, y)


def cutLongAudio(file_name, median):
    files_path = (r'C:\Users\User\PycharmProjects\pythonProject1\train')
    files_path = files_path + '\\' + file_name
    y, sr = librosa.load('train/' + file_name)
    duration = librosa.get_duration(y=y, sr=sr)
    duration = duration / 2
    l1 = np.array([duration - median / 2, duration + median / 2])
    startTime = l1[0] * 1000
    endTime = l1[1] * 1000
    newAudio = AudioSegment.from_wav(files_path)
    newAudio = newAudio[startTime:endTime]
    newAudio.export("median/" + file_name + 'newSong.wav', format="wav")


def calculateMedian():
    directory = (r'C:\Users\User\PycharmProjects\pythonProject1\train')
    arrayDuration = []
    for file in sorted(os.listdir(directory)):
        fileTemp = directory + '\\' + file
        filename = os.fsdecode(fileTemp)
        if filename.endswith(".wav"):
            y, sr = librosa.load(filename)
            arrayDuration.append(librosa.get_duration(y=y, sr=sr))
            continue
        else:
            continue
    median = statistics.median(arrayDuration)

    for file in sorted(os.listdir(directory)):
        fileTemp = directory + '\\' + file
        filename = os.fsdecode(fileTemp)
        y, sr = librosa.load(filename)
        dur = librosa.get_duration(y=y, sr=sr)
        if dur == median:
            vectorLength = y.shape[0]
            break
        else:
            continue
    return median, vectorLength


def melSpec():
    directory = (r'C:\Users\User\PycharmProjects\pythonProject1\medianTest')
    cols = ['file name', 'spectorgama']
    lst = np.array([])
    count = 0
    for file in sorted(os.listdir(directory)):
        print("count " + str(count))
        count += 1
        fileTemp = directory + '\\' + file
        name = os.fsdecode(file)
        label = name[0:2]
        filename = os.fsdecode(fileTemp)
        if filename.endswith(".wav"):
            y, sr = librosa.load(filename)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=35, fmax=8000)
            #fig, ax = plt.subplots()
            S_dB = librosa.power_to_db(S, ref=np.max) #מעביר אמפליטודה לדציבל
            #img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
            #fig.colorbar(img, ax=ax, format='%+2.0f dB')
            #ax.set(title='Mel-frequency spectrogram')
            #fig.savefig(name+'.jpg')
            #plt.close()
            #Sreshape = S.reshape(-1)
            #Sreshape = Sreshape.reshape((Sreshape.shape[0], 1))
            S_dB = S_dB.reshape((1, S_dB.shape[0] * S_dB.shape[1]))
            label = int(label)
            label = np.array(label)
            label = label.reshape((1, 1))

            if count == 1:
                df1 = pd.DataFrame({'label': [label], 'spectorgam': [S_dB]})
            else:
                df2 = pd.DataFrame({'label': [label], 'spectorgam': [S_dB]})
                df1 = pd.concat([df1, df2], axis=0)
            continue
        else:
            continue

    dataframe = pd.DataFrame(df1)
    dataframe.to_pickle("dataframe.pkl")


def checkDuration():
    median, vectorLength = calculateMedian();
    directory = (r'C:\Users\User\PycharmProjects\pythonProject1\train')
    for file in sorted(os.listdir(directory)):
        fileTemp = directory + '\\' + file
        filename = os.fsdecode(fileTemp)
        if filename.endswith(".wav"):
            y, sr = librosa.load(filename)
            duration = librosa.get_duration(y=y, sr=sr)
            if duration >= median:
                cutLongAudio(file, median)
            else:
                paddingShortAudio(file, vectorLength)
        else:
            continue
    melSpec()


melSpec()
