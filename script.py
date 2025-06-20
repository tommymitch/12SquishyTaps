import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ximu3csv
from aeon.utils.discovery import all_estimators

all_estimators("classifier", tag_filter={"algorithm_type": "convolution"})

from aeon.classification.convolution_based import (
    Arsenal,
    HydraClassifier,
    MiniRocketClassifier,
    MultiRocketClassifier,
    MultiRocketHydraClassifier,
    RocketClassifier,
)
from sklearn.metrics import accuracy_score


@dataclass(frozen=True)
class Tap:
    surface: str
    seconds: np.ndarray
    timeseries: np.ndarray
    fft_time: np.ndarray
    fft_frequencies: np.ndarray
    fft_dbs: np.ndarray


# fft: Optional[np.ndarray]


def csv_to_taps(surface: str):
    seconds, accelerometer = read_data(surface)
    return get_taps(surface, seconds, accelerometer)


def read_data(surface: str) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", surface)
    devices = ximu3csv.read(path, ximu3csv.DataMessageType.INERTIAL)

    device = ximu3csv.zero_first_timestamp(devices)[0]

    seconds = device.inertial.timestamp / 1e6

    accelerometer = device.inertial.accelerometer.xyz
    return seconds, accelerometer


def get_taps(surface: str, seconds, accelerometer: np.ndarray) -> List[Tap]:
    magnitude = np.linalg.norm(accelerometer, axis=1)

    return [
        Tap(
            surface,
            seconds[s:e],
            magnitude[s:e],
            *fft(seconds[s:e], magnitude[s:e]),  # * = unpack and map tuple to Tap members: fft_time, fft_frequencies, fft_dbs
        )
        for s, e in detect_impacts(seconds, magnitude)
    ]


def detect_impacts(seconds: np.ndarray, magnitude: np.ndarray) -> List[Tuple[int, int]]:
    sample_rate = 1 / np.median(np.diff(seconds))
    THRESHOLD = 3  # 3 g
    HOLDOFF = int(sample_rate / 2)  # 500 ms
    ATTACK = int(sample_rate / 20)  # 50 ms
    DECAY = int(sample_rate / 10)  # 100 ms

    threshold_exceeded = magnitude > THRESHOLD

    threshold_exceeded = np.maximum.reduce([np.roll(threshold_exceeded, i) for i in range(HOLDOFF + 1)], axis=0)  # extend true values forward in time by IMPACT_HOLDOFF period

    threshold_indices = np.where((threshold_exceeded & ~np.roll(threshold_exceeded, 1)))[0]  # index of each false to true transition

    start_indices = threshold_indices - ATTACK
    end_indices = threshold_indices + DECAY

    if True:
        plt.plot(seconds, magnitude)
        plt.plot([seconds[0], seconds[-1]], [THRESHOLD, THRESHOLD])
        for index, (impact_start, impact_end) in enumerate(zip(start_indices, end_indices)):
            plt.fill_between([seconds[impact_start], seconds[impact_end]], np.min(magnitude), np.max(magnitude), color="tab:green", alpha=0.2)
        plt.show()

    return [(s, e) for s, e in zip(start_indices, end_indices)]


def fft(seconds: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    sample_rate = 1 / np.median(np.diff(seconds))

    fft_size = 64
    hop_size = 16
    window = np.hanning(fft_size)

    number_of_windows = int((len(magnitude) - fft_size) / (fft_size - hop_size)) + 1

    stft = np.empty((fft_size, number_of_windows), dtype=complex)

    for index in range(number_of_windows):
        start = index * (fft_size - hop_size)  # why is ths not index * hop_size
        end = start + fft_size

        stft[:, index] = np.fft.fft(magnitude[start:end] * window)

    times = (np.arange(number_of_windows) * (fft_size - hop_size)) / sample_rate

    frequencies = np.fft.fftfreq(fft_size, 1.0 / sample_rate)[: fft_size // 2]  # only positive frequencies
    dbs = 20 * np.log10(np.abs(stft[: fft_size // 2, :]))

    # plt.pcolormesh(times, frequencies, dbs, shading="auto")
    # plt.show()
    # print(dbs.shape)
    return times, frequencies, dbs


#   return np.empty((32, 135))  # e.g. 32 frequencies, 135 time steps

# plt.plot(seconds, magnitude)
# plt.show()

surfaces = ["soft", "medium", "hard"]

desk_taps = csv_to_taps(surfaces[0])
bubble_taps = csv_to_taps(surfaces[1])
hand_taps = csv_to_taps(surfaces[2])


def even_items(list: List[Tap]) -> List[Tap]:
    return [l for i, l in enumerate(list) if i % 2 == 0]


def odd_items(list: List[Tap]) -> List[Tap]:
    return [l for i, l in enumerate(list) if i % 2 != 0]


desk_taps_train = even_items(desk_taps)
desk_taps_test = odd_items(desk_taps)

bubble_taps_train = even_items(bubble_taps)
bubble_taps_test = odd_items(bubble_taps)

hand_taps_train = even_items(hand_taps)
hand_taps_test = odd_items(hand_taps)

motion_train_labels = []
motion_train = []

for tap in desk_taps_train:
    motion_train_labels.append(tap.surface)
    motion_train.append([tap.fft_dbs])

for tap in bubble_taps_train:
    motion_train_labels.append(tap.surface)
    motion_train.append([tap.fft_dbs])

for tap in hand_taps_train:
    motion_train_labels.append(tap.surface)
    motion_train.append([tap.fft_dbs])

motion_train_labels = np.array(motion_train_labels)
motion_train = np.stack(motion_train, axis=0).squeeze()

motion_test_labels = []
motion_test = []

for tap in desk_taps_test:
    motion_test_labels.append(tap.surface)
    motion_test.append([tap.fft_dbs])

for tap in bubble_taps_test:
    motion_test_labels.append(tap.surface)
    motion_test.append([tap.fft_dbs])

for tap in hand_taps_test:
    motion_test_labels.append(tap.surface)
    motion_test.append([tap.fft_dbs])

motion_test_labels = np.array(motion_test_labels)
motion_test = np.stack(motion_test, axis=0).squeeze()


rocket = RocketClassifier()
rocket.fit(motion_train, motion_train_labels)

y_pred = rocket.predict(motion_test)
accuracy = accuracy_score(motion_test_labels, y_pred)

print(y_pred)
print(accuracy)

for surface in surfaces:
    seconds, accelerometer = read_data(surface)
    taps = get_taps(surface, seconds, accelerometer)
    figure, axes = plt.subplots(nrows=2, sharex=True)

    for tap in taps:
        axes[0].plot(tap.seconds, tap.timeseries)
        axes[1].pcolormesh(tap.fft_time + tap.seconds[0], tap.fft_frequencies, tap.fft_dbs, shading="auto")

    plt.show()
