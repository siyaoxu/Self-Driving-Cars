import numpy as np
from collections import deque

class Heatmap():
    def __init__(self, n, shape):
        """
        A class to keep track of heatmaps of detected objects through the lastest n frames
        """
        self.n = n
        self.detected = False

        # A list of heat maps
        self.maps = deque(maxlen = n)
        # initialize with an empty heatmap
        self.maps.append(np.zeros(shape))
        # initialize the averaged heatmap
        self.avgMap = np.zeros(shape)

    def getAvg(self):
        return (self.avgMap)

    def add(self, newMap):
        """
        Take the heatmap from the latest frame, append them into the list, and average each for the n
        """
        # Check if the queue is full
        flag_q = len(self.maps) >= self.n

        # Append the new map
        self.maps.append(newMap)

        # Pop from the beginning if full
        if flag_q:
            _ = self.maps.popleft()

        # Simple average of line coefficients
        self.avgMap = np.mean(np.stack(self.maps),0)


        return self.avgMap