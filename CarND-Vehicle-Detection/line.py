import numpy as np

class Line():
    def __init__(self, n):
        """
        The line class to keep track of polynomial lines fitted for the lastest n frames
        """
        self.n = n
        self.detected = False

        # Polynomial coefficients: x = E[i][0]*y^2 + E[i][1]*y + E[i][2]
        self.E = []
        # Average of each coefficients for the latest n frames
        self.avgE0 = 0.
        self.avgE1 = 0.
        self.avgE2 = 0.

    def getAvg(self):
        return (self.avgE0, self.avgE1, self.avgE2)

    def add(self, new_poly):
        """
        Take the coefficients from the latest frame, append them into the list, and average each for the n
        """
        # Check if the queue is full
        flag_q = len(self.E) >= self.n

        # Append the new polynomial line
        self.E.append(new_poly)

        # Pop from the beginning if full
        if flag_q:
            _ = self.E.pop(0)

        # Simple average of line coefficients
        self.avgE0 = np.mean([e[0] for e in self.E])
        self.avgE1 = np.mean([e[1] for e in self.E])
        self.avgE2 = np.mean([e[2] for e in self.E])

        return (self.avgE0, self.avgE1, self.avgE2)