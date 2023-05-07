import numpy as np


def GaussKernel(size: tuple[int, int], sigma: float) -> np.ndarray:
    w, h = size
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))  # 根据w, h的值生成一个网格的x，y坐标
    center_x, center_y = (w - 1) / 2, (h - 1) / 2
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma**2)
    labels = np.exp(-0.5 * dist)
    return labels


class Mosse:
    def __init__(self) -> None:
        self.W = 320
        self.H = 240
        self.w = 32
        self.h = 32
        self.eta = 0.125
        self.sigma = 2.0
        self.Kernel = np.fft.fft2(GaussKernel(size=(self.h, self.w), sigma=self.sigma))
        self.window = np.hanning(self.h)[:, np.newaxis].dot(
            np.hanning(self.w)[np.newaxis, :]
        )

        self.A = np.zeros((self.h, self.w), dtype=np.complex64)
        self.B = np.zeros((self.h, self.w), dtype=np.float16)

        self.xc = 0.0
        self.yc = 0.0

    def init(self, frame: np.ndarray, xc0: float, yc0: float):
        self.xc = xc0
        self.yc = yc0

        frame = frame / 256 - 0.5
        print(frame.shape)
        f = (
            frame[int(yc0) - 16 : int(yc0) + 16, int(xc0) - 16 : int(xc0) + 16]
            * self.window
        )
        F = np.fft.fft2(f)
        self.A = self.Kernel * np.conj(F)
        self.B = F * np.conj(F)
        # print(f"{np.max(F):.0f} {np.max(self.A):.0f} {np.max(self.B):.0f}")

    def update(self, frame: np.ndarray) -> tuple[float, float]:
        frame = frame / 256 - 0.5

        f = (
            frame[
                int(self.yc) - 16 : int(self.yc) + 16,
                int(self.xc) - 16 : int(self.xc) + 16,
            ]
            * self.window
        )
        F = np.fft.fft2(f)
        H = self.A / self.B
        G = F * H
        g = np.real(np.fft.ifft2(G))
        # print(f"{np.max(F):.0f} {np.max(H):.0f} {np.max(G):.0f} {np.max(g):.2f}")
        position = np.unravel_index(np.argmax(g, axis=None), g.shape)
        dy, dx = position[0] - 15.5, position[1] - 15.5
        self.xc, self.yc = (self.xc + dx, self.yc + dy)

        f = (
            frame[
                int(self.yc) - 16 : int(self.yc) + 16,
                int(self.xc) - 16 : int(self.xc) + 16,
            ]
            * self.window
        )
        F = np.fft.fft2(f)
        self.A = self.eta * self.Kernel * np.conj(F) + (1 - self.eta) * self.A
        self.B = self.eta * F * np.conj(F) + (1 - self.eta) * self.B
        # print(f"{np.max(F):.0f} {np.max(self.A):.0f} {np.max(self.B):.0f}")
        return (self.xc, self.yc)
