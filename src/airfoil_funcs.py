from svgpathtools import svg2paths
import matplotlib.pyplot as plt
import numpy as np


def read_airfoil(svg_path):
    paths, _ = svg2paths(svg_path)

    coordinates = []
    for segment in paths[0]:
        for point in segment:
            coordinates.append([point.real, -point.imag])
    
    return np.transpose(np.array(coordinates))

def scale_airfoil(x, y):
    x_range = np.ptp(x)
    y_camber = y[np.argmin(x)]

    x_scale = (x - x.min()) / x_range
    y_scale = (y - y_camber) / x_range

    return x_scale, y_scale

def interp_paths(x1, y1, x2, y2, ratio):
    return (
        x1 * (1 - ratio) + x2 * ratio,
        y1 * (1 - ratio) + y2 * ratio
    )


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    x1, y1 = scale_airfoil(*read_airfoil("../resources/0012.svg"))
    x2, y2 = scale_airfoil(*read_airfoil("../resources/0012-up.svg"))
    x, y = interp_paths(x1, y1, x2, y2, 0.2)

    # print(x, y)

    plt.figure(figsize=(6, 6))
    plt.plot(x1, y1, '-o', markersize=2)
    plt.plot(x2, y2, '-o', markersize=2)
    plt.plot(x, y, '-o', markersize=2)
    plt.axis("equal")
    plt.title("SVG Path Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig("asdf.png", dpi=300)
