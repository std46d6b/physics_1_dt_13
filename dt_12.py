import logging

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.ERROR, format=log_format)

log = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.INFO)


class PointCharge:
    def __init__(self, q, x, y):
        self.q = q
        self.x = x
        self.y = y


def electric_field(charges, x, y):
    """
    Вычисление компонентов электростатического поля в точке (x, y).
    """
    Ex, Ey = 0, 0
    k = 8.99e9

    for charge in charges:
        dx = x - charge.x
        dy = y - charge.y
        r_squared = dx ** 2 + dy ** 2
        if r_squared < 1e-6:
            continue

        r = np.sqrt(r_squared)
        E = k * charge.q / r_squared
        Ex += E * (dx / r)
        Ey += E * (dy / r)

    return Ex, Ey


def potential(charges, x, y):
    """
    Вычисление потенциала в точке (x, y).
    """
    V = 0
    k = 8.99e9

    for charge in charges:
        dx = x - charge.x
        dy = y - charge.y
        r_squared = dx ** 2 + dy ** 2
        if r_squared < 1e-6:
            continue

        r = np.sqrt(r_squared)
        V += k * charge.q / r

    return V


print("Введите параметры для моделирования электростатического поля")

log.debug("Ожидание ввода параметров")
n = int(input("Количество зарядов (пример: 3): ") or 3)

qs, xs, ys = [], [], []

for i in range(n):
    qs.append(float(
        input(
            f"Значение заряда     {i + 1:<3} (пример: {'-' * ((i + 1) % 2):1}0.02): "
        ) or (0.02 * (1 if (i % 2) else -1))))
    xr = (20 + 10 * max(i - 0, 0)) * (-1 if (i % 2) else 1)
    xs.append(float(
        input(
            f"X-координату заряда {i + 1:<3} (пример: {xr:5}): "
        ) or xr))
    yr = (10 + 10 * max(i - 0, 0)) * (-1 if (i % 2) else 1)
    ys.append(float(
        input(
            f"Y-координату заряда {i + 1:<3} (пример: {yr:5}): "
        ) or yr))

log.debug("Параметры введены")

for i in range(n):
    log.debug(f"Заряд {i + 1}: {qs[i]} Кл, X-координата: {xs[i]}, Y-координата: {ys[i]}")

log.info("Начало вычислений")

charges = []
for i in range(n):
    charges.append(PointCharge(qs[i], xs[i], ys[i]))

linspace_r = int(np.sqrt(abs(max(xs) - min(xs)) ** 2 + abs(max(ys) - min(ys)) ** 2) * 1.1 + 1)
grids = 500
x_mid = (max(xs) + min(xs)) / 2
y_mid = (max(ys) + min(ys)) / 2

log.info("Генерация сетки координат")
x = np.linspace(-linspace_r + x_mid, linspace_r + x_mid, grids)
y = np.linspace(-linspace_r + y_mid, linspace_r + y_mid, grids)
X, Y = np.meshgrid(x, y)

Ex, Ey = np.zeros(X.shape), np.zeros(Y.shape)
V = np.zeros(X.shape)

log.info("Вычисление компонентов поля и потенциала")
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Ex[i, j], Ey[i, j] = electric_field(charges, X[i, j], Y[i, j])
        V[i, j] = potential(charges, X[i, j], Y[i, j])

log.info("Построение графика")
plt.figure(figsize=(10, 10))
colors = np.sqrt(Ex ** 2 + Ey ** 2)
plt.streamplot(X, Y, Ex, Ey, color=colors, linewidth=1, density=2.5, cmap=cm.rainbow, arrowstyle='->', arrowsize=1.5)

log.info("Построение контурных линий потенциала")
vv = np.sort(np.unique(V))
log.debug(f"Число уникальных значений потенциала: {len(vv)} / {len(V)}")
vn, vp = [0] + vv[vv < 0].tolist(), [0] + vv[vv > 0].tolist()

log.debug(f"Интервалы потенциала: {vn[0]}, {vp[-1]}")

amp = 0.1
contour_levels = np.linspace(vn[int(len(vn) / 100 * amp)], vp[int(len(vp) / 100 * (100 - amp))], 250)

plt.contour(X, Y, V, levels=contour_levels, cmap=cm.viridis, alpha=0.75)

for charge in charges:
    plt.plot(charge.x, charge.y, 'ro' if charge.q > 0 else 'bo', markersize=15)

log.info("Настройка графика")
plt.xlabel('X')
plt.ylabel('Y', rotation=0)
plt.title('Электростатическое поле и эквипотенциальные поверхности')
plt.grid()
plt.axis('equal')
log.info("Вывод графика")
plt.show()
