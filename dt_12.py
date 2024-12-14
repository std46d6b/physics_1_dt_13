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
q1 = float(input("Значение заряда 1 (пример:  0.02 Кл): ") or 0.02)
x1 = float(input("X-координату заряда 1   (пример: -2): ") or -20)
y1 = float(input("Y-координату заряда 1   (пример: -1): ") or -10)

q2 = float(input("Значение заряда 2 (пример: -0.01 Кл): ") or -0.01)
x2 = float(input("X-координату заряда 2    (пример: 2): ") or 20)
y2 = float(input("Y-координату заряда 2    (пример: 1): ") or 10)
log.debug("Параметры введены")

log.debug(f"Заряд 1: {q1} Кл, X-координата: {x1}, Y-координата: {y1}")
log.debug(f"Заряд 2: {q2} Кл, X-координата: {x2}, Y-координата: {y2}")

log.info("Начало вычислений")

charges = [
    PointCharge(q1, x1, y1),
    PointCharge(q2, x2, y2)
]

linspace_r = int(np.sqrt(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2 + q2 ** 2) * 1.1 + 1)
grids = 500
x_mid = (x2 + x1) / 2
y_mid = (y2 + y1) / 2

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
vn, vp = vv[vv < 0], vv[vv > 0]

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
