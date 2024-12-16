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

    def __str__(self):
        return f"PointCharge(q={self.q}, x={self.x}, y={self.y})"


class Dipole:
    def __init__(self, p, x, y, theta):
        self.p = p  # Модуль дипольного момента
        self.x = x
        self.y = y
        self.theta = theta  # Угол направления дипольного момента (в радианах)

    def as_charges(self):
        """
        Представляет диполь как два точечных заряда.
        """
        d_half = self.p / abs(self.p) * 1e+1  # Маленькое расстояние между зарядами
        q_magnitude = self.p / (2 * d_half)
        dx = np.cos(self.theta) * d_half
        dy = np.sin(self.theta) * d_half

        positive_charge = PointCharge(q_magnitude, self.x + dx * 2, self.y + dy * 2)
        negative_charge = PointCharge(-q_magnitude, self.x, self.y)

        return [positive_charge, negative_charge]


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


def dipole_force_and_torque(dipole, charges):
    """
    Вычисление силы и момента силы, действующих на диполь.
    """
    Fx, Fy, torque = 0, 0, 0

    for charge in charges:
        dx = dipole.x - charge.x
        dy = dipole.y - charge.y
        r_squared = dx ** 2 + dy ** 2
        if r_squared < 1e-6:
            continue

        r = np.sqrt(r_squared)
        k = 8.99e9
        E = k * charge.q / r_squared

        Ex = E * (dx / r)
        Ey = E * (dy / r)

        Fx += dipole.p * np.cos(dipole.theta) * Ex
        Fy += dipole.p * np.sin(dipole.theta) * Ey

        torque += dipole.p * (np.sin(dipole.theta) * Ex - np.cos(dipole.theta) * Ey)

    return Fx, Fy, torque


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

charges = [PointCharge(qs[i], xs[i], ys[i]) for i in range(n)]

print("Введите параметры диполя")
p = float(input("Модуль дипольного момента (пример: 1): ") or 1)
x_d = float(input("X-координата диполя (пример: 0): ") or 0)
y_d = float(input("Y-координата диполя (пример: 0): ") or 0)
theta = float(input("Угол диполя в градусах (пример: 45): ") or 45) * np.pi / 180
log.debug("Окончание ввода параметров")
print()

dipole = Dipole(p, x_d, y_d, theta)
dipole_charges = dipole.as_charges()
charges.extend(dipole_charges)

Fx, Fy, torque = dipole_force_and_torque(dipole, charges)

print(f"Сила, действующая на диполь: Fx = {Fx:.2e} Н, Fy = {Fy:.2e} Н")
print(f"Момент силы, действующий на диполь: {torque:.2e} Н·м")

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
plt.contour(X, Y, V, levels=50, cmap=cm.viridis, alpha=0.75)

for charge in charges:
    plt.plot(charge.x, charge.y, 'ro' if charge.q > 0 else 'bo', markersize=15)

plt.plot(dipole.x, dipole.y, 'ks', markersize=10, label='Dipole')

plt.quiver(dipole.x, dipole.y, np.cos(dipole.theta), np.sin(dipole.theta), color='k', scale=10, label='Moment')

log.info("Настройка графика")
plt.xlabel('X')
plt.ylabel('Y', rotation=0)
plt.title('Электростатическое поле и эквипотенциальные поверхности с диполем')
plt.legend()
plt.grid()
plt.axis('equal')
log.info("Вывод графика")
plt.show()
