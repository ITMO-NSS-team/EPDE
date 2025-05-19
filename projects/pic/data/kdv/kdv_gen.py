import numpy as np
import matplotlib.pyplot as plt


k = 0.7
A = 1
c = 1
ksi0 = 0
alf = 0.5
x = np.linspace(-3, 3, 480)#границы
t = np.linspace(0, 1, 120)#время
T, X = np.meshgrid(t, x, indexing = 'ij')

dx = x[1] - x[0]
dt = t[1] - t[0]

u = (2 * (k ** 2)) / ((np.cosh(- k * (X - 4 * (k ** 2) * T))) ** 2) #уравнение u_t + u_xxx + 6uu_x = 0
u_t = (- 16 * (k ** 5) *np.sinh(- k * (X - 4* (k**2) * T)))/ (np.cosh(- k * (X - 4* (k**2) * T)) ** 3)
u_x = (4 * (k ** 3) *np.sinh(- k * (X - 4* (k**2) * T)))/ (np.cosh(- k * (X - 4* (k**2) * T)) ** 3)
u_xxx = ((16 * (k ** 5) *(((3  * ((np.sinh(- k * (X - 4* (k**2) * T))) ** 2)) / ((np.cosh(- k * (X - 4* (k**2) * T))) ** 2)) - 2)) * np.sinh(- k * (X - 4* (k**2) * T)))/ (np.cosh(- k * (X - 4* (k**2) * T)) ** 3)
uu_x6 = (48 * (k ** 5) *np.sinh(- k * (X - 4* (k**2) * T))) / (np.cosh(- k * (X - 4* (k**2) * T)) ) ** 5

plt.figure(figsize=(5, 4))
plt.imshow(u, aspect='auto', extent=[np.min(x), np.max(x),np.min(t), np.max(t)], cmap='viridis')
plt.colorbar()
plt.title('KdV')
plt.xlabel('x')
plt.ylabel('t')

u_t1 = np.gradient(u, dt, axis=0, edge_order=2) # по t
u_x1 = np.gradient(u, dx, axis=1, edge_order=2) # по x
u_xxx1 = np.gradient(np.gradient(u_x1, dx, axis=1, edge_order=2), dx, axis=1, edge_order=2) # по x
u2=u_t

plt.figure(figsize=(5, 4))
plt.imshow(u_xxx1, aspect='auto', extent=[np.min(x), np.max(x),np.min(t), np.max(t)], cmap='viridis')
plt.colorbar()
plt.title('KdV')
plt.xlabel('x')
plt.ylabel('t')

plt.figure(figsize=(5, 4))
plt.imshow(u_xxx, aspect='auto', extent=[np.min(x), np.max(x),np.min(t), np.max(t)], cmap='viridis')
plt.colorbar()
plt.title('KdV')
plt.xlabel('x')
plt.ylabel('t')

print(np.square(u_xxx1 - u_xxx).mean(axis=None))
print(np.max(u_xxx1 - u_xxx))
print(np.min(u_xxx1 - u_xxx))

np.save('data_kdv_homogen.npy', u)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

X = np.vstack((u_xxx1.flatten(), u.flatten() * u_x1.flatten())).T  #признаки
y = -u_t1.flatten()  #целевая переменная

model.fit(X, y)
coefficients = model.coef_

# Вывод коэффициентов
print(f"Коэффициенты перед членами уравнения:")
print(f"Коэффициент перед u_xxx: {coefficients[0]}")
print(f"Коэффициент перед uu_x: {coefficients[1]}")

plt.show()