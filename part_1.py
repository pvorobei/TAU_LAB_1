import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control as ctrl

# Настройка matplotlib
plt.rcParams['font.size'] = 10
plt.style.use('seaborn-v0_8')

print("="*60)
print("ИССЛЕДОВАНИЕ КОЛЕБАТЕЛЬНОГО КОНТУРА. ВАРИАНТ 2.")
print("="*60)

# ========================================
# ЧАСТЬ 1: RC-ЦЕПЬ
# ========================================
print("\n--- ЧАСТЬ 1: RC-ЦЕПЬ ---")

# 1. Параметры RC-цепи
R = 15           # Сопротивление [Ом]
C = 0.015e-6     # Ёмкость [Ф] (0.015 мкФ)
U = 50           # Напряжение [В]

print(f"Параметры RC-цепи:")
print(f"R = {R} Ом")
print(f"C = {C*1e6} мкФ")
print(f"U = {U} В")
print(f"Постоянная времени τ = RC = {R*C:.2e} с")

# 2. Передаточная функция RC-цепи
# Для RC-цепи: H(s) = 1/(RCs + 1)
num1 = [1]
den1 = [R*C, 1]
sys1 = signal.TransferFunction(num1, den1)
sys1_ctrl = ctrl.tf(num1, den1)

print(f"\nПередаточная функция: H(s) = 1/({R*C:.2e}s + 1)")

# 3.1. Ступенчатый импульс
print("\n3.1. Моделирование ступенчатого сигнала...")
t1 = np.linspace(0, 5*R*C, 1000)
u_step = U * np.ones_like(t1)
t_out1, y_out1, _ = signal.lsim(sys1, u_step, t1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t1*1e6, u_step, 'b--', linewidth=2, label='Входной сигнал')
plt.plot(t_out1*1e6, y_out1, 'r-', linewidth=2, label='Выходной сигнал')
plt.title('Отклик на ступенчатое воздействие')
plt.xlabel('Время [мкс]')
plt.ylabel('Напряжение [В]')
plt.legend()
plt.grid(True, alpha=0.3)

# 3.2. Прямоугольный импульс (меандр)
print("3.2. Моделирование прямоугольного импульса...")
t2 = np.linspace(0, 30*R*C, 3000)
period = 12*R*C
duty_cycle = 0.5

# Создание меандра
u_pulse = np.zeros_like(t2)
for i, time in enumerate(t2):
    cycle_pos = (time % period) / period
    if cycle_pos < duty_cycle:
        u_pulse[i] = U

t_out2, y_out2, _ = signal.lsim(sys1, u_pulse, t2)

plt.subplot(1, 2, 2)
plt.plot(t2*1e6, u_pulse, 'g-', linewidth=2, label='Прямоугольный импульс')
plt.plot(t_out2*1e6, y_out2, 'm-', linewidth=2, label='Отклик RC-цепи')
plt.title('Реакция на прямоугольный импульс')
plt.xlabel('Время [мкс]')
plt.ylabel('Напряжение [В]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Сравнительный анализ
print("\n4. Сравнительный анализ:")
print(f"   - Постоянная времени τ = {R*C*1e6:.2f} мкс")
print(f"   - Время нарастания (10%-90%): ≈ {2.2*R*C*1e6:.2f} мкс")
print(f"   - Время установления (±5%): ≈ {3*R*C*1e6:.2f} мкс")

# 5. Характеристики системы
print("\n5. Построение характеристик системы...")

# 5.1. Переходная характеристика
t_step, y_step = signal.step(sys1)

plt.figure(figsize=(15, 10))

# Переходная характеристика
plt.subplot(2, 3, 1)
plt.plot(t_step*1e6, y_step, 'purple', linewidth=2)
plt.title('Переходная характеристика')
plt.xlabel('Время [мкс]')
plt.ylabel('Амплитуда')
plt.grid(True, alpha=0.3)

# 5.2. Импульсная характеристика
t_impulse, y_impulse = signal.impulse(sys1)

plt.subplot(2, 3, 2)
plt.plot(t_impulse*1e6, y_impulse, 'orange', linewidth=2)
plt.title('Импульсная характеристика')
plt.xlabel('Время [мкс]')
plt.ylabel('Амплитуда')
plt.grid(True, alpha=0.3)

# 5.3. АЧХ и ФЧХ
w, mag, phase = signal.bode(sys1)

plt.subplot(2, 3, 3)
plt.semilogx(w, mag, 'b-', linewidth=2)
plt.title('АЧХ')
plt.xlabel('Частота [рад/с]')
plt.ylabel('Амплитуда [дБ]')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
plt.semilogx(w, phase*180/np.pi, 'r-', linewidth=2)
plt.title('ФЧХ')
plt.xlabel('Частота [рад/с]')
plt.ylabel('Фаза [град]')
plt.grid(True, alpha=0.3)

# 5.4. АФХ
w0 = 1/(R*C)  # Характерная частота системы
w_nyq = np.logspace(np.log10(w0)-4, np.log10(w0)+2, 10000)
w_nyq, H = signal.freqresp(sys1, w_nyq)  # Получение АФХ

plt.subplot(2, 3, 5)
plt.plot(H.real, H.imag, 'g-', linewidth=2, label='ω≥0')
plt.plot(H.real, -H.imag, 'g--', linewidth=2, label='ω≤0')
# Отметим характерные точки
plt.plot(1, 0, 'bo', markersize=6, label='ω=0')  # Начало (ω=0)
plt.plot(0.5, -0.5, 'ro', markersize=6, label=f'ω={w0:.1e} rad/s')  # Характерная частота
plt.title('АФХ')
plt.xlabel('Действительная часть')
plt.ylabel('Мнимая часть')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.legend()

# 5.5. Корни на комплексной плоскости
poles = sys1_ctrl.poles()
plt.subplot(2, 3, 6)
plt.plot(np.real(poles), np.imag(poles), 'rx', markersize=10, markeredgewidth=3)
plt.title('Расположение корней')
plt.xlabel('Действительная часть')
plt.ylabel('Мнимая часть')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

print(f"\nПолюс системы: s = {poles[0]:.2e}")

print("\n" + "="*60)
print("АНАЛИЗ ЗАВЕРШЕН")
print("="*60)