import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control as ctrl

# Настройка matplotlib
plt.rcParams['font.size'] = 10
plt.style.use('seaborn-v0_8')

print("\n\n--- ЧАСТЬ 2: СИСТЕМА 2-ГО ПОРЯДКА ---")

# Параметры системы
k = 4       # Коэффициент усиления
T = 2       # Постоянная времени
xi = 0.2    # Коэффициент демпфирования

print(f"Параметры системы:")
print(f"k = {k}")
print(f"T = {T}")
print(f"ξ = {xi}")

# Передаточная функция: k/(T²s² + 2ξTs + 1)
num2 = [k]
den2 = [T**2, 2*T*xi, 1]
sys2 = signal.TransferFunction(num2, den2)
sys2_ctrl = ctrl.tf(num2, den2)

print(f"Передаточная функция: H(s) = {k}/({T**2}s² + {2*T*xi}s + 1)")

# Основные характеристики
plt.figure(figsize=(15, 10))

# Переходная характеристика
t_step2, y_step2 = signal.step(sys2)

plt.subplot(2, 3, 1)
plt.plot(t_step2, y_step2, 'teal', linewidth=2)
plt.axhline(y=k*1.05, color='r', linestyle='--', alpha=0.7, label='±5%')
plt.axhline(y=k*0.95, color='r', linestyle='--', alpha=0.7)
plt.axhline(y=k, color='k', linestyle='-', alpha=0.5, label='Установившееся')
plt.title('Переходная характеристика')
plt.xlabel('Время [с]')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True, alpha=0.3)

# Частотные характеристики
w2, mag2, phase2 = signal.bode(sys2)

plt.subplot(2, 3, 2)
plt.semilogx(w2, mag2, 'b-', linewidth=2)
plt.title('АЧХ')
plt.xlabel('Частота [рад/с]')
plt.ylabel('Амплитуда [дБ]')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.semilogx(w2, phase2, 'r-', linewidth=2) 
plt.title('ФЧХ')
plt.xlabel('Частота [рад/с]')
plt.ylabel('Фаза [град]')
plt.grid(True, alpha=0.3)

# АФХ
w0 = 1/T  # Характерная частота
w_nyq = np.logspace(np.log10(w0)-3, np.log10(w0)+2, 10000)
w_nyq, H = signal.freqresp(sys2, w_nyq)

plt.subplot(2, 3, 4)
plt.plot(H.real, H.imag, 'g-', linewidth=2, label='ω≥0')
plt.plot(H.real, -H.imag, 'g--', linewidth=2, label='ω≤0')
plt.plot(k, 0, 'bo', markersize=6, label='ω=0')
plt.plot(0, -k/(2*xi), 'ro', markersize=6, label=f'ω={w0:.1f} рад/с')
plt.title('АФХ')
plt.xlabel('Действительная часть')
plt.ylabel('Мнимая часть')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.legend()

# 1. Анализ движения корней при изменении ξ
print("\n1. Анализ движения корней при изменении ξ...")
xi_range = np.linspace(0.1, 2.0, 50)
root_real = []
root_imag = []

for xi_val in xi_range:
    den_temp = [T**2, 2*T*xi_val, 1]
    sys_temp = ctrl.tf([k], den_temp)
    poles_temp = sys_temp.poles()
    
    root_real.extend(np.real(poles_temp))
    root_imag.extend(np.imag(poles_temp))

plt.subplot(2, 3, 5)
plt.scatter(root_real, root_imag, c=np.tile(xi_range, 2), cmap='viridis', s=20)
plt.colorbar(label='ξ')
plt.title('Траектория корней при изменении ξ')
plt.xlabel('Действительная часть')
plt.ylabel('Мнимая часть')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# Текущие корни
poles2 = sys2_ctrl.poles()
plt.plot(np.real(poles2), np.imag(poles2), 'rx', markersize=10, markeredgewidth=3, label=f'ξ={xi}')
plt.legend()

plt.tight_layout()
plt.show()

# 2. Зависимость резонансного пика от ξ
print("\n2. Зависимость резонансного пика АЧХ от ξ...")
xi_vals = np.linspace(0.05, 1.0, 100)
resonance_peaks = []

for xi_val in xi_vals:
    if xi_val < 1/np.sqrt(2):  # Условие существования резонанса
        w_res = 1/T * np.sqrt(1 - 2*xi_val**2)
        peak_magnitude = k / (2*xi_val*np.sqrt(1 - xi_val**2))
        resonance_peaks.append(20*np.log10(peak_magnitude))
    else:
        resonance_peaks.append(20*np.log10(k))

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(xi_vals, resonance_peaks, 'b-', linewidth=2)
plt.axvline(x=1/np.sqrt(2), color='r', linestyle='--', alpha=0.7, label='ξ = 1/√2')
plt.title('Зависимость резонансного пика от ξ')
plt.xlabel('Коэффициент демпфирования ξ')
plt.ylabel('Резонансный пик [дБ]')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Зависимость резонансной частоты от T
print("\n3. Зависимость резонансной частоты от T...")
T_vals = np.linspace(0.5, 5.0, 100)
resonance_freqs = []

for T_val in T_vals:
    if xi < 1/np.sqrt(2):
        w_res = 1/T_val * np.sqrt(1 - 2*xi**2)
        resonance_freqs.append(w_res)
    else:
        resonance_freqs.append(0)

plt.subplot(2, 2, 2)
plt.plot(T_vals, resonance_freqs, 'g-', linewidth=2)
plt.title(f'Зависимость резонансной частоты от T (ξ={xi})')
plt.xlabel('Постоянная времени T [с]')
plt.ylabel('Резонансная частота [рад/с]')
plt.grid(True, alpha=0.3)

# 4. Определение оптимального ξ
print("\n4. Определение оптимального ξ...")

def settling_time(xi_val):
    """Улучшенный расчет времени установления"""
    den_temp = [T**2, 2*T*xi_val, 1]
    sys_temp = signal.TransferFunction([k], den_temp)
    t_temp, y_temp = signal.step(sys_temp)
    
    # Границы ±5%
    low_bound = 0.95 * k
    high_bound = 1.05 * k
    
    # Ищем последний выход за границы
    last_crossing_idx = -1
    for i in range(len(y_temp)-1, -1, -1):
        if y_temp[i] < low_bound or y_temp[i] > high_bound:
            last_crossing_idx = i
            break
    
    return t_temp[last_crossing_idx] if last_crossing_idx != -1 else 0.0

# Поиск оптимального ξ
xi_test_range = np.linspace(0.1, 2.0, 200)
settling_times = [settling_time(xi_val) for xi_val in xi_test_range]

valid_times = list(zip(xi_test_range, settling_times))
xi_opt, t_opt = min(valid_times, key=lambda x: x[1])
print(f"Оптимальное ξ = {xi_opt:.3f}")
print(f"Время установления = {t_opt:.3f} с")

# Корни при оптимальном ξ
den_opt = [T**2, 2*T*xi_opt, 1]
sys_opt = ctrl.tf([k], den_opt)
poles_opt = sys_opt.poles()
print(f"Корни при ξ_опт: {poles_opt}")

# Высота пика ЛАЧХ
w_opt, mag_opt, _ = signal.bode(signal.TransferFunction([k], den_opt))
peak_height = np.max(mag_opt)
print(f"Высота пика ЛАЧХ: {peak_height:.2f} дБ")

plt.subplot(2, 2, 3)
valid_xi = [xi for xi, _ in valid_times]
valid_st = [st for _, st in valid_times]
plt.plot(valid_xi, valid_st, 'purple', linewidth=2)
plt.axvline(x=xi_opt, color='r', linestyle='--', alpha=0.7, label=f'ξ_опт = {xi_opt:.3f}')
plt.plot(xi_opt, t_opt, 'ro', markersize=8)
plt.title('Время установления vs ξ')
plt.xlabel('Коэффициент демпфирования ξ')
plt.ylabel('Время установления [с]')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Характеристики при отрицательном ξ
print("\n5. Характеристики при ξ = -ξ...")
xi_neg = -xi
den_neg = [T**2, 2*T*xi_neg, 1]
sys_neg = signal.TransferFunction([k], den_neg)

t_step_neg, y_step_neg = signal.step(sys_neg)

plt.subplot(2, 2, 4)
plt.plot(t_step_neg, y_step_neg, 'orange', linewidth=2, label=f'ξ = {xi_neg}')
plt.plot(t_step2, y_step2, 'teal', linewidth=2, alpha=0.7, label=f'ξ = {xi}')
plt.title('Сравнение переходных характеристик')
plt.xlabel('Время [с]')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Сравнение частотных характеристик
w_neg, mag_neg, phase_neg = signal.bode(sys_neg)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.semilogx(w2, mag2, 'b-', linewidth=2, label=f'ξ = {xi}')
plt.semilogx(w_neg, mag_neg, 'r--', linewidth=2, label=f'ξ = {xi_neg}')
plt.title('Сравнение АЧХ')
plt.xlabel('Частота [рад/с]')
plt.ylabel('Амплитуда [дБ]')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogx(w2, phase2, 'b-', linewidth=2, label=f'ξ = {xi}')
plt.semilogx(w_neg, phase_neg, 'r-', linewidth=2, label=f'ξ = {xi_neg}')
plt.title('Сравнение ФЧХ')
plt.xlabel('Частота [рад/с]')
plt.ylabel('Фаза [град]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Сравнение АФХ для отрицательного ξ
w_nyq_neg, H_neg = signal.freqresp(sys_neg, w_nyq)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(H.real, H.imag, 'b-', label=f'ξ={xi} (ω≥0)')
plt.plot(H.real, -H.imag, 'b--', label=f'ξ={xi} (ω≤0)')
# plt.plot(H_neg.real, H_neg.imag, 'r-', label=f'ξ={xi_neg} (ω≥0)')
# plt.plot(H_neg.real, -H_neg.imag, 'r--', label=f'ξ={xi_neg} (ω≤0)')
plt.title('Сравнение АФХ')
plt.xlabel('Действительная часть')
plt.ylabel('Мнимая часть')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(H_neg.real, H_neg.imag, 'r-', label=f'ξ={xi_neg} (ω≥0)')
plt.plot(H_neg.real, -H_neg.imag, 'r--', label=f'ξ={xi_neg} (ω≤0)')
plt.title(f'АФХ при ξ={xi_neg}')
plt.xlabel('Действительная часть')
plt.ylabel('Мнимая часть')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("АНАЛИЗ ЗАВЕРШЕН")
print("="*60)