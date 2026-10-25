import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    df = pd.read_csv('ocr_benchmark_results.csv')
except FileNotFoundError:
    print("Файл ocr_benchmark_results.csv не найден")
    exit()

summary = df.groupby('Model')[['CER (%)', 'Time (s)']].mean().reset_index()

fig, ax1 = plt.subplots(figsize=(10, 6))

bars = ax1.bar(summary['Model'], summary['CER (%)'], color='#4C72B0', alpha=0.85, width=0.5, label='Ошибки (CER, %)')
ax1.set_ylabel('Средний процент ошибок (CER, %)', color='#4C72B0', fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(summary['CER (%)']) * 1.25)
ax1.tick_params(axis='y', labelcolor='#4C72B0')

ax2 = ax1.twinx()
line = ax2.plot(summary['Model'], summary['Time (s)'], color='#C44E52', marker='o', linewidth=2.5, markersize=8, label='Время (сек)')
ax2.set_ylabel('Среднее время обработки (с)', color='#C44E52', fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(summary['Time (s)']) * 1.25)
ax2.tick_params(axis='y', labelcolor='#C44E52')

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{round(yval, 1)}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Сравнение моделей OCR: Ошибки vs Скорость', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig('./graphs/ocr_overall_comparison.png', dpi=300)
print("График 'ocr_overall_comparison.png' сохранен.")
plt.close()

plt.figure(figsize=(12, 6))
defects = df['Defect'].unique()
models = df['Model'].unique()

x = np.arange(len(models))
width = 0.25
colors = ['#55A868', '#F1A340', '#9970AB']

for i, defect in enumerate(defects):
    defect_data = df[df['Defect'] == defect].groupby('Model')['CER (%)'].mean()
    y = [defect_data.get(m, 0) for m in models]
    plt.bar(x + i*width - width, y, width, label=defect, color=colors[i % len(colors)], edgecolor='black')

plt.ylabel('Процент ошибок (CER, %)', fontsize=12, fontweight='bold')
plt.title('Устойчивость алгоритмов OCR к заводским дефектам', fontsize=14, fontweight='bold')
plt.xticks(x, models, fontsize=11)
plt.legend(title='Тип изображения')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('./graphs/ocr_defect_robustness.png', dpi=300)
print("График 'ocr_defect_robustness.png' сохранен.")
plt.close()
