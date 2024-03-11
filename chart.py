import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['Condition Number', 'MSE', 'SNR']
values_4_terms = [7.1445996e+07, 34.42989453651755, -5.928310053508577]
values_25_terms = [3.481935164122768e+19, 31.11613023219873, -4.599106710485936]

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, values_4_terms, width, label='4 Terms')
rects2 = ax.bar(x + width/2, values_25_terms, width, label='25 Terms')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Values')
ax.set_title('Comparison of EEG Basis Function with 4 and 25 Terms')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Function to auto-label the bars with their values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# Show plot
plt.tight_layout()
plt.show()
