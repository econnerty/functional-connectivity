import matplotlib.pyplot as plt
import numpy as np

# Correcting the calculation for Bézier curves

fig, ax = plt.subplots(figsize=(10, 6))

# Input signal
ax.annotate('Input Signal', xy=(0.1, 0.5), xytext=(0.1, 0.7),
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='center', verticalalignment='top')

# Depicting the reservoir network as a complex, nonlinear "black box"
reservoir_network = plt.Circle((0.4, 0.5), 0.15, color='gray', fill=True)
ax.add_patch(reservoir_network)
ax.text(0.4, 0.5, 'Reservoir Network\n(Transfer Function)', ha='center', va='center', color='white')

# Adding more pronounced curved arrows for feedback
for i, angle in enumerate(np.linspace(0, 2*np.pi, 8, endpoint=False)):
    # Starting and ending points for arrows
    start_x = 0.4 + 0.15*np.cos(angle)
    start_y = 0.5 + 0.15*np.sin(angle)
    control_x = 0.4 + 0.3*np.cos(angle + np.pi/16)
    control_y = 0.5 + 0.3*np.sin(angle + np.pi/16)
    end_x = 0.4 + 0.15*np.cos(angle + np.pi/8)
    end_y = 0.5 + 0.15*np.sin(angle + np.pi/8)
    
    # Generating Bézier curves
    bezier_x = np.array([start_x, control_x, end_x])
    bezier_y = np.array([start_y, control_y, end_y])
    t = np.linspace(0, 1, 100)
    bezier_curve_x = (1-t)**2 * bezier_x[0] + 2*(1-t)*t*bezier_x[1] + t**2*bezier_x[2]
    bezier_curve_y = (1-t)**2 * bezier_y[0] + 2*(1-t)*t*bezier_y[1] + t**2*bezier_y[2]

    # Plot curve
    ax.plot(bezier_curve_x, bezier_curve_y, color="black")
    # Add arrowhead manually at the end of the curve
    ax.annotate('', xy=(end_x, end_y), xytext=(bezier_curve_x[-2], bezier_curve_y[-2]),
                arrowprops=dict(arrowstyle="->", color="black"))

# Readout layer
ax.annotate('Readout Layer', xy=(0.7, 0.5), xytext=(0.7, 0.7),
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='center', verticalalignment='top')

# Predicted signal
ax.annotate('Target Signal', xy=(0.9, 0.5), xytext=(0.9, 0.3),
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='center', verticalalignment='bottom')

# Arrows indicating flow
ax.arrow(0.1, 0.5, 0.1, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
ax.arrow(0.55, 0.5, 0.1, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.show()
