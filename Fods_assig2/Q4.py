import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Initial input for prior distribution
a, b = map(int, input("Enter prior distribution ").split())

# Generate X values
x = np.arange(0.01, 1, 0.01)

# Initial Beta distribution
y = beta.pdf(x, a, b)

# First Update (after observing likes)
a1, b1 = map(int, input("Enter likes and total trials: ").split())
a_updated1 = a + a1            # Increase α by number of likes
b_updated1 = b + (b1 - a1)     # Increase β by number of remaining trials (total - likes)
y1 = beta.pdf(x, a_updated1, b_updated1)

# Second Update (after observing dislikes)
a2, b2 = map(int, input("Enter dislikes and total trials: ").split())
a_updated2 = a_updated1 + (b2 - a2)   # Update α with (total - dislikes)
b_updated2 = b_updated1 + a2          # Update β with dislikes
y2 = beta.pdf(x, a_updated2, b_updated2)

# Plot all three distributions on one graph with different colors
plt.figure(figsize=(10, 6))  # Set figure size

plt.plot(x, y, label='Prior Distribution', color='blue')          # Original
plt.plot(x, y1, label='After 1st Update (Likes)', color='green')  # First update
plt.plot(x, y2, label='After 2nd Update (Dislikes)', color='red') # Second update

# Add title, labels, and legend
plt.title('Beta Distribution Updates')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()  # Display the legend to identify each line

# Show the plot
plt.show()
