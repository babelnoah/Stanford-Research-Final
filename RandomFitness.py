import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image

convergent_points = []
points = []
points_convergences = {}

# Define coordinates of the tetrahedron vertices
a_vertex = np.array([np.sqrt(8/9), 0, -1/3])
b_vertex = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3])
c_vertex = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3])
d_vertex = np.array([0,0,1])

vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])

def create_matrix():
    W = np.zeros((4, 4))

    W[0][0] = np.random.rand() #w11
    W[1][1] = np.random.rand() #w22
    W[0][1] = W[1][0] = np.random.rand() # w12 = w21
    W[0][2] = W[2][0] = np.random.rand() # w13 = w31
    W[1][2] = W[2][1] = W[3][0] = W[0][3] = np.random.rand() # w23 = w32 = w41 = w14
    W[1][3] = W[3][1] = np.random.rand() # w24 = w42
    W[2][3] = W[3][2] = np.random.rand() # w34 = w43
    W[2][2] = np.random.rand() #w33
    W[3][3] = np.random.rand() #w44
    return W

def calculate_next_generation(x, r, W):

    # Compute W_i (average fitness values)
    W_avg = np.zeros(4)  # Initialize an array with zeros
    for i in range(4):
        for j in range(4):
            W_avg[i] += W[i][j] * x[j]
    
    # w_bar calculation
    w_bar = sum([W_avg[i] * x[i] for i in range(4)])

    # D (Differential of x)
    D = x[0]*x[3] - x[1]*x[2]

    # Next Generation calculations
    x_next = np.zeros(4)
    x_next[0] = (x[0] * W_avg[0] - W[0][3] * r * D) / w_bar
    x_next[1] = (x[1] * W_avg[1] + W[0][3] * r * D) / w_bar
    x_next[2] = (x[2] * W_avg[2] + W[0][3] * r * D) / w_bar
    x_next[3] = (x[3] * W_avg[3] - W[0][3] * r * D) / w_bar

    return x_next

def iterate_generations(x, matrix):
    stable_generations = 0  # Initialize counter for stable generations
    change_threshold = 10**-10  # Set change threshold
    num_generations = 0

    while True:
        num_generations +=1
        # Save current x for later comparison
        x_old = x.copy()
        r = 0.5 #Recombination Fraction
        #r = np.random.uniform(0,0.5)
        x = calculate_next_generation(x, r, matrix)

        # Check if absolute change in all components of x is less than threshold
        if np.all(np.abs(x - x_old) < change_threshold):
            stable_generations += 1
        else:
            # Reset counter if change is above threshold
            stable_generations = 0

        points.append(x.tolist())

        #Break loop if x has been stable for 100 generations
        if stable_generations >= 100:
            print(num_generations)
            break
        # if num_generations == 10**4:
        #     print(num_generations)
        #     break
    
total_iterations = 10000  # Number of tests
for i in range(total_iterations):
    matrix = create_matrix()
    initial_point = np.random.dirichlet(np.ones(4), size=1)[0]
    iterate_generations(initial_point, matrix) 

    # Store the initial point and its corresponding convergent point
    points_convergences[tuple(initial_point)] = points[-1]

    # Total progress calculation
    progress = (i + 1) / total_iterations * 100
    print(f"Total progress: {progress:.2f}%")

# Convert dict to two separate lists
initial_points = list(points_convergences.keys())
convergent_points = list(points_convergences.values())

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Extract convergent points for plotting
convergent_points = np.array(convergent_points)

# Convert to Cartesian coordinates
cartesian_convergent_points = []
for i, element in enumerate(convergent_points):
    cartesian_convergent_points.append(np.dot(vertices.T, element))
cartesian_convergent_points = np.array(cartesian_convergent_points)

# Make sure it's an array
cartesian_convergent_points = np.array(cartesian_convergent_points)

x_conv = cartesian_convergent_points[:, 0]
y_conv = cartesian_convergent_points[:, 1]
z_conv = cartesian_convergent_points[:, 2]

xyz = np.vstack([x_conv, y_conv, z_conv])
print("Density")
density = gaussian_kde(xyz)(xyz)

# Normalize the density values to use them as colors for the heatmap.
density_norm = density / max(density)
print("Min:", np.min(density_norm))
print("Max:", np.max(density_norm))
print("Mean:", np.mean(density_norm))
print("Std Dev:", np.std(density_norm))


# Plot the points using the normalized density values.
scatter = ax.scatter(x_conv, y_conv, z_conv, c=density_norm, cmap='viridis', s=30,edgecolor='k', linewidth=0.02)

# Add colorbar to indicate the density
# cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
# cbar.set_label('Density')

#Plot tetrahedral
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
ax.plot_trisurf(*vertices.T, color='r', alpha=0.1)
edges = [
    (a_vertex, b_vertex),
    (a_vertex, c_vertex),
    (a_vertex, d_vertex),
    (b_vertex, c_vertex),
    (b_vertex, d_vertex),
    (c_vertex, d_vertex)
]
for start, end in edges:
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='black')
barycentric_coords = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
labels = ['AB', 'Ab', 'aB', 'ab']
barycentric_to_cartesian = []
for i, element in enumerate(barycentric_coords):
    barycentric_to_cartesian.append(np.dot(vertices.T, element))


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
font_properties = {'weight': 'normal', 'size': 12}

offsets = {
    "AB": (0.03, 0, 0),   # Slightly to the right
    "Ab": (-0.1, 0, 0), # Slightly to the left
    "aB": (-0.07, -0.14, 0), # Slightly down
    "ab": (0, 0.1, 0)   # Slightly up
}

for i in range(len(barycentric_to_cartesian)):
    x, y, z = barycentric_to_cartesian[i]
    offset = offsets[labels[i]]
    ax.scatter(x, y, z, color='black')
    ax.text(x + offset[0], y + offset[1], z + offset[2], labels[i], color='black', **font_properties)
#convergent_scatter = ax.scatter(x_conv, y_conv, z_conv, c='green', s=30, label='Convergent Points')
ax.view_init(elev=25, azim=-105)
ax.w_xaxis.pane.fill = ax.w_yaxis.pane.fill = ax.w_zaxis.pane.fill = False

# Remove tick labels to declutter:
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.grid(color="white", linestyle='solid')
plt.tight_layout()
# Save the image to your desktop with high DPI for better quality
filename = "placeholder.png"
plt.show()
plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)

# Close the plt figure to release memory
plt.close()

# Crop the image to remove any whitespace (using PIL)
img = Image.open(filename)
cropped_img = img.crop(img.getbbox()) 
cropped_img.save(filename)