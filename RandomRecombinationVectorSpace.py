import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


convergent_points = []

# Define coordinates of the tetrahedron vertices
a_vertex = np.array([np.sqrt(8/9), 0, -1/3])
b_vertex = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3])
c_vertex = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3])
d_vertex = np.array([0,0,1])

vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])

def calculate_next_generation(x, r, delta, alpha, beta, gamma):
    D = x[0]*x[3] - x[1]*x[2] 
    w_bar = 1 - delta*(x[0]**2 + x[3]**2) - alpha*(x[1]**2 + x[2]**2) - 2*beta*(x[2]*x[3] + x[0]*x[1]) - 2*gamma*(x[0]*x[2] + x[1]*x[3])
    x_next = np.zeros(4)
    x_next[0] = (x[0] - delta*x[0]**2 - beta*x[0]*x[1] - gamma*x[0]*x[2] - r*D)/w_bar
    x_next[1] = (x[1] - beta*x[0]*x[1] - alpha*x[1]**2 - gamma*x[1]*x[3] + r*D)/w_bar
    x_next[2] = (x[2] - gamma*x[0]*x[2] - alpha*x[2]**2 - beta*x[2]*x[3] + r*D)/w_bar
    x_next[3] = (x[3] - gamma*x[1]*x[3] - beta*x[2]*x[3] - delta*x[3]**2 - r*D)/ w_bar

    # Normalize x_next so that it sums to 1
    x_next /= np.sum(x_next)

    return x_next

def iterate_generations(x, delta, alpha, beta, gamma):
    stable_generations = 0  # Initialize counter for stable generations
    change_threshold = 10**-7  # Set change threshold (alternative option)
    total_generations =0

    while True:
        total_generations+=1
        # Save current x for later comparison
        x_old = x.copy()
        
        #Recombination Fraction
        r = 0.02
        #r = np.random.uniform(0,0.04)

        x = calculate_next_generation(x, r, delta, alpha, beta, gamma)

        # Check if absolute change in all components of x is less than threshold
        if np.all(np.abs(x - x_old) < change_threshold):
            stable_generations += 1
        else:
            # Reset counter if change is above threshold
            stable_generations = 0

        points.append(x.tolist())

        # Break loop if x has been stable for 100 generations
        # if stable_generations >= 100:
        #     break
        if total_generations >= 4000:
            break

#Alpha-Gamma translates Markov update equations as shown in Karlin and Felman "Linkage and selection: two locus symmetric viability model"
a = 0.03
b = 0.004
d = 0.005
g = b

points = []

points_convergences = {}
total_iterations = 1000  # Number of tests
for i in range(total_iterations):
    initial_point = np.random.dirichlet(np.ones(4), size=1)[0]
    iterate_generations(initial_point, d, a, b, g) 

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

# Convert to Cartesian coordinates
cartesian_initial_points = np.array([np.dot(vertices.T, point) for point in initial_points])
cartesian_convergent_points = np.array([np.dot(vertices.T, point) for point in convergent_points])

# Calculate displacement vectors
displacement_vectors = cartesian_convergent_points - cartesian_initial_points

# Vector lengths
vector_lengths = np.linalg.norm(displacement_vectors, axis=1)
max_length = np.max(vector_lengths)

# Normalize vectors for better visualization
scaling_factor = 0.2 / max_length * 3  # Make arrows 3x bigger
scaled_vectors = displacement_vectors * scaling_factor

# Sample 50% of the data to reduce clutter
sample_indices = np.random.choice(len(cartesian_initial_points), len(cartesian_initial_points) // 8, replace=False)
sampled_initial_points = cartesian_initial_points[sample_indices]
sampled_vectors = scaled_vectors[sample_indices]
sampled_colors = vector_lengths[sample_indices]

# Plotting the arrows with improved visibility
colors = cm.jet(sampled_colors/max_length)
# Plotting the arrows with improved visibility
for i, start_point in enumerate(sampled_initial_points):
    end_point = start_point + sampled_vectors[i]
    color = colors[i]
    ax.quiver(start_point[0], start_point[1], start_point[2],
              sampled_vectors[i, 0], sampled_vectors[i, 1], sampled_vectors[i, 2],
              color=color, linewidth=2, arrow_length_ratio=0.3)

# Plot the outline of the tetrahedron
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
ax.plot_trisurf(*vertices.T, color='r', alpha=0.1)
# Adding edges for the tetrahedron
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

# Calculate the barycentric coordinates for these points
barycentric_coords = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
labels = ['AB', 'Ab', 'aB', 'ab']
barycentric_to_cartesian = []

for i, element in enumerate(barycentric_coords):
    barycentric_to_cartesian.append(np.dot(vertices.T, element))

barycentric_to_cartesian = np.array(barycentric_to_cartesian)

cartesian_convergent_points = cartesian_convergent_points[::8]
ax.scatter(cartesian_convergent_points[:, 0], cartesian_convergent_points[:, 1], cartesian_convergent_points[:, 2], c='red', marker='x', label='Convergent Points')

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


# Grid and Axes enhancements
ax.view_init(elev=25, azim=-105)
# Remove tick labels to declutter:
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.grid(color="white", linestyle='solid')
plt.tight_layout()
# Save the image to your desktop with high DPI for better quality
filename = "/Users/noah/Desktop/3.png"
plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)

# Close the plt figure to release memory
plt.close()

# Crop the image to remove any whitespace (using PIL)
img = Image.open(filename)
cropped_img = img.crop(img.getbbox())
cropped_img.save(filename)

