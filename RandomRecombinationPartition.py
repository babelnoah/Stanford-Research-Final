from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
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
    x_next /= np.sum(x_next)

    return x_next

def iterate_generations(x, delta, alpha, beta, gamma):
    stable_generations = 0  # Initialize counter for stable generations
    change_threshold = 1.3*10**-6 # Set change threshold
    total_generations =0

    while True:
        total_generations +=1
        # Save current x for later comparison
        x_old = x.copy()

        #Recombination Fraction (0.25 Placeholder)
        #r = 0.2
        r = np.random.uniform(0,0.25)

        x = calculate_next_generation(x, r, delta, alpha, beta, gamma)

        # Check if absolute change in all components of x is less than threshold
        if np.all(np.abs(x - x_old) < change_threshold):
            stable_generations += 1
        else:
            # Reset counter if change is above threshold
            stable_generations = 0

        points.append(x.tolist())

        # Break loop if x has been stable for 100 generations
        if stable_generations >= 100:
            break
        # if total_generations >= 1000:
        #     print(total_generations)
        #     break


#Alpha-Gamma translates Markov update equations as shown in Karlin and Felman "Linkage and selection: two locus symmetric viability model"
a = 0.3
b = 0.4
d = 0.1
g = 0.4

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
print(convergent_points)

# Start plotting
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

mesh_points = np.column_stack((x_conv, y_conv, z_conv))

# Apply DBSCAN to identify clusters
db = DBSCAN(eps=0.3, min_samples=10).fit(mesh_points)
labels = db.labels_

# Identify the number of clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Initialize a list to store the points for each cluster
clusters = [mesh_points[labels == i] for i in range(n_clusters)]

# Initialize a dict to store the initial points for each cluster
initial_clusters = {i: [] for i in range(n_clusters)}

# Build the mapping of initial points to their cluster
for initial_point, convergent_point in zip(initial_points, convergent_points):
    # Transform the convergent point to Cartesian coordinates
    cartesian_convergent_point = np.dot(vertices.T, convergent_point)
    # Find the label of the convergent_point
    label = db.labels_[np.where((mesh_points == cartesian_convergent_point).all(axis=1))[0][0]]
    if label != -1:  # Exclude noise points
        # Append the initial_point to the corresponding cluster
        initial_clusters[label].append(initial_point)

# Convert lists to np arrays for easier manipulation and transform to Cartesian coordinates
for label in initial_clusters.keys():
    initial_clusters[label] = np.array([np.dot(vertices.T, point) for point in initial_clusters[label]])

# Plot the points for each cluster
for i, cluster in enumerate(clusters):

    unique_cluster_points = np.unique(cluster, axis=0)

    ax.scatter(*unique_cluster_points.T, color=f'C{i}', s=50, cmap="viridis", edgecolor='k', linewidth=0.02)##################################################

    # Plot the cluster points
    #ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], alpha=0.1, zorder=1, color=f'C{i}')

    # Plot the initial points for the cluster with more opacity
    initial_cluster = initial_clusters[i]    
    # Compute the Convex Hull for the initial points of the current cluster
    hull_initial = ConvexHull(initial_cluster)

    # Plot the Convex Hull for the initial points
    for s in hull_initial.simplices:
        ax.plot_trisurf(initial_cluster[s, 0], initial_cluster[s, 1], initial_cluster[s, 2], alpha=0.25, color=f'C{i}')

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

x_conv = cartesian_convergent_points[:, 0]
y_conv = cartesian_convergent_points[:, 1]
z_conv = cartesian_convergent_points[:, 2]

# Plot convergent points in different color and size
#convergent_scatter = ax.scatter(x_conv, y_conv, z_conv, c='green', s=30, label='Convergent Points')

#Display convergent point barycentric coords on graph
convergent_points = np.around(convergent_points, 4)
unique_points = [convergent_points[0]]
for point in convergent_points[1:]:
    if not any(np.allclose(point, unique_point, atol=0.0175) for unique_point in unique_points):
        unique_points.append(point)
unique_points_str = '\n'.join(map(str, unique_points))

#POTENTIAL PARAMETRIC DIAGONAL OVERLAY
# # Convert barycentric coordinates of diagonal to cartesian
# barycentric_coords_diagonal = []
# for t in np.linspace(0, 1, 100):
#     x1_t = 0.5 * (1-t)
#     x2_t = 0.5 * t
#     x3_t = 0.5 * t
#     x4_t = 0.5 * (1-t)
#     barycentric_coords_diagonal.append([x1_t, x2_t, x3_t, x4_t])

# cartesian_diagonal = [np.dot(vertices.T, coord) for coord in barycentric_coords_diagonal]
# cartesian_diagonal = np.array(cartesian_diagonal)

# # Plotting the curve with modifications for aesthetics
# ax.plot(cartesian_diagonal[:, 0], 
#         cartesian_diagonal[:, 1], 
#         cartesian_diagonal[:, 2], 
#         color='#34A853',  # Adjusted green color
#         linestyle='--',  # Dashed line
#         linewidth=1.5,   # Line width
#         alpha=0.8,       # Slight transparency
#         label='Diagonal') # Label for legend


# Add a legend
ax.view_init(elev=25, azim=-105)
ax.w_xaxis.pane.fill = ax.w_yaxis.pane.fill = ax.w_zaxis.pane.fill = False

# Remove tick labels to declutter:
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.grid(color="white", linestyle='solid')
plt.tight_layout()


# Save the image to your desktop with high DPI for better quality
filename = "/Users/noah/Desktop/6.png"
plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
# Close the plt figure to release memory
plt.close()

# Crop the image to remove any whitespace (using PIL)
img = Image.open(filename)
cropped_img = img.crop(img.getbbox())
cropped_img.save(filename)