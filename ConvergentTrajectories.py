import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

convergent_points = []

# Define coordinates of the tetrahedron vertices
a_vertex = np.array([np.sqrt(8/9), 0, -1/3])
b_vertex = np.array([-np.sqrt(2/9), np.sqrt(2/3), -1/3])
c_vertex = np.array([-np.sqrt(2/9), -np.sqrt(2/3), -1/3])
d_vertex = np.array([0,0,1])

def is_point_in_tetrahedron(point, a_vertex, b_vertex, c_vertex, d_vertex, buffer=5):
    vertices = [a_vertex, b_vertex, c_vertex, d_vertex]
    max_dist = max(np.linalg.norm(v1-v2) for v1 in vertices for v2 in vertices)
    
    for vertex in vertices:
        if np.linalg.norm(point - vertex) > max_dist + buffer:
            return False
    return True

def calculate_next_generation(x, r, delta, alpha, beta, gamma):
    D = x[0]*x[3] - x[1]*x[2] 
    w_bar = 1 - delta*(x[0]**2 + x[3]**2) - alpha*(x[1]**2 + x[2]**2) - 2*beta*(x[2]*x[3] + x[0]*x[1]) - 2*gamma*(x[0]*x[2] + x[1]*x[3])
    x_next = np.zeros(4)
    x_next[0] = (x[0] - delta*x[0]**2 - beta*x[0]*x[1] - gamma*x[0]*x[2] - r*D)/w_bar
    x_next[1] = (x[1] - beta*x[0]*x[1] - alpha*x[1]**2 - gamma*x[1]*x[3] + r*D)/w_bar
    x_next[2] = (x[2] - gamma*x[0]*x[2] - alpha*x[2]**2 - beta*x[2]*x[3] + r*D)/w_bar
    x_next[3] = (x[3] - gamma*x[1]*x[3] - beta*x[2]*x[3] - delta*x[3]**2 - r*D)/ w_bar
    #D_next = x_next[0]*x_next[3] - x_next[1]*x_next[2]

    # Normalize x_next so that it sums to 1
    x_next /= np.sum(x_next)

    return x_next

def iterate_generations(x, delta, alpha, beta, gamma):
    stable_generations = 0  # Initialize counter for stable generations
    change_threshold = 8*10**-4  # Set change threshold

    while True:
        # Save current x for later comparison
        x_old = x.copy()
        r = 0.04
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
        if stable_generations >= 100:
            break

#Alpha-Gamma translates Markov update equations as shown in Karlin and Felman "Linkage and selection: two locus symmetric viability model"
a = 0.02
b = 0.04
d = 0.03
g = b

points = []

total_iterations = 100  # Number of tests
for i in range(total_iterations):
    iterate_generations(np.random.dirichlet(np.ones(4), size=1)[0],d,a,b,g) 

    # Print the final convergent point
    print(f"Convergent point: {points[-1]}")
    print("Sum of convergent point: " + str(np.sum(points[-1])))
    convergent_points.append(points[-1])

    # Total progress calculation
    progress = (i + 1) / total_iterations * 100
    print(f"Total progress: {progress:.2f}%")


# Convert points list to numpy array for easier manipulation
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
points = np.array(points)

cartesian_points = []  # Define the cartesian_points list
total_points = len(points)

for i, element in enumerate(points):
    cartesian_points.append(np.dot(vertices.T, element))
    progress = (i + 1) / total_points * 100
    

filtered_points = []

for point in cartesian_points:
    if is_point_in_tetrahedron(point, a_vertex, b_vertex, c_vertex, d_vertex):
        filtered_points.append(point)

points = np.array(filtered_points)

# Extract coordinates for plotting
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

print("Density")
# Convert points to a 2D array with each row being a point
points = np.vstack([x, y, z])
# Calculate the point density
density = stats.gaussian_kde(points)(points)
print("Density done")
# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Sort the points by density, so that the densest points are plotted last
idx = density.argsort()
x, y, z, density = x[idx], y[idx], z[idx], density[idx]

# Use density as the color
sc = ax.scatter(x, y, z, c=density, cmap='jet', alpha=0.005)

print("Outline")

# Plot the outline of the tetrahedron
vertices = np.array([a_vertex, b_vertex, c_vertex, d_vertex])
ax.plot_trisurf(*vertices.T, color='r', alpha=0.1)

# Add a colorbar for the density
fig.colorbar(sc, ax=ax, label='Density')

# Calculate the barycentric coordinates for these points
barycentric_coords = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0.248,0.251,0.249,0.252],[0.497,0.001,0.002,0.500],[0.001,0.497,0.500,0.002],[0.1539,0.6562,0.036,0.1539],[0.1539,0.036,0.6562,0.1539],[0.8888,0.05,0.05,0.0112],[0.0112,0.05,0.05,0.8888]])
labels = ['AB', 'Ab', 'aB', 'ab','s1','s2','s3','p1','p2','p3','p4']
barycentric_to_cartesian = []

for i, element in enumerate(barycentric_coords):
    barycentric_to_cartesian.append(np.dot(vertices.T, element))

barycentric_to_cartesian = np.array(barycentric_to_cartesian)

for i in range(len(barycentric_to_cartesian)):
    x, y, z = barycentric_to_cartesian[i]
    ax.scatter(x, y, z, color='black')
    ax.text(x, y, z, labels[i], color='black')

# Extract convergent points for plotting
cartesian_convergent_points = []  
convergent_points = np.array(convergent_points)
print(convergent_points)

for i, p in enumerate(convergent_points):
    cartesian_convergent_points.append(np.dot(vertices.T, p))

cartesian_convergent_points = np.array(cartesian_convergent_points)
x_conv = cartesian_convergent_points[:, 0]
y_conv = cartesian_convergent_points[:, 1]
z_conv = cartesian_convergent_points[:, 2]

# Plot convergent points in different color and size
convergent_scatter = ax.scatter(x_conv, y_conv, z_conv, c='green', s=85, label='Convergent Points')

#Display convergent point barycentric coords on graph
convergent_points = np.around(convergent_points, 4)
unique_points = [convergent_points[0]]
for point in convergent_points[1:]:
    if not any(np.allclose(point, unique_point, atol=0.0175) for unique_point in unique_points):
        unique_points.append(point)
unique_points_str = '\n'.join(map(str, unique_points))
ax.text2D(0.05, 0.95, f'Convergent points:\n{unique_points_str}', transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# # Add a legend
# ax.legend()

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Random R (R = [placeholder], alpha: {a}, beta: {b}, delta: {d})')

# Show the plot
plt.show()
print("Done")