import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree


# Define parameters
num_units = 50
num_iterations = 300
max_speed = 0.015
min_speed = -0.015
neighbor_distance = 0.1
swarm_center = np.array([0.5, 0.5])  # Center of the simulation area
alignment_factor = 0.015  # Factor influencing alignment behavior
cohesion_factor = 0.015   # Factor influencing cohesion behavior
separation_factor = 0.02 # Factor influencing separation behavior
trail_decay = 0.95  # Trail decay factor
building_repulsion = 0.01  # Factor influencing building avoidance behavior


# Generate urban landscape (random buildings)
def generate_buildings(num_buildings, min_size=0.01, max_size=0.1):
    buildings = []
    for _ in range(num_buildings):
        size = np.random.uniform(min_size, max_size)
        x = np.random.uniform(0, 1 - size)
        y = np.random.uniform(0, 1 - size)
        buildings.append((x, y, size))
    return buildings


# Generate military unit positions
def generate_unit_positions(num_units):
    return np.random.rand(num_units, 2)


# Initialize unit positions and velocities
positions = generate_unit_positions(num_units)
velocities = np.random.uniform(min_speed, max_speed, (num_units, 2))


# Initialize KD-tree for efficient nearest neighbor queries
kdtree = cKDTree(positions)


# Initialize figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


# Create empty lists to store trail data
trail_segments = []


# Generate urban landscape
buildings = generate_buildings(100)  # Adjust the number of buildings as needed


# Plot buildings
for building in buildings:
    x, y, size = building
    rect = plt.Rectangle((x, y), size, size, color='gray', alpha=0.5)
    ax.add_patch(rect)


# Define update function for animation
def update(frame):
    global positions, velocities, kdtree, trail_segments
   
    # Clear previous trail segments
    for line in trail_segments:
        line.remove()
    trail_segments.clear()
   
    # Update unit positions based on velocities
    positions += velocities
   
    # Apply wrap-around behavior to keep units within simulation area
    positions = positions % 1
   
    # Update KD-tree with new positions
    kdtree = cKDTree(positions)
   
    # Generate new trail segments and update velocities
    for i in range(num_units):
        # Query nearest neighbors to determine intensity
        neighbor_indices = kdtree.query_ball_point(positions[i], neighbor_distance)
        num_neighbors = len(neighbor_indices)
       
        # Calculate alignment, cohesion, and separation vectors
        alignment = np.mean(velocities[neighbor_indices], axis=0) - velocities[i]
        cohesion = np.mean(positions[neighbor_indices], axis=0) - positions[i]
        separation = np.mean(positions[neighbor_indices], axis=0) - positions[i]
       
        # Update velocity based on alignment, cohesion, and separation
        velocities[i] += (alignment_factor * alignment + cohesion_factor * cohesion - separation_factor * separation)
       
        # Avoid buildings
        for building in buildings:
            bx, by, bsize = building
            dist = np.linalg.norm(positions[i] - np.array([bx + bsize / 2, by + bsize / 2]))
            if dist < bsize:
                velocities[i] += building_repulsion * (positions[i] - np.array([bx + bsize / 2, by + bsize / 2])) / (dist ** 2)
       
        # Limit maximum speed
        speed = np.linalg.norm(velocities[i])
        if speed > max_speed:
            velocities[i] *= max_speed / speed
       
        # Generate trail segment
        trail_x = [positions[i, 0]]
        trail_y = [positions[i, 1]]
        for neighbor_index in neighbor_indices:
            trail_x.append(positions[neighbor_index, 0])
            trail_y.append(positions[neighbor_index, 1])
       
        # Adjust color and alpha based on intensity
        intensity = num_neighbors / num_units
        trail, = ax.plot(trail_x, trail_y, 'o-', color='darkgreen', alpha=intensity*trail_decay)
        trail_segments.append(trail)
   
    return trail_segments


# Flag to toggle behavior
move_towards_edge = False


# Function to handle mouse click event
def onclick(event):
    global velocities, move_towards_edge
    move_towards_edge = not move_towards_edge
    if move_towards_edge:
        print("Move towards edge mode activated.")
    else:
        print("Normal mode activated.")


# Attach mouse click event handler
cid = fig.canvas.mpl_connect('button_press_event', onclick)


# Function to move towards the nearest building edge
def move_towards_building_edge():
    for i in range(num_units):
        # Find nearest building edge
        nearest_edge_dist = float('inf')
        nearest_edge = None
        for building in buildings:
            bx, by, bsize = building
            dist_to_edge = min(abs(positions[i, 0] - bx), abs(positions[i, 0] - (bx + bsize)), abs(positions[i, 1] - by), abs(positions[i, 1] - (by + bsize)))
            if dist_to_edge < nearest_edge_dist:
                nearest_edge_dist = dist_to_edge
                nearest_edge = (bx, by, bsize)
       
        # Calculate vector from unit to nearest building edge
        bx, by, bsize = nearest_edge
        edge_position = np.array([bx + bsize / 2, by + bsize / 2])
        direction = edge_position - positions[i]
       
        # Normalize and scale the vector to adjust the velocity
        velocities[i] += 0.05 * direction / np.linalg.norm(direction)


# Create animation
animation = FuncAnimation(fig, update, frames=num_iterations, interval=50, blit=True)


def toggle_behavior(frame):
    if move_towards_edge:
        move_towards_building_edge()


# Timer to toggle behavior every 0.2 seconds
toggle_timer = fig.canvas.new_timer(interval=200)
toggle_timer.add_callback(toggle_behavior, 0)
toggle_timer.start()


plt.show()
