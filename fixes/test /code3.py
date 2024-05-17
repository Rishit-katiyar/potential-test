import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree


# Define simulation parameters
num_soldiers = 50  # Increase number of soldiers
num_iterations = 800  # Increase number of simulation iterations
max_speed = 0.03  # Increase maximum speed
min_speed = -0.03  # Increase minimum speed
neighbor_distance = 0.2  # Increase neighbor distance
alignment_factor = 0.03  # Increase alignment factor
cohesion_factor = 0.03  # Increase cohesion factor
separation_factor = 0.04  # Increase separation factor
trail_decay = 0.95  # Trail decay factor
building_repulsion = 0.03  # Increase building repulsion
explosion_radius = 0.15  # Increase explosion radius


# Define types of facilities within the military base with their maximum capacities
facility_types = {
    'Barracks': 50,
    'Armory': 30,
    'Command Center': 20,
    'Training Grounds': 40,
    'Mess Hall': 60,
    'Medical Center': 15
}


# Generate layout of the military base (large buildings representing facilities)
def generate_base_layout():
    buildings = []
    for facility_type, max_capacity in facility_types.items():
        for _ in range(np.random.randint(1, 4)):  # Randomly vary the number of each facility type
            size = np.random.uniform(0.05, 0.2)
            x = np.random.uniform(0, 1 - size)
            y = np.random.uniform(0, 1 - size)
            buildings.append((x, y, size, facility_type, max_capacity))
    return buildings


# Generate initial positions of soldiers within the military base
def generate_soldiers_positions(num_soldiers):
    return np.random.rand(num_soldiers, 2)


# Initialize positions and velocities of soldiers
positions = generate_soldiers_positions(num_soldiers)
velocities = np.random.uniform(min_speed, max_speed, (num_soldiers, 2))


# Initialize KD-tree for efficient nearest neighbor queries
kdtree = cKDTree(positions)


# Initialize figure and axis for visualization
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


# Create empty lists to store trail data and building patches
trail_segments = []
building_patches = []


# Generate layout of the military base
facilities = generate_base_layout()  # Adjust the number of facilities as needed


# Plot facilities and label them with their capacity
for idx, facility in enumerate(facilities):
    x, y, size, facility_type, max_capacity = facility
    rect = plt.Rectangle((x, y), size, size, color='gray', alpha=0.5)
    ax.add_patch(rect)
    building_text = plt.text(x + size / 2, y + size / 2, f"{facility_type}\nCapacity: {max_capacity}", ha='center', va='center', fontsize=8, color='white')
    building_patches.append((rect, building_text))


# Define update function for animation
def update(frame):
    global positions, velocities, kdtree, trail_segments
   
    # Clear previous trail segments
    for line in trail_segments:
        line.remove()
    trail_segments.clear()
   
    # Update positions of soldiers based on velocities
    positions += velocities
   
    # Apply wrap-around behavior to keep soldiers within the military base
    positions = positions % 1
   
    # Update KD-tree with new positions
    kdtree = cKDTree(positions)
   
    # Track the number of soldiers near each facility
    facility_occupancy = {facility: 0 for facility in facilities}
   
    # Assign soldiers to facilities based on their proximity
    for i in range(num_soldiers):
        nearest_facility = None
        nearest_facility_dist = float('inf')
        for facility in facilities:
            fx, fy, fsize, _, max_capacity = facility
            dist = np.linalg.norm(positions[i] - np.array([fx + fsize / 2, fy + fsize / 2]))
            if dist < nearest_facility_dist:
                nearest_facility_dist = dist
                nearest_facility = facility
        if nearest_facility:
            if facility_occupancy[nearest_facility] < max_capacity:
                facility_occupancy[nearest_facility] += 1
            else:
                # Soldier moves away from the facility if it's at full capacity
                direction = positions[i] - np.array([nearest_facility[0] + nearest_facility[2] / 2, nearest_facility[1] + nearest_facility[2] / 2])
                velocities[i] += 0.05 * direction / np.linalg.norm(direction)
   
    # Generate new trail segments and update velocities
    for i in range(num_soldiers):
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
        for facility in facilities:
            fx, fy, fsize, _, _ = facility
            dist = np.linalg.norm(positions[i] - np.array([fx + fsize / 2, fy + fsize / 2]))
            if dist < fsize:
                velocities[i] += building_repulsion * (positions[i] - np.array([fx + fsize / 2, fy + fsize / 2])) / (dist ** 2)
       
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
        intensity = num_neighbors / num_soldiers
        trail, = ax.plot(trail_x, trail_y, 'o-', color='darkgreen', alpha=intensity * trail_decay)
        trail_segments.append(trail)
   
    return trail_segments


# Flag to toggle behavior
move_towards_edge = False


# Function to handle mouse click event for toggling behavior and bombing
def onclick(event):
    global velocities, move_towards_edge, facilities, building_patches
   
    # Create a copy of building_patches to iterate over
    building_patches_copy = building_patches.copy()
   
    if event.button == 1:  # Left mouse button
        # Check if the click is on a building
        for rect, building_text in building_patches_copy:
            fx, fy = rect.get_xy()
            size = rect.get_width()
            if fx <= event.xdata <= fx + size and fy <= event.ydata <= fy + size:
                # Remove the building and update the plot
                rect.remove()
                building_text.set_text("Destroyed")
                building_text.set_color('red')
                building_patches.remove((rect, building_text))
                for facility in facilities:
                    if (fx, fy, size, facility[3], facility[4]) == facility:
                        facilities.remove(facility)
                break
       
        # If the click is not on a building, simulate bomb detonation
        else:
            detonation_position = np.array([event.xdata, event.ydata])
            for i in range(num_soldiers):
                dist_to_detonation = np.linalg.norm(positions[i] - detonation_position)
                if dist_to_detonation < explosion_radius:
                    # Modify soldier's velocity based on the distance to detonation
                    direction = positions[i] - detonation_position
                    velocities[i] += 0.05 * direction / np.linalg.norm(direction)
   
    if event.button == 3:  # Right mouse button
        move_towards_edge = not move_towards_edge
        if move_towards_edge:
            print("Evacuation towards edges of facilities activated.")
        else:
            print("Normal evacuation mode activated.")


# Attach mouse click event handler
cid = fig.canvas.mpl_connect('button_press_event', onclick)


# Function to move soldiers towards the nearest facility edge during evacuation
def move_towards_facility_edge():
    for i in range(num_soldiers):
        # Find nearest facility edge
        nearest_facility_dist = float('inf')
        nearest_facility = None
        for facility in facilities:
            fx, fy, fsize, _, _ = facility
            dist_to_edge = min(abs(positions[i, 0] - fx), abs(positions[i, 0] - (fx + fsize)), abs(positions[i, 1] - fy), abs(positions[i, 1] - (fy + fsize)))
            if dist_to_edge < nearest_facility_dist:
                nearest_facility_dist = dist_to_edge
                nearest_facility = facility
       
        if nearest_facility:
            # Calculate vector from soldier to nearest facility edge
            fx, fy, fsize, _, _ = nearest_facility
            edge_position = np.array([fx + fsize / 2, fy + fsize / 2])
            direction = edge_position - positions[i]
           
            # Normalize and scale the vector to adjust the velocity
            velocities[i] += 0.05 * direction / np.linalg.norm(direction)


# Create animation
animation = FuncAnimation(fig, update, frames=num_iterations, interval=50, blit=True)


# Timer to toggle behavior every 0.2 seconds
def toggle_behavior(frame):
    if move_towards_edge:
        move_towards_facility_edge()
       
    # Refresh canvas every 0.2 seconds
    fig.canvas.draw_idle()


# Timer to toggle behavior
toggle_timer = fig.canvas.new_timer(interval=200)
toggle_timer.add_callback(toggle_behavior, 0)
toggle_timer.start()


# Function to periodically adjust figure size to prevent refresh issue
def adjust_figure_size():
    fig.set_size_inches(fig.get_size_inches() + np.array([0.0000000000001, 0.0000000000001]))  # Increase figure size by 1 pixel
    plt.pause(0.0001)  # Pause for 0.0001 seconds
    fig.set_size_inches(fig.get_size_inches() - np.array([0.0000000000001, 0.0000000000001]))  # Decrease figure size by 1 pixel
    plt.pause(0.0001)  # Pause for 0.0001 seconds


# Start the loop to adjust figure size periodically
while True:
    adjust_figure_size()


plt.show()
