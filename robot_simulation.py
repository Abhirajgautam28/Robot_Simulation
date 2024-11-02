import matplotlib.pyplot as plt
import numpy as np

warehouse_width = 10
warehouse_height = 10
start_position = np.array([0, 0], dtype=float)

obstacles = [
    (3, 3, 2, 1),
    (6, 5, 1, 2),
    (2, 7, 3, 1),
]

people = [
    np.array([1, 1], dtype=float),
    np.array([4, 6], dtype=float),
    np.array([8, 2], dtype=float),
]

robot_speed = 1.0
robot_step_distance = robot_speed * 0.1
collision_tolerance = 0.2
person_speed = 0.2


def is_collision(position, obstacles):
    for (x, y, width, height) in obstacles:
        if x <= position[0] <= x + width and y <= position[1] <= y + height:
            return True
    return False


def generate_random_destination(obstacles):
    while True:
        end_position = np.random.uniform(0, warehouse_width, size=2)
        if not is_collision(end_position, obstacles):
            return end_position


plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(0, warehouse_width)
ax.set_ylim(0, warehouse_height)
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_title('Autonomous Robot Path in Warehouse with Obstacle Avoidance')
ax.scatter(*start_position, color='green', label='Start')

end_position = generate_random_destination(obstacles)
ax.scatter(*end_position, color='#FFD700', marker='*', s=150, label='Destination')
robot_dot, = ax.plot([], [], marker='s', color='black', markersize=6, label='Robot')

# Draw static obstacles
for (x, y, width, height) in obstacles:
    rect = plt.Rectangle((x, y), width, height, color='gray', alpha=0.7)
    ax.add_patch(rect)

person_dots = []
for i, person in enumerate(people):
    person_dot, = ax.plot(person[0], person[1], marker='o', color='red', markersize=10,
                          label='Moving Workers' if i == 0 else "")
    person_dots.append(person_dot)

plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1, 1))
plt.grid(True)

robot_path_history = [start_position.copy()]


def find_alternate_direction(current_position, target_position, obstacles):
    for angle in np.linspace(-90, 90, num=12):
        rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                    [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
        new_direction = np.dot(rotation_matrix, target_position - current_position)
        new_position = current_position + new_direction * robot_step_distance

        if not is_collision(new_position, obstacles):
            return new_direction / np.linalg.norm(new_direction)
    return None


def check_person_proximity(robot_position):
    for person in people:
        distance = np.linalg.norm(robot_position - person)
        if distance < collision_tolerance:
            return person
    return None


def move_person_safely(person):
    for _ in range(10):
        move_direction = np.random.rand(2) * 2 - 1
        move_direction /= np.linalg.norm(move_direction)
        new_position = person + move_direction * person_speed
        if (0 <= new_position[0] <= warehouse_width and 0 <= new_position[1] <= warehouse_height
                and not is_collision(new_position, obstacles)):
            return new_position
    return person


current_position = start_position.copy()
robot_dot.set_data([current_position[0]], [current_position[1]])

while np.linalg.norm(current_position - end_position) > robot_step_distance:
    for i in range(len(people)):
        people[i] = move_person_safely(people[i])
        person_dots[i].set_data([people[i][0]], [people[i][1]])

    original_direction = end_position - current_position
    direction_vector = original_direction / np.linalg.norm(original_direction)

    next_position = current_position + direction_vector * robot_step_distance
    collision_with_obstacle = is_collision(next_position, obstacles)

    person = check_person_proximity(next_position)

    if collision_with_obstacle or person is not None:
        direction_vector = find_alternate_direction(current_position, end_position, obstacles)
        if direction_vector is None:
            break

    current_position += direction_vector * robot_step_distance
    current_position = np.clip(current_position, 0, warehouse_width)
    robot_path_history.append(current_position.copy())
    robot_dot.set_data([current_position[0]], [current_position[1]])

    path_x, path_y = zip(*robot_path_history)
    ax.plot(path_x, path_y, color='green', linestyle='--', linewidth=1)

    plt.draw()
    plt.pause(0.01)

plt.ioff()
plt.show()
