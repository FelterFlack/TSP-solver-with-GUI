import tkinter as tk
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Solver")
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<B1-Motion>", self.drag_point)
        self.canvas.bind("<ButtonRelease-1>", self.release_point)
        self.canvas.bind("<Button-3>", self.start_obstacle)
        self.canvas.bind("<B3-Motion>", self.drag_obstacle)
        self.canvas.bind("<ButtonRelease-3>", self.release_obstacle)

        self.points = []
        self.obstacles = []
        self.dragging_point = None
        self.current_obstacle = None

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.solve_button = tk.Button(self.button_frame, text="Solve TSP (GA)", command=self.solve_tsp)
        self.solve_button.pack(side=tk.LEFT)

        self.solve_aco_button = tk.Button(self.button_frame, text="Solve TSP (ACO)", command=self.solve_tsp_aco)
        self.solve_aco_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self.reset)
        self.reset_button.pack(side=tk.LEFT)

        self.undo_button = tk.Button(self.button_frame, text="Undo", command=self.undo)
        self.undo_button.pack(side=tk.LEFT)

        self.redo_button = tk.Button(self.button_frame, text="Redo", command=self.redo)
        self.redo_button.pack(side=tk.LEFT)

        self.distance_label = tk.Label(self.button_frame, text="Total Distance: N/A")
        self.distance_label.pack(side=tk.LEFT)

        self.history = []
        self.future = []

        self.param_frame = tk.Frame(root)
        self.param_frame.pack()

        self.pop_size_label = tk.Label(self.param_frame, text="Population Size:")
        self.pop_size_label.pack(side=tk.LEFT)
        self.pop_size_entry = tk.Entry(self.param_frame)
        self.pop_size_entry.insert(0, "100")
        self.pop_size_entry.pack(side=tk.LEFT)

        self.gen_num_label = tk.Label(self.param_frame, text="Generations:")
        self.gen_num_label.pack(side=tk.LEFT)
        self.gen_num_entry = tk.Entry(self.param_frame)
        self.gen_num_entry.insert(0, "500")
        self.gen_num_entry.pack(side=tk.LEFT)

        self.mut_rate_label = tk.Label(self.param_frame, text="Mutation Rate:")
        self.mut_rate_label.pack(side=tk.LEFT)
        self.mut_rate_entry = tk.Entry(self.param_frame)
        self.mut_rate_entry.insert(0, "0.01")
        self.mut_rate_entry.pack(side=tk.LEFT)

        self.aco_ant_count_label = tk.Label(self.param_frame, text="Ant Count:")
        self.aco_ant_count_label.pack(side=tk.LEFT)
        self.aco_ant_count_entry = tk.Entry(self.param_frame)
        self.aco_ant_count_entry.insert(0, "50")
        self.aco_ant_count_entry.pack(side=tk.LEFT)

        self.aco_gen_num_label = tk.Label(self.param_frame, text="ACO Generations:")
        self.aco_gen_num_label.pack(side=tk.LEFT)
        self.aco_gen_num_entry = tk.Entry(self.param_frame)
        self.aco_gen_num_entry.insert(0, "100")
        self.aco_gen_num_entry.pack(side=tk.LEFT)

        self.fig, self.ax = plt.subplots()
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_plot.get_tk_widget().pack()

    def add_point(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        self.history.append((self.points[:], self.obstacles[:]))
        self.future = []
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")

    def drag_point(self, event):
        if self.points:
            self.points[-1] = (event.x, event.y)
            self.canvas.delete("point")
            for x, y in self.points:
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black", tags="point")

    def release_point(self, event):
        self.dragging_point = None

    def start_obstacle(self, event):
        self.current_obstacle = (event.x, event.y, event.x, event.y)
        self.obstacles.append(self.current_obstacle)
        self.history.append((self.points[:], self.obstacles[:]))
        self.future = []

    def drag_obstacle(self, event):
        if self.current_obstacle:
            x1, y1, _, _ = self.current_obstacle
            self.current_obstacle = (x1, y1, event.x, event.y)
            self.obstacles[-1] = self.current_obstacle
            self.canvas.delete("obstacle")
            for x1, y1, x2, y2 in self.obstacles:
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="red", tags="obstacle")

    def release_obstacle(self, event):
        self.current_obstacle = None

    def reset(self):
        self.history = []
        self.future = []
        self.points = []
        self.obstacles = []
        self.canvas.delete("all")
        self.distance_label.config(text="Total Distance: N/A")
        self.ax.clear()
        self.canvas_plot.draw()

    def undo(self):
        if not self.history:
            return
        self.future.append((self.points[:], self.obstacles[:]))
        last_action = self.history.pop()
        self.points, self.obstacles = last_action
        self.canvas.delete("all")
        for x, y in self.points:
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")
        for x1, y1, x2, y2 in self.obstacles:
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="red")
        self.distance_label.config(text="Total Distance: N/A")

    def redo(self):
        if not self.future:
            return
        self.history.append((self.points[:], self.obstacles[:]))
        next_action = self.future.pop()
        self.points, self.obstacles = next_action
        self.canvas.delete("all")
        for x, y in self.points:
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")
        for x1, y1, x2, y2 in self.obstacles:
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="red")
        self.distance_label.config(text="Total Distance: N/A")

    def solve_tsp(self):
        if len(self.points) < 2:
            return

        try:
            pop_size = int(self.pop_size_entry.get())
            num_generations = int(self.gen_num_entry.get())
            mutation_rate = float(self.mut_rate_entry.get())
        except ValueError:
            return

        cities = np.array(self.points)
        best_route, best_distance, fitness_history = genetic_algorithm(cities, self.obstacles, self.canvas, self.distance_label, pop_size, num_generations, mutation_rate)

        self.canvas.delete("route")
        for i in range(len(best_route) - 1):
            x1, y1 = cities[best_route[i]]
            x2, y2 = cities[best_route[i+1]]
            self.canvas.create_line(x1, y1, x2, y2, fill="gray", tags="route")
        x1, y1 = cities[best_route[-1]]
        x2, y2 = cities[best_route[0]]
        self.canvas.create_line(x1, y1, x2, y2, fill="gray", tags="route")

        self.distance_label.config(text=f"Total Distance: {best_distance:.2f}")

        # Plot fitness history
        self.ax.clear()
        self.ax.plot(fitness_history)
        self.ax.set_title("Fitness over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Distance")
        self.canvas_plot.draw()

    def solve_tsp_aco(self):
        if len(self.points) < 2:
            return

        try:
            num_ants = int(self.aco_ant_count_entry.get())
            num_generations = int(self.aco_gen_num_entry.get())
        except ValueError:
            return

        cities = np.array(self.points)
        best_route, best_distance, fitness_history = ant_colony_optimization(cities, self.obstacles, self.canvas, self.distance_label, num_ants, num_generations)

        self.canvas.delete("route")
        for i in range(len(best_route) - 1):
            x1, y1 = cities[best_route[i]]
            x2, y2 = cities[best_route[i+1]]
            self.canvas.create_line(x1, y1, x2, y2, fill="gray", tags="route")
        x1, y1 = cities[best_route[-1]]
        x2, y2 = cities[best_route[0]]
        self.canvas.create_line(x1, y1, x2, y2, fill="gray", tags="route")

        self.distance_label.config(text=f"Total Distance: {best_distance:.2f}")

        # Plot fitness history
        self.ax.clear()
        self.ax.plot(fitness_history)
        self.ax.set_title("Fitness over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Distance")
        self.canvas_plot.draw()
        
    def draw_route(self, route, cities, tag, color="blue"):
        self.canvas.delete(tag)
        for i in range(len(route) - 1):
            x1, y1 = cities[route[i]]
            x2, y2 = cities[route[i+1]]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, tags=tag)
        x1, y1 = cities[route[-1]]
        x2, y2 = cities[route[0]]
        self.canvas.create_line(x1, y1, x2, y2, fill=color, tags=tag)


def distance(city1, city2):
    return np.sqrt(np.sum((city1 - city2)**2))

def create_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

def fitness(route, cities, obstacles):
    total_distance = 0
    for i in range(len(route)-1):
        d = distance(cities[route[i]], cities[route[i+1]])
        if intersects_obstacle(cities[route[i]], cities[route[i+1]], obstacles):
            d *= 10  # Çarpan, engel bulunan rotaları cezalandırmak için
        total_distance += d
    total_distance += distance(cities[route[-1]], cities[route[0]])
    if intersects_obstacle(cities[route[-1]], cities[route[0]], obstacles):
        total_distance *= 10  # Çarpan, engel bulunan rotaları cezalandırmak için
    return total_distance

def intersects_obstacle(p1, p2, obstacles):
    for x1, y1, x2, y2 in obstacles:
        if do_lines_intersect(p1, p2, (x1, y1), (x2, y2)):
            return True
    return False

def do_lines_intersect(p1, p2, p3, p4):
    # Check if line segments p1p2 and p3p4 intersect
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def selection(population, cities, obstacles, num_offspring, tournament_size=5):
    selected = []
    for _ in range(num_offspring):
        tournament = random.sample(population, tournament_size)
        winner = min(tournament, key=lambda route: fitness(route, cities, obstacles))
        selected.append(winner)
    return selected

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None]*size
    child[start:end] = parent1[start:end]
    pointer = 0
    for i in range(size):
        if child[i] is None:
            while parent2[pointer] in child:
                pointer += 1
            child[i] = parent2[pointer]
    return child

def mutate(route, mutation_rate=0.01):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm(cities, obstacles, canvas, distance_label, pop_size=100, num_generations=500, mutation_rate=0.01, elitism=True):
    population = create_population(pop_size, len(cities))
    best_route = min(population, key=lambda route: fitness(route, cities, obstacles))
    best_distance = fitness(best_route, cities, obstacles)
    fitness_history = [best_distance]

    for generation in range(num_generations):
        if elitism:
            elite_size = int(pop_size * 0.1)
            elite = selection(population, cities, obstacles, elite_size)
        else:
            elite = []

        population = selection(population, cities, obstacles, pop_size // 2)
        offspring = []
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i], population[(i+1) % len(population)]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            offspring.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        population = elite + offspring
        current_best_route = min(population, key=lambda route: fitness(route, cities, obstacles))
        current_best_distance = fitness(current_best_route, cities, obstacles)

        if current_best_distance < best_distance:
            best_route, best_distance = current_best_route, current_best_distance

        fitness_history.append(current_best_distance)

        # Çizgileri güncelle
        canvas.delete("route")
        for i in range(len(current_best_route) - 1):
            x1, y1 = cities[current_best_route[i]]
            x2, y2 = cities[current_best_route[i+1]]
            canvas.create_line(x1, y1, x2, y2, fill="gray", tags="route")
        x1, y1 = cities[current_best_route[-1]]
        x2, y2 = cities[current_best_route[0]]
        canvas.create_line(x1, y1, x2, y2, fill="gray", tags="route")

        distance_label.config(text=f"Generation: {generation+1}, Distance: {current_best_distance:.2f}")
        canvas.update()
        canvas.after(1)  # Kısa bir bekleme süresi

    return best_route, best_distance, fitness_history

# ACO Algorithm Functions
def ant_colony_optimization(cities, obstacles, canvas, distance_label, num_ants=50, num_generations=100, alpha=1.0, beta=5.0, rho=0.5, Q=100):
    num_cities = len(cities)
    pheromone = np.ones((num_cities, num_cities))
    best_route = None
    best_distance = float('inf')
    fitness_history = []

    for generation in range(num_generations):
        all_routes = []
        all_distances = []

        for ant in range(num_ants):
            route = generate_route(cities, pheromone, alpha, beta, obstacles)
            distance = fitness(route, cities, obstacles)
            all_routes.append(route)
            all_distances.append(distance)

            if distance < best_distance:
                best_distance = distance
                best_route = route

            # Visualize each ant's route
            app.draw_route(route, cities, f"ant_route_{ant}", color="blue")

        pheromone = (1 - rho) * pheromone
        for route, distance in zip(all_routes, all_distances):
            for i in range(num_cities - 1):
                pheromone[route[i]][route[i+1]] += Q / distance
            pheromone[route[-1]][route[0]] += Q / distance

        fitness_history.append(best_distance)

        # Visualize the best route
        app.draw_route(best_route, cities, "best_route", color="red")

        distance_label.config(text=f"ACO Generation: {generation+1}, Distance: {best_distance:.2f}")
        canvas.update()
        canvas.after(1)  # Short delay to visualize
        
    for generation in range(num_generations):
        for ant in range(num_ants):
            canvas.delete(f"ant_route_{ant}")

    return best_route, best_distance, fitness_history



def generate_route(cities, pheromone, alpha, beta, obstacles):
    num_cities = len(cities)
    route = []
    unvisited = list(range(num_cities))
    current_city = random.choice(unvisited)
    route.append(current_city)
    unvisited.remove(current_city)

    while unvisited:
        probabilities = []
        for city in unvisited:
            pheromone_value = pheromone[current_city][city] ** alpha
            heuristic_value = (1 / distance(cities[current_city], cities[city])) ** beta
            probabilities.append(pheromone_value * heuristic_value)
        probabilities = np.array(probabilities) / sum(probabilities)
        next_city = np.random.choice(unvisited, p=probabilities)
        route.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city

    return route


if __name__ == "__main__":
    root = tk.Tk()
    app = TSPApp(root)
    root.mainloop()

