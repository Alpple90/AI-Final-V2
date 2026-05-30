# gui.py - tkinter GUI for TBRGS

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from config import WINDOW_WIDTH, WINDOW_HEIGHT, LEFT_PANEL_WIDTH


class TBRGSGUI:

    def __init__(self, root, map_viewer, pathfinder):
        self.root = root
        self.map_viewer = map_viewer
        self.pathfinder = pathfinder

        self.current_model = tk.StringVar(value='lstm')
        self.origin_var = tk.StringVar()
        self.dest_var = tk.StringVar()
        self.time_var = tk.StringVar(value="12:00")

        self.map_widget = None
        self.results_text = None
        self.status_var = None
        self.current_paths = []

        self.setup_window()
        self.build_gui()

        if self.map_viewer.map_available():
            self.init_map()
        else:
            self.show_map_unavail()

    # set the window title, size and background colour
    def setup_window(self):
        self.root.title("TBRGS - Traffic-Based Route Guidance System")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg='#f0f0f0')

    # lay out the left control panel and right map panel
    def build_gui(self):
        left_panel = tk.Frame(self.root, bg='#f0f0f0', width=LEFT_PANEL_WIDTH)
        left_panel.pack(side='left', fill='both', expand=False, padx=(10, 5), pady=10)
        left_panel.pack_propagate(False)

        self.right_panel = tk.Frame(self.root, bg='#ffffff', bd=2, relief='sunken')
        self.right_panel.pack(side='right', fill='both', expand=True, padx=(5, 10), pady=10)

        tk.Label(left_panel, text="TRAFFIC-BASED ROUTE GUIDANCE",
                font=('Arial', 14, 'bold'), bg='#f0f0f0', fg='#1a237e').pack(pady=(0, 5))

        self.status_var = tk.StringVar(value="")
        tk.Label(left_panel, textvariable=self.status_var, font=('Arial', 9),
                bg='#f0f0f0', fg='#2e7d32', wraplength=LEFT_PANEL_WIDTH-20,
                justify='left').pack(fill='x', padx=10, pady=(0, 10))

        self.build_input_frame(left_panel)
        self.build_model_frame(left_panel)
        self.build_btn_frame(left_panel)
        self.build_route_selector(left_panel)
        self.build_results_frame(left_panel)

    # build the origin/destination/time input section
    def build_input_frame(self, parent):
        input_frame = tk.LabelFrame(parent, text="Trip Information",
                                    font=('Arial', 11, 'bold'),
                                    bg='#f0f0f0', padx=10, pady=10)
        input_frame.pack(fill='x', pady=(0, 10))

        tk.Label(input_frame, text="Origin SCATS:", font=('Arial', 10),
                bg='#f0f0f0').grid(row=0, column=0, sticky='w', pady=5)
        self.origin_combo = ttk.Combobox(input_frame, textvariable=self.origin_var, width=20)
        self.origin_combo.grid(row=0, column=1, pady=5, padx=(10, 0))
        tk.Button(input_frame, text="Locate", command=self.locate_origin,
                 font=('Arial', 8), width=6).grid(row=0, column=2, padx=5)

        tk.Label(input_frame, text="Destination SCATS:", font=('Arial', 10),
                bg='#f0f0f0').grid(row=1, column=0, sticky='w', pady=5)
        self.dest_combo = ttk.Combobox(input_frame, textvariable=self.dest_var, width=20)
        self.dest_combo.grid(row=1, column=1, pady=5, padx=(10, 0))
        tk.Button(input_frame, text="Locate", command=self.locate_dest,
                 font=('Arial', 8), width=6).grid(row=1, column=2, padx=5)

        tk.Label(input_frame, text="Departure Time:", font=('Arial', 10),
                bg='#f0f0f0').grid(row=2, column=0, sticky='w', pady=5)
        tk.Entry(input_frame, textvariable=self.time_var, width=10,
                font=('Arial', 10)).grid(row=2, column=1, sticky='w', pady=5, padx=(10, 0))
        tk.Label(input_frame, text="(HH:MM, 24hr)", font=('Arial', 8),
                bg='#f0f0f0').grid(row=2, column=1, sticky='e', padx=(0, 10))

    # add radio buttons for choosing LSTM, GRU or XGBoost
    def build_model_frame(self, parent):
        model_frame = tk.LabelFrame(parent, text="ML Model Selection",
                                    font=('Arial', 11, 'bold'),
                                    bg='#f0f0f0', padx=10, pady=10)
        model_frame.pack(fill='x', pady=(0, 10))

        models = [('LSTM', 'lstm'), ('GRU', 'gru'), ('XGBoost', 'xgboost')]
        for i, (text, value) in enumerate(models):
            tk.Radiobutton(model_frame, text=text, variable=self.current_model,
                          value=value, font=('Arial', 10), bg='#f0f0f0').grid(
                          row=0, column=i, padx=20, pady=5)

    # add the find routes, clear map and compare algorithms buttons
    def build_btn_frame(self, parent):
        btn_frame = tk.Frame(parent, bg='#f0f0f0')
        btn_frame.pack(fill='x', pady=(0, 10))

        row1 = tk.Frame(btn_frame, bg='#f0f0f0')
        row1.pack(fill='x', pady=(0, 5))

        tk.Button(row1, text="FIND ROUTES",
                  command=self.find_routes,
                  bg='#2e7d32', fg='white',
                  font=('Arial', 11, 'bold'),
                  padx=20, pady=5).pack(side='left', padx=5)

        tk.Button(row1, text="CLEAR MAP",
                  command=self.clear_route,
                  bg='#ef6c00', fg='white',
                  font=('Arial', 10),
                  padx=15, pady=5).pack(side='left', padx=5)

        row2 = tk.Frame(btn_frame, bg='#f0f0f0')
        row2.pack(fill='x')

        tk.Button(row2, text="COMPARE ALGORITHMS",
                  command=self.compare_algos,
                  bg='#9c27b0', fg='white',
                  font=('Arial', 10),
                  padx=15, pady=5).pack(side='left', padx=5)

    # create the row of colour-coded route buttons (populated after search)
    def build_route_selector(self, parent):
        self.route_selector_frame = tk.LabelFrame(parent, text="Display Route",
                                                   font=('Arial', 11, 'bold'),
                                                   bg='#f0f0f0', padx=10, pady=5)
        self.route_selector_frame.pack(fill='x', pady=(0, 10))
        self.route_btns_row = tk.Frame(self.route_selector_frame, bg='#f0f0f0')
        self.route_btns_row.pack(fill='x')
        tk.Label(self.route_selector_frame, text="Find routes to see options.",
                 font=('Arial', 9), bg='#f0f0f0', fg='#888888').pack()

    # add the scrollable text box where route results are printed
    def build_results_frame(self, parent):
        results_frame = tk.LabelFrame(parent, text="Route Results (Top-K Routes)",
                                      font=('Arial', 11, 'bold'),
                                      bg='#f0f0f0', padx=10, pady=10)
        results_frame.pack(fill='both', expand=True)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=14,
                                                       width=55, font=('Courier', 9))
        self.results_text.pack(fill='both', expand=True)

    # create the map widget and draw the SCATS network on it
    def init_map(self):
        self.map_widget = self.map_viewer.create_map(self.right_panel)
        self.map_viewer.draw_network()
        self.update_site_lists()

    # show an error message in place of the map when tkintermapview isn't installed
    def show_map_unavail(self):
        tk.Label(self.right_panel,
                text="Map visualization unavailable.\n\nPlease install tkintermapview:\npip install tkintermapview",
                font=('Arial', 12), bg='#ffffff', fg='#ff0000').pack(expand=True)

    # fill both dropdowns with available SCATS sites
    def update_site_lists(self):
        sites = self.map_viewer.get_sites()
        self.origin_combo['values'] = sites
        self.dest_combo['values'] = sites
        if len(sites) >= 2:
            self.origin_combo.set(str(sites[0]))
            self.dest_combo.set(str(sites[1]))

    # pan the map to the selected origin site
    def locate_origin(self):
        site = self.origin_var.get()
        if site:
            self.map_viewer.locate_site(site)
            self.status_var.set(f"Located SCATS {site}")

    # pan the map to the selected destination site
    def locate_dest(self):
        site = self.dest_var.get()
        if site:
            self.map_viewer.locate_site(site)
            self.status_var.set(f"Located SCATS {site}")

    # rebuild the route selector buttons to match the latest search results
    def populate_route_btns(self):
        for w in self.route_btns_row.winfo_children():
            w.destroy()
        for w in self.route_selector_frame.winfo_children():
            if isinstance(w, tk.Label):
                w.destroy()

        route_colors = ['#ff6f00', '#1565c0', '#6a1b9a', '#00838f', '#558b2f']
        for i, (_, total_time, _) in enumerate(self.current_paths):
            color = route_colors[i % len(route_colors)]
            btn = tk.Button(self.route_btns_row,
                            text=f"Route {i+1}",
                            command=lambda idx=i: self.select_route(idx),
                            bg=color, fg='white',
                            font=('Arial', 9, 'bold'),
                            padx=8, pady=3,
                            relief='sunken' if i == 0 else 'raised')
            btn.pack(side='left', padx=3, pady=3)

    # highlight the chosen route on the map and press its button in
    def select_route(self, idx):
        route_colors = ['#ff6f00', '#1565c0', '#6a1b9a', '#00838f', '#558b2f']
        path, _, _ = self.current_paths[idx]
        self.map_viewer.clear_route()
        self.map_viewer.draw_route(path, color=route_colors[idx % len(route_colors)], is_best=True)
        for i, btn in enumerate(self.route_btns_row.winfo_children()):
            btn.config(relief='sunken' if i == idx else 'raised')

    # wipe the drawn route off the map and clear the results text box
    def clear_route(self):
        self.map_viewer.clear_route()
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Route cleared from map.")

    # grab and validate origin, destination and departure hour from the form
    def get_user_input(self):
        origin_str = self.origin_var.get()
        dest_str = self.dest_var.get()

        if not origin_str or not dest_str:
            messagebox.showwarning("Input Error", "Select origin and destination")
            return None, None, None

        try:
            origin = int(origin_str)
            dest = int(dest_str)
        except ValueError:
            messagebox.showwarning("Input Error", "Invalid site selection")
            return None, None, None

        if origin == dest:
            messagebox.showwarning("Input Error", "Origin and destination must be different")
            return None, None, None

        try:
            time_str = self.time_var.get()
            hour = int(time_str.split(':')[0]) if ':' in time_str else int(time_str)
            hour = max(0, min(23, hour))
        except:
            hour = 12

        return origin, dest, hour

    # kick off a route search using all 6 algorithms and show the top results
    def find_routes(self):
        origin, dest, hour = self.get_user_input()
        if origin is None:
            return

        model_name = self.current_model.get()

        self.status_var.set(f"Finding routes from {origin} to {dest} using all algorithms "
                            f"with {model_name.upper()}...")
        self.root.update()

        self.results_text.delete(1.0, tk.END)
        self.map_viewer.clear_route()

        self.pathfinder.set_model(model_name)

        paths = self.pathfinder.find_unique_paths(origin, dest, hour, max_paths=5)

        self.current_paths = paths
        if paths:
            best_algos = " & ".join(a.upper() for a in paths[0][2])
            self.status_var.set(f"Found {len(paths)} unique route(s). Best: {paths[0][1]:.1f} minutes ({best_algos})")
        else:
            self.status_var.set("No routes found. Try different origin/destination.")
        self.populate_route_btns()
        if paths:
            self.select_route(0)

        self.display_results(origin, dest, hour, model_name, paths)

    # format and print all found routes into the scrollable results box
    def display_results(self, origin, dest, hour, model_name, paths):
        SEP = "=" * 40
        self.results_text.insert(tk.END, SEP + "\n")
        self.results_text.insert(tk.END, "TBRGS ROUTE RESULTS\n")
        self.results_text.insert(tk.END, SEP + "\n")
        self.results_text.insert(tk.END, f"Origin:    SCATS {origin}\n")
        self.results_text.insert(tk.END, f"Dest:      SCATS {dest}\n")
        self.results_text.insert(tk.END, f"Departure: {self.time_var.get()} (Hour {hour}:00)\n")
        self.results_text.insert(tk.END, f"ML Model:  {model_name.upper()}\n")
        self.results_text.insert(tk.END, SEP + "\n\n")

        if not paths:
            self.results_text.insert(tk.END, "No routes found!\n\n")
            return

        algo_display = {
            'astar': 'A*', 'bidirectional': 'Bidirectional A*',
            'dijkstra': "Dijkstra's", 'greedy': 'Greedy', 'bfs': 'BFS', 'dfs': 'DFS'
        }

        self.results_text.insert(tk.END, f"Found {len(paths)} unique route(s):\n\n")

        for i, (path, total_time, algos) in enumerate(paths, 1):
            algo_names = " & ".join(algo_display.get(a, a) for a in algos)
            self.results_text.insert(tk.END, "─" * 40 + "\n")
            self.results_text.insert(tk.END, f"ROUTE {i} │ {total_time:.1f} min\n")
            self.results_text.insert(tk.END, f"By: {algo_names}\n\n")

            # wrap node list so it fits the text box
            nodes = [str(n) for n in path]
            curr_line = []
            curr_len = 0
            for node in nodes:
                if curr_len + len(node) + 4 > 40:
                    self.results_text.insert(tk.END, "  " + " → ".join(curr_line) + "\n")
                    curr_line = [node]
                    curr_len = len(node)
                else:
                    curr_line.append(node)
                    curr_len += len(node) + 4
            if curr_line:
                self.results_text.insert(tk.END, "  " + " → ".join(curr_line) + "\n")

            self.results_text.insert(tk.END, "\n")

        self.results_text.insert(tk.END, SEP + "\n")

    # run all 6 algorithms on the same trip and print a side-by-side comparison
    def compare_algos(self):
        origin, dest, hour = self.get_user_input()
        if origin is None:
            return

        model_name = self.current_model.get()

        self.status_var.set(f"Comparing all algorithms from {origin} to {dest}...")
        self.root.update()

        self.results_text.delete(1.0, tk.END)

        SEP = "=" * 40
        self.results_text.insert(tk.END, SEP + "\n")
        self.results_text.insert(tk.END, "ALGORITHM COMPARISON\n")
        self.results_text.insert(tk.END, SEP + "\n")
        self.results_text.insert(tk.END, f"Origin: {origin}  Dest: {dest}\n")
        self.results_text.insert(tk.END, f"Time: {self.time_var.get()}  Model: {model_name.upper()}\n\n")

        self.results_text.insert(tk.END, f"{'Algorithm':<16} {'min':<8} {'Nodes':<7} \n")
        self.results_text.insert(tk.END, "-" * 40 + "\n")

        algorithms = ['astar', 'bidirectional', 'dijkstra', 'greedy', 'bfs', 'dfs']
        algo_names = ['A*', 'Bidir A*', 'Dijkstra', 'Greedy', 'BFS', 'DFS']

        best_time = float('inf')
        best_algo = None

        for algo, name in zip(algorithms, algo_names):
            self.pathfinder.set_algorithm(algo)
            self.pathfinder.set_model(model_name)
            path, cost, nodes = self.pathfinder.find_path(origin, dest, hour)

            if path:
                self.results_text.insert(tk.END, f"{name:<16} {cost:<8.1f} {nodes:<7} Y\n")
                if cost < best_time:
                    best_time = cost
                    best_algo = name
            else:
                self.results_text.insert(tk.END, f"{name:<16} {'N/A':<8} {nodes:<7} N\n")

        self.results_text.insert(tk.END, "-" * 40 + "\n")
        if best_algo:
            self.results_text.insert(tk.END, f"\nBest: {best_algo} ({best_time:.1f} min)\n")

        self.results_text.insert(tk.END, "\n" + SEP + "\n")
        self.status_var.set(f"Comparison complete. Best: {best_algo} ({best_time:.1f} min)")
