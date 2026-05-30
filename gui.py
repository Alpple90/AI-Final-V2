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

        self._setup_window()
        self._build_gui()

        if self.map_viewer.is_map_available():
            self._init_map()
        else:
            self._show_map_unavailable()

    def _setup_window(self):
        self.root.title("TBRGS - Traffic-Based Route Guidance System")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg='#f0f0f0')

    def _build_gui(self):
        leftPanel = tk.Frame(self.root, bg='#f0f0f0', width=LEFT_PANEL_WIDTH)
        leftPanel.pack(side='left', fill='both', expand=False, padx=(10, 5), pady=10)
        leftPanel.pack_propagate(False)

        self.right_panel = tk.Frame(self.root, bg='#ffffff', bd=2, relief='sunken')
        self.right_panel.pack(side='right', fill='both', expand=True, padx=(5, 10), pady=10)

        tk.Label(leftPanel, text="TRAFFIC-BASED ROUTE GUIDANCE",
                font=('Arial', 14, 'bold'), bg='#f0f0f0', fg='#1a237e').pack(pady=(0, 5))

        self.status_var = tk.StringVar(value="")
        tk.Label(leftPanel, textvariable=self.status_var, font=('Arial', 9),
                bg='#f0f0f0', fg='#2e7d32', wraplength=LEFT_PANEL_WIDTH-20,
                justify='left').pack(fill='x', padx=10, pady=(0, 10))

        self._build_input_frame(leftPanel)
        self._build_model_frame(leftPanel)
        self._build_button_frame(leftPanel)
        self._build_route_selector(leftPanel)
        self._build_results_frame(leftPanel)

    def _build_input_frame(self, parent):
        inputFrame = tk.LabelFrame(parent, text="Trip Information",
                                    font=('Arial', 11, 'bold'),
                                    bg='#f0f0f0', padx=10, pady=10)
        inputFrame.pack(fill='x', pady=(0, 10))

        tk.Label(inputFrame, text="Origin SCATS:", font=('Arial', 10),
                bg='#f0f0f0').grid(row=0, column=0, sticky='w', pady=5)
        self.origin_combo = ttk.Combobox(inputFrame, textvariable=self.origin_var, width=20)
        self.origin_combo.grid(row=0, column=1, pady=5, padx=(10, 0))
        tk.Button(inputFrame, text="Locate", command=self._locate_origin,
                 font=('Arial', 8), width=6).grid(row=0, column=2, padx=5)

        tk.Label(inputFrame, text="Destination SCATS:", font=('Arial', 10),
                bg='#f0f0f0').grid(row=1, column=0, sticky='w', pady=5)
        self.dest_combo = ttk.Combobox(inputFrame, textvariable=self.dest_var, width=20)
        self.dest_combo.grid(row=1, column=1, pady=5, padx=(10, 0))
        tk.Button(inputFrame, text="Locate", command=self._locate_destination,
                 font=('Arial', 8), width=6).grid(row=1, column=2, padx=5)

        tk.Label(inputFrame, text="Departure Time:", font=('Arial', 10),
                bg='#f0f0f0').grid(row=2, column=0, sticky='w', pady=5)
        tk.Entry(inputFrame, textvariable=self.time_var, width=10,
                font=('Arial', 10)).grid(row=2, column=1, sticky='w', pady=5, padx=(10, 0))
        tk.Label(inputFrame, text="(HH:MM, 24hr)", font=('Arial', 8),
                bg='#f0f0f0').grid(row=2, column=1, sticky='e', padx=(0, 10))

    def _build_model_frame(self, parent):
        modelFrame = tk.LabelFrame(parent, text="ML Model Selection",
                                    font=('Arial', 11, 'bold'),
                                    bg='#f0f0f0', padx=10, pady=10)
        modelFrame.pack(fill='x', pady=(0, 10))

        models = [('LSTM', 'lstm'), ('GRU', 'gru'), ('XGBoost', 'xgboost')]
        for i, (text, value) in enumerate(models):
            tk.Radiobutton(modelFrame, text=text, variable=self.current_model,
                          value=value, font=('Arial', 10), bg='#f0f0f0').grid(
                          row=0, column=i, padx=20, pady=5)

    def _build_button_frame(self, parent):
        btnFrame = tk.Frame(parent, bg='#f0f0f0')
        btnFrame.pack(fill='x', pady=(0, 10))

        row1 = tk.Frame(btnFrame, bg='#f0f0f0')
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

        row2 = tk.Frame(btnFrame, bg='#f0f0f0')
        row2.pack(fill='x')

        tk.Button(row2, text="COMPARE ALGORITHMS",
                  command=self.compare_algorithms,
                  bg='#9c27b0', fg='white',
                  font=('Arial', 10),
                  padx=15, pady=5).pack(side='left', padx=5)

    def _build_route_selector(self, parent):
        self.route_selector_frame = tk.LabelFrame(parent, text="Display Route",
                                                   font=('Arial', 11, 'bold'),
                                                   bg='#f0f0f0', padx=10, pady=5)
        self.route_selector_frame.pack(fill='x', pady=(0, 10))
        self.route_buttons_row = tk.Frame(self.route_selector_frame, bg='#f0f0f0')
        self.route_buttons_row.pack(fill='x')
        tk.Label(self.route_selector_frame, text="Find routes to see options.",
                 font=('Arial', 9), bg='#f0f0f0', fg='#888888').pack()

    def _build_results_frame(self, parent):
        resultsFrame = tk.LabelFrame(parent, text="Route Results (Top-K Routes)",
                                      font=('Arial', 11, 'bold'),
                                      bg='#f0f0f0', padx=10, pady=10)
        resultsFrame.pack(fill='both', expand=True)

        self.results_text = scrolledtext.ScrolledText(resultsFrame, height=14,
                                                       width=55, font=('Courier', 9))
        self.results_text.pack(fill='both', expand=True)

    def _init_map(self):
        self.map_widget = self.map_viewer.create_map(self.right_panel)
        self.map_viewer.draw_network()
        self._update_site_lists()

    def _show_map_unavailable(self):
        tk.Label(self.right_panel,
                text="Map visualization unavailable.\n\nPlease install tkintermapview:\npip install tkintermapview",
                font=('Arial', 12), bg='#ffffff', fg='#ff0000').pack(expand=True)

    def _update_site_lists(self):
        # fill both dropdowns with available SCATS sites
        sites = self.map_viewer.get_available_sites()
        self.origin_combo['values'] = sites
        self.dest_combo['values'] = sites
        if len(sites) >= 2:
            self.origin_combo.set(str(sites[0]))
            self.dest_combo.set(str(sites[1]))

    def _locate_origin(self):
        site = self.origin_var.get()
        if site:
            self.map_viewer.locate_site(site)
            self.status_var.set(f"Located SCATS {site}")

    def _locate_destination(self):
        site = self.dest_var.get()
        if site:
            self.map_viewer.locate_site(site)
            self.status_var.set(f"Located SCATS {site}")

    def _populate_route_buttons(self):
        for w in self.route_buttons_row.winfo_children():
            w.destroy()
        for w in self.route_selector_frame.winfo_children():
            if isinstance(w, tk.Label):
                w.destroy()

        routeColors = ['#ff6f00', '#1565c0', '#6a1b9a', '#00838f', '#558b2f']
        for i, (_, total_time, _) in enumerate(self.current_paths):
            color = routeColors[i % len(routeColors)]
            btn = tk.Button(self.route_buttons_row,
                            text=f"Route {i+1}",
                            command=lambda idx=i: self._select_route(idx),
                            bg=color, fg='white',
                            font=('Arial', 9, 'bold'),
                            padx=8, pady=3,
                            relief='sunken' if i == 0 else 'raised')
            btn.pack(side='left', padx=3, pady=3)

    def _select_route(self, idx):
        routeColors = ['#ff6f00', '#1565c0', '#6a1b9a', '#00838f', '#558b2f']
        path, _, _ = self.current_paths[idx]
        self.map_viewer.clear_route()
        self.map_viewer.draw_route(path, color=routeColors[idx % len(routeColors)], is_best=True)
        for i, btn in enumerate(self.route_buttons_row.winfo_children()):
            btn.config(relief='sunken' if i == idx else 'raised')

    def clear_route(self):
        self.map_viewer.clear_route()
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Route cleared from map.")

    def get_user_input(self):
        # grab and validate origin/dest/time from the form
        originStr = self.origin_var.get()
        destStr = self.dest_var.get()

        if not originStr or not destStr:
            messagebox.showwarning("Input Error", "Select origin and destination")
            return None, None, None

        try:
            origin = int(originStr)
            dest = int(destStr)
        except ValueError:
            messagebox.showwarning("Input Error", "Invalid site selection")
            return None, None, None

        if origin == dest:
            messagebox.showwarning("Input Error", "Origin and destination must be different")
            return None, None, None

        try:
            timeStr = self.time_var.get()
            hour = int(timeStr.split(':')[0]) if ':' in timeStr else int(timeStr)
            hour = max(0, min(23, hour))
        except:
            hour = 12

        return origin, dest, hour

    def find_routes(self):
        # run all 6 algorithms and collect up to 5 unique routes
        origin, dest, hour = self.get_user_input()
        if origin is None:
            return

        modelName = self.current_model.get()

        self.status_var.set(f"Finding routes from {origin} to {dest} using all algorithms "
                            f"with {modelName.upper()}...")
        self.root.update()

        self.results_text.delete(1.0, tk.END)
        self.map_viewer.clear_route()

        self.pathfinder.set_model(modelName)

        paths = self.pathfinder.find_unique_paths_all_algorithms(origin, dest, hour, max_paths=5)

        self.current_paths = paths
        if paths:
            bestAlgos = " & ".join(a.upper() for a in paths[0][2])
            self.status_var.set(f"Found {len(paths)} unique route(s). Best: {paths[0][1]:.1f} minutes ({bestAlgos})")
        else:
            self.status_var.set("No routes found. Try different origin/destination.")
        self._populate_route_buttons()
        if paths:
            self._select_route(0)

        self._display_results(origin, dest, hour, modelName, paths)

    def _display_results(self, origin, dest, hour, modelName, paths):
        SEP = "=" * 40
        self.results_text.insert(tk.END, SEP + "\n")
        self.results_text.insert(tk.END, "TBRGS ROUTE RESULTS\n")
        self.results_text.insert(tk.END, SEP + "\n")
        self.results_text.insert(tk.END, f"Origin:    SCATS {origin}\n")
        self.results_text.insert(tk.END, f"Dest:      SCATS {dest}\n")
        self.results_text.insert(tk.END, f"Departure: {self.time_var.get()} (Hour {hour}:00)\n")
        self.results_text.insert(tk.END, f"ML Model:  {modelName.upper()}\n")
        self.results_text.insert(tk.END, SEP + "\n\n")

        if not paths:
            self.results_text.insert(tk.END, "No routes found!\n\n")
            return

        algoDisplay = {
            'astar': 'A*', 'bidirectional': 'Bidirectional A*',
            'dijkstra': "Dijkstra's", 'greedy': 'Greedy', 'bfs': 'BFS', 'dfs': 'DFS'
        }

        self.results_text.insert(tk.END, f"Found {len(paths)} unique route(s):\n\n")

        for i, (path, total_time, algos) in enumerate(paths, 1):
            algoNames = " & ".join(algoDisplay.get(a, a) for a in algos)
            self.results_text.insert(tk.END, "─" * 40 + "\n")
            self.results_text.insert(tk.END, f"ROUTE {i} │ {total_time:.1f} min\n")
            self.results_text.insert(tk.END, f"By: {algoNames}\n\n")

            # wrap node list so it fits the text box
            nodes = [str(n) for n in path]
            currLine = []
            currLen = 0
            for node in nodes:
                if currLen + len(node) + 4 > 40:
                    self.results_text.insert(tk.END, "  " + " → ".join(currLine) + "\n")
                    currLine = [node]
                    currLen = len(node)
                else:
                    currLine.append(node)
                    currLen += len(node) + 4
            if currLine:
                self.results_text.insert(tk.END, "  " + " → ".join(currLine) + "\n")

            self.results_text.insert(tk.END, "\n")

        self.results_text.insert(tk.END, SEP + "\n")

    def compare_algorithms(self):
        origin, dest, hour = self.get_user_input()
        if origin is None:
            return

        modelName = self.current_model.get()

        self.status_var.set(f"Comparing all algorithms from {origin} to {dest}...")
        self.root.update()

        self.results_text.delete(1.0, tk.END)

        SEP = "=" * 40
        self.results_text.insert(tk.END, SEP + "\n")
        self.results_text.insert(tk.END, "ALGORITHM COMPARISON\n")
        self.results_text.insert(tk.END, SEP + "\n")
        self.results_text.insert(tk.END, f"Origin: {origin}  Dest: {dest}\n")
        self.results_text.insert(tk.END, f"Time: {self.time_var.get()}  Model: {modelName.upper()}\n\n")

        self.results_text.insert(tk.END, f"{'Algorithm':<16} {'min':<8} {'Nodes':<7} \n")
        self.results_text.insert(tk.END, "-" * 40 + "\n")

        algorithms = ['astar', 'bidirectional', 'dijkstra', 'greedy', 'bfs', 'dfs']
        algoNames = ['A*', 'Bidir A*', 'Dijkstra', 'Greedy', 'BFS', 'DFS']

        bestTime = float('inf')
        bestAlgo = None

        for algo, name in zip(algorithms, algoNames):
            self.pathfinder.set_algorithm(algo)
            self.pathfinder.set_model(modelName)
            path, cost, nodes = self.pathfinder.find_path(origin, dest, hour)

            if path:
                self.results_text.insert(tk.END, f"{name:<16} {cost:<8.1f} {nodes:<7} Y\n")
                if cost < bestTime:
                    bestTime = cost
                    bestAlgo = name
            else:
                self.results_text.insert(tk.END, f"{name:<16} {'N/A':<8} {nodes:<7} N\n")

        self.results_text.insert(tk.END, "-" * 40 + "\n")
        if bestAlgo:
            self.results_text.insert(tk.END, f"\nBest: {bestAlgo} ({bestTime:.1f} min)\n")

        self.results_text.insert(tk.END, "\n" + SEP + "\n")
        self.status_var.set(f"Comparison complete. Best: {bestAlgo} ({bestTime:.1f} min)")
