# gui.py
"""
Tkinter GUI for TBRGS with Top-K Routes and Algorithm Selection
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from config import WINDOW_WIDTH, WINDOW_HEIGHT, LEFT_PANEL_WIDTH


class TBRGSGUI:
    """Main GUI application for TBRGS"""
    
    def __init__(self, root, map_viewer, pathfinder):
        self.root = root
        self.map_viewer = map_viewer
        self.pathfinder = pathfinder
        
        # GUI variables
        self.current_model = tk.StringVar(value='lstm')
        self.origin_var = tk.StringVar()
        self.dest_var = tk.StringVar()
        self.time_var = tk.StringVar(value="12:00")
        
        # Map widgets reference
        self.map_widget = None
        self.results_text = None
        self.status_var = None
        
        # Setup window
        self._setup_window()
        self._build_gui()
        
        # Initialize map
        if self.map_viewer.is_map_available():
            self._init_map()
        else:
            self._show_map_unavailable()
    
    def _setup_window(self):
        self.root.title("TBRGS - Traffic-Based Route Guidance System")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg='#f0f0f0')
    
    def _build_gui(self):
        # Left panel
        left_panel = tk.Frame(self.root, bg='#f0f0f0', width=LEFT_PANEL_WIDTH)
        left_panel.pack(side='left', fill='both', expand=False, padx=(10, 5), pady=10)
        left_panel.pack_propagate(False)
        
        # Right panel for map
        self.right_panel = tk.Frame(self.root, bg='#ffffff', bd=2, relief='sunken')
        self.right_panel.pack(side='right', fill='both', expand=True, padx=(5, 10), pady=10)
        
        # Title
        tk.Label(left_panel, text="TRAFFIC-BASED ROUTE GUIDANCE", 
                font=('Arial', 14, 'bold'), bg='#f0f0f0', fg='#1a237e').pack(pady=(0, 15))
        
        # Build all frames
        self._build_input_frame(left_panel)
        self._build_model_frame(left_panel)
        self._build_button_frame(left_panel)
        self._build_results_frame(left_panel)
        self._build_status_bar()
    
    def _build_input_frame(self, parent):
        input_frame = tk.LabelFrame(parent, text="Trip Information", 
                                     font=('Arial', 11, 'bold'), 
                                     bg='#f0f0f0', padx=10, pady=10)
        input_frame.pack(fill='x', pady=(0, 10))
        
        # Origin
        tk.Label(input_frame, text="Origin SCATS:", font=('Arial', 10), 
                bg='#f0f0f0').grid(row=0, column=0, sticky='w', pady=5)
        self.origin_combo = ttk.Combobox(input_frame, textvariable=self.origin_var, width=20)
        self.origin_combo.grid(row=0, column=1, pady=5, padx=(10, 0))
        tk.Button(input_frame, text="📍", command=self._locate_origin,
                 font=('Arial', 10), width=3).grid(row=0, column=2, padx=5)
        
        # Destination
        tk.Label(input_frame, text="Destination SCATS:", font=('Arial', 10), 
                bg='#f0f0f0').grid(row=1, column=0, sticky='w', pady=5)
        self.dest_combo = ttk.Combobox(input_frame, textvariable=self.dest_var, width=20)
        self.dest_combo.grid(row=1, column=1, pady=5, padx=(10, 0))
        tk.Button(input_frame, text="📍", command=self._locate_destination,
                 font=('Arial', 10), width=3).grid(row=1, column=2, padx=5)
        
        # Time
        tk.Label(input_frame, text="Departure Time:", font=('Arial', 10), 
                bg='#f0f0f0').grid(row=2, column=0, sticky='w', pady=5)
        tk.Entry(input_frame, textvariable=self.time_var, width=10, 
                font=('Arial', 10)).grid(row=2, column=1, sticky='w', pady=5, padx=(10, 0))
        tk.Label(input_frame, text="(HH:MM, 24hr)", font=('Arial', 8), 
                bg='#f0f0f0').grid(row=2, column=1, sticky='e', padx=(0, 10))
    
    def _build_model_frame(self, parent):
        model_frame = tk.LabelFrame(parent, text="ML Model Selection", 
                                     font=('Arial', 11, 'bold'), 
                                     bg='#f0f0f0', padx=10, pady=10)
        model_frame.pack(fill='x', pady=(0, 10))
        
        models = [('LSTM', 'lstm'), ('GRU', 'gru'), ('XGBoost', 'xgboost')]
        for i, (text, value) in enumerate(models):
            tk.Radiobutton(model_frame, text=text, variable=self.current_model,
                          value=value, font=('Arial', 10), bg='#f0f0f0').grid(
                          row=0, column=i, padx=20, pady=5)
    
    def _build_button_frame(self, parent):
        button_frame = tk.Frame(parent, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=(0, 10))
        
        self.find_button = tk.Button(button_frame, text="FIND ROUTES", 
                                     command=self.find_routes,
                                     bg='#2e7d32', fg='white', 
                                     font=('Arial', 11, 'bold'), 
                                     padx=20, pady=5)
        self.find_button.pack(side='left', padx=5)
        
        self.clear_button = tk.Button(button_frame, text="CLEAR MAP", 
                                      command=self.clear_route,
                                      bg='#ef6c00', fg='white', 
                                      font=('Arial', 10), 
                                      padx=15, pady=5)
        self.clear_button.pack(side='left', padx=5)
        
        self.compare_button = tk.Button(button_frame, text="COMPARE ALGORITHMS", 
                                        command=self.compare_algorithms,
                                        bg='#9c27b0', fg='white', 
                                        font=('Arial', 10), 
                                        padx=15, pady=5)
        self.compare_button.pack(side='left', padx=5)
    
    def _build_results_frame(self, parent):
        results_frame = tk.LabelFrame(parent, text="Route Results (Top-K Routes)", 
                                       font=('Arial', 11, 'bold'), 
                                       bg='#f0f0f0', padx=10, pady=10)
        results_frame.pack(fill='both', expand=True)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=14, 
                                                       width=55, font=('Courier', 9))
        self.results_text.pack(fill='both', expand=True)
    
    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="Ready. Select origin and destination to find routes.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                              bd=1, relief='sunken', anchor='w')
        status_bar.pack(side='bottom', fill='x')
    
    def _init_map(self):
        """Initialize map using map_viewer"""
        self.map_widget = self.map_viewer.create_map(self.right_panel)
        self.map_viewer.draw_network()
        self._update_site_lists()
    
    def _show_map_unavailable(self):
        tk.Label(self.right_panel, 
                text="Map visualization unavailable.\n\nPlease install tkintermapview:\npip install tkintermapview",
                font=('Arial', 12), bg='#ffffff', fg='#ff0000').pack(expand=True)
    
    def _update_site_lists(self):
        """Populate site dropdowns from available sites"""
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
    
    def clear_route(self):
        """Clear route from map and results"""
        self.map_viewer.clear_route()
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Route cleared from map.")
    
    def get_user_input(self):
        """Get validated user input"""
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
        
        # Parse time
        try:
            time_str = self.time_var.get()
            hour = int(time_str.split(':')[0]) if ':' in time_str else int(time_str)
            hour = max(0, min(23, hour))
        except:
            hour = 12
        
        return origin, dest, hour
    
    def find_routes(self):
        """Run all 6 algorithms once each and display up to 5 unique routes."""
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

        paths = self.pathfinder.find_unique_paths_all_algorithms(origin, dest, hour, max_paths=5)

        if paths:
            self.map_viewer.draw_route(paths[0][0])
            best_algos = " & ".join(a.upper() for a in paths[0][2])
            self.status_var.set(f"✓ Found {len(paths)} unique route(s). Best: {paths[0][1]:.1f} minutes ({best_algos})")
        else:
            self.status_var.set("No routes found. Try different origin/destination.")

        self._display_results(origin, dest, hour, model_name, paths)
    
    def _display_results(self, origin, dest, hour, model_name, paths):
        """Display formatted results including all unique routes."""
        self.results_text.insert(tk.END, "=" * 65 + "\n")
        self.results_text.insert(tk.END, "TBRGS ROUTE RESULTS\n")
        self.results_text.insert(tk.END, "=" * 65 + "\n\n")

        self.results_text.insert(tk.END, f"Origin:      SCATS {origin}\n")
        self.results_text.insert(tk.END, f"Destination: SCATS {dest}\n")
        self.results_text.insert(tk.END, f"Departure:   {self.time_var.get()} (Hour {hour}:00)\n")
        self.results_text.insert(tk.END, f"ML Model:    {model_name.upper()}\n")
        self.results_text.insert(tk.END, "=" * 65 + "\n\n")
        
        if not paths:
            self.results_text.insert(tk.END, "❌ No routes found!\n\n")
            return

        algo_display = {
            'astar': 'A*', 'bidirectional': 'Bidirectional A*',
            'dijkstra': "Dijkstra's", 'greedy': 'Greedy', 'bfs': 'BFS', 'dfs': 'DFS'
        }

        self.results_text.insert(tk.END, f"✓ Found {len(paths)} unique route(s):\n\n")

        for i, (path, total_time, algos) in enumerate(paths, 1):
            algo_names = " & ".join(algo_display.get(a, a) for a in algos)
            self.results_text.insert(tk.END, f"{'─' * 60}\n")
            self.results_text.insert(tk.END,
                f"ROUTE {i} │ {total_time:.1f} min ({total_time/60:.1f} hrs) │ {algo_names}\n")
            self.results_text.insert(tk.END, f"{'─' * 60}\n")
            
            # Show path with arrows
            path_str = " → ".join(str(n) for n in path)
            if len(path_str) > 70:
                nodes = [str(n) for n in path]
                lines = []
                current_line = []
                curr_len = 0
                for node in nodes:
                    if curr_len + len(node) + 3 > 70:
                        lines.append(" → ".join(current_line))
                        current_line = [node]
                        curr_len = len(node)
                    else:
                        current_line.append(node)
                        curr_len += len(node) + 3
                if current_line:
                    lines.append(" → ".join(current_line))
                for line in lines:
                    self.results_text.insert(tk.END, f"  {line}\n")
            else:
                self.results_text.insert(tk.END, f"  {path_str}\n")
            
            self.results_text.insert(tk.END, f"\n  Total travel time: {total_time:.1f} minutes  [Found by {algo_names}]\n")
        
        self.results_text.insert(tk.END, f"\n{'=' * 65}\n")
    
    def compare_algorithms(self):
        """Compare all algorithms on the current origin/destination"""
        origin, dest, hour = self.get_user_input()
        if origin is None:
            return
        
        model_name = self.current_model.get()
        
        self.status_var.set(f"Comparing all algorithms from {origin} to {dest}...")
        self.root.update()
        
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, "=" * 65 + "\n")
        self.results_text.insert(tk.END, "ALGORITHM COMPARISON\n")
        self.results_text.insert(tk.END, "=" * 65 + "\n\n")
        self.results_text.insert(tk.END, f"Origin: {origin}  →  Destination: {dest}\n")
        self.results_text.insert(tk.END, f"Time: {self.time_var.get()} (Hour {hour}:00)\n")
        self.results_text.insert(tk.END, f"ML Model: {model_name.upper()}\n\n")
        
        self.results_text.insert(tk.END, f"{'Algorithm':<18} {'Time(min)':<12} {'Nodes':<10} {'Success'}\n")
        self.results_text.insert(tk.END, "-" * 55 + "\n")
        
        algorithms = ['astar', 'bidirectional', 'dijkstra', 'greedy', 'bfs', 'dfs']
        algo_names = ['A*', 'Bidirectional A*', 'Dijkstra', 'Greedy', 'BFS', 'DFS']
        
        best_time = float('inf')
        best_algo = None
        
        for algo, name in zip(algorithms, algo_names):
            self.pathfinder.set_algorithm(algo)
            self.pathfinder.set_model(model_name)
            path, cost, nodes = self.pathfinder.find_path(origin, dest, hour)
            
            if path:
                success = "✓"
                self.results_text.insert(tk.END, f"{name:<18} {cost:<12.1f} {nodes:<10} {success}\n")
                if cost < best_time:
                    best_time = cost
                    best_algo = name
            else:
                self.results_text.insert(tk.END, f"{name:<18} {'N/A':<12} {nodes:<10} {'✗'}\n")
        
        self.results_text.insert(tk.END, "-" * 55 + "\n")
        if best_algo:
            self.results_text.insert(tk.END, f"\n🏆 Best algorithm: {best_algo} ({best_time:.1f} minutes)\n")
        
        self.results_text.insert(tk.END, f"\n{'=' * 65}\n")
        self.status_var.set(f"Comparison complete. Best: {best_algo} ({best_time:.1f} min)")