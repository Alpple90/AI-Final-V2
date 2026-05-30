# gui.py - tkinter GUI for TBRGS

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from config import WINDOW_WIDTH, WINDOW_HEIGHT, LEFT_PANEL_WIDTH


class TBRGSGUI:

    def __init__(self, root, map_viewer, pathfinder):
        self.root = root
        self.map_viewer = map_viewer
        self.pathfinder = pathfinder

        self.currentModel = tk.StringVar(value='lstm')
        self.originVar = tk.StringVar()
        self.destVar = tk.StringVar()
        self.time_var = tk.StringVar(value="12:00")

        self.mapWidget = None
        self.resultsText = None
        self.statusVar = None
        self.currentPaths = []

        self.setupWindow()
        self.buildGui()

        if self.map_viewer.mapAvailable():
            self.initMap()
        else:
            self.showMapUnavail()

    # set the window title, size and background colour
    def setupWindow(self):
        self.root.title("TBRGS - Traffic-Based Route Guidance System")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg='#f0f0f0')

    # lay out the left control panel and right map panel
    def buildGui(self):
        leftPanel = tk.Frame(self.root, bg='#f0f0f0', width=LEFT_PANEL_WIDTH)
        leftPanel.pack(side='left', fill='both', expand=False, padx=(10, 5), pady=10)
        leftPanel.pack_propagate(False)

        self.rightPanel = tk.Frame(self.root, bg='#ffffff', bd=2, relief='sunken')
        self.rightPanel.pack(side='right', fill='both', expand=True, padx=(5, 10), pady=10)

        tk.Label(leftPanel, text="TRAFFIC-BASED ROUTE GUIDANCE",
                font=('Arial', 14, 'bold'), bg='#f0f0f0', fg='#1a237e').pack(pady=(0, 5))

        self.statusVar = tk.StringVar(value="")
        tk.Label(leftPanel, textvariable=self.statusVar, font=('Arial', 9),
                bg='#f0f0f0', fg='#2e7d32', wraplength=LEFT_PANEL_WIDTH-20,
                justify='left').pack(fill='x', padx=10, pady=(0, 10))

        self.buildInputFrame(leftPanel)
        self.buildModelFrame(leftPanel)
        self.buildBtnFrame(leftPanel)
        self.buildRouteSelector(leftPanel)
        self.buildResultsFrame(leftPanel)

    # build the origin/destination/time input section
    def buildInputFrame(self, parent):
        inputFrame = tk.LabelFrame(parent, text="Trip Information",
                                    font=('Arial', 11, 'bold'),
                                    bg='#f0f0f0', padx=10, pady=10)
        inputFrame.pack(fill='x', pady=(0, 10))

        tk.Label(inputFrame, text="Origin SCATS:", font=('Arial', 10),
                bg='#f0f0f0').grid(row=0, column=0, sticky='w', pady=5)
        self.originCombo = ttk.Combobox(inputFrame, textvariable=self.originVar, width=20)
        self.originCombo.grid(row=0, column=1, pady=5, padx=(10, 0))
        tk.Button(inputFrame, text="Locate", command=self.locateOrigin,
                 font=('Arial', 8), width=6).grid(row=0, column=2, padx=5)

        tk.Label(inputFrame, text="Destination SCATS:", font=('Arial', 10),
                bg='#f0f0f0').grid(row=1, column=0, sticky='w', pady=5)
        self.destCombo = ttk.Combobox(inputFrame, textvariable=self.destVar, width=20)
        self.destCombo.grid(row=1, column=1, pady=5, padx=(10, 0))
        tk.Button(inputFrame, text="Locate", command=self.locateDest,
                 font=('Arial', 8), width=6).grid(row=1, column=2, padx=5)

        tk.Label(inputFrame, text="Departure Time:", font=('Arial', 10),
                bg='#f0f0f0').grid(row=2, column=0, sticky='w', pady=5)
        tk.Entry(inputFrame, textvariable=self.time_var, width=10,
                font=('Arial', 10)).grid(row=2, column=1, sticky='w', pady=5, padx=(10, 0))
        tk.Label(inputFrame, text="(HH:MM, 24hr)", font=('Arial', 8),
                bg='#f0f0f0').grid(row=2, column=1, sticky='e', padx=(0, 10))

    # add radio buttons for choosing LSTM, GRU or XGBoost
    def buildModelFrame(self, parent):
        modelFrame = tk.LabelFrame(parent, text="ML Model Selection",
                                    font=('Arial', 11, 'bold'),
                                    bg='#f0f0f0', padx=10, pady=10)
        modelFrame.pack(fill='x', pady=(0, 10))

        models = [('LSTM', 'lstm'), ('GRU', 'gru'), ('XGBoost', 'xgboost')]
        for i, (text, value) in enumerate(models):
            tk.Radiobutton(modelFrame, text=text, variable=self.currentModel,
                          value=value, font=('Arial', 10), bg='#f0f0f0').grid(
                          row=0, column=i, padx=20, pady=5)

    # add the find routes, clear map and compare algorithms buttons
    def buildBtnFrame(self, parent):
        btnFrame = tk.Frame(parent, bg='#f0f0f0')
        btnFrame.pack(fill='x', pady=(0, 10))

        row1 = tk.Frame(btnFrame, bg='#f0f0f0')
        row1.pack(fill='x', pady=(0, 5))

        tk.Button(row1, text="FIND ROUTES",
                  command=self.findRoutes,
                  bg='#2e7d32', fg='white',
                  font=('Arial', 11, 'bold'),
                  padx=20, pady=5).pack(side='left', padx=5)

        tk.Button(row1, text="CLEAR MAP",
                  command=self.clearRoute,
                  bg='#ef6c00', fg='white',
                  font=('Arial', 10),
                  padx=15, pady=5).pack(side='left', padx=5)

        row2 = tk.Frame(btnFrame, bg='#f0f0f0')
        row2.pack(fill='x')

        tk.Button(row2, text="COMPARE ALGORITHMS",
                  command=self.compareAlgos,
                  bg='#9c27b0', fg='white',
                  font=('Arial', 10),
                  padx=15, pady=5).pack(side='left', padx=5)

    # create the row of colour-coded route buttons (populated after search)
    def buildRouteSelector(self, parent):
        self.routeSelectorFrame = tk.LabelFrame(parent, text="Display Route",
                                                   font=('Arial', 11, 'bold'),
                                                   bg='#f0f0f0', padx=10, pady=5)
        self.routeSelectorFrame.pack(fill='x', pady=(0, 10))
        self.routeBtnsRow = tk.Frame(self.routeSelectorFrame, bg='#f0f0f0')
        self.routeBtnsRow.pack(fill='x')
        tk.Label(self.routeSelectorFrame, text="Find routes to see options.",
                 font=('Arial', 9), bg='#f0f0f0', fg='#888888').pack()

    # add the scrollable text box where route results are printed
    def buildResultsFrame(self, parent):
        resultsFrame = tk.LabelFrame(parent, text="Route Results (Top-K Routes)",
                                      font=('Arial', 11, 'bold'),
                                      bg='#f0f0f0', padx=10, pady=10)
        resultsFrame.pack(fill='both', expand=True)

        self.resultsText = scrolledtext.ScrolledText(resultsFrame, height=14,
                                                       width=55, font=('Courier', 9))
        self.resultsText.pack(fill='both', expand=True)

    # create the map widget and draw the SCATS network on it
    def initMap(self):
        self.mapWidget = self.map_viewer.createMap(self.rightPanel)
        self.map_viewer.drawNetwork()
        self.updateSiteLists()

    # show an error message in place of the map when tkintermapview isn't installed
    def showMapUnavail(self):
        tk.Label(self.rightPanel,
                text="Map visualization unavailable.\n\nPlease install tkintermapview:\npip install tkintermapview",
                font=('Arial', 12), bg='#ffffff', fg='#ff0000').pack(expand=True)

    # fill both dropdowns with available SCATS sites
    def updateSiteLists(self):
        sites = self.map_viewer.getSites()
        self.originCombo['values'] = sites
        self.destCombo['values'] = sites
        if len(sites) >= 2:
            self.originCombo.set(str(sites[0]))
            self.destCombo.set(str(sites[1]))

    # pan the map to the selected origin site
    def locateOrigin(self):
        site = self.originVar.get()
        if site:
            self.map_viewer.locateSite(site)
            self.statusVar.set(f"Located SCATS {site}")

    # pan the map to the selected destination site
    def locateDest(self):
        site = self.destVar.get()
        if site:
            self.map_viewer.locateSite(site)
            self.statusVar.set(f"Located SCATS {site}")

    # rebuild the route selector buttons to match the latest search results
    def populateRouteBtns(self):
        for w in self.routeBtnsRow.winfo_children():
            w.destroy()
        for w in self.routeSelectorFrame.winfo_children():
            if isinstance(w, tk.Label):
                w.destroy()

        routeColors = ['#ff6f00', '#1565c0', '#6a1b9a', '#00838f', '#558b2f']
        for i, (_, total_time, _) in enumerate(self.currentPaths):
            color = routeColors[i % len(routeColors)]
            btn = tk.Button(self.routeBtnsRow,
                            text=f"Route {i+1}",
                            command=lambda idx=i: self.selectRoute(idx),
                            bg=color, fg='white',
                            font=('Arial', 9, 'bold'),
                            padx=8, pady=3,
                            relief='sunken' if i == 0 else 'raised')
            btn.pack(side='left', padx=3, pady=3)

    # highlight the chosen route on the map and press its button in
    def selectRoute(self, idx):
        routeColors = ['#ff6f00', '#1565c0', '#6a1b9a', '#00838f', '#558b2f']
        path, _, _ = self.currentPaths[idx]
        self.map_viewer.clearRoute()
        self.map_viewer.drawRoute(path, color=routeColors[idx % len(routeColors)], isBest=True)
        for i, btn in enumerate(self.routeBtnsRow.winfo_children()):
            btn.config(relief='sunken' if i == idx else 'raised')

    # wipe the drawn route off the map and clear the results text box
    def clearRoute(self):
        self.map_viewer.clearRoute()
        self.resultsText.delete(1.0, tk.END)
        self.statusVar.set("Route cleared from map.")

    # grab and validate origin, destination and departure hour from the form
    def getUserInput(self):
        originStr = self.originVar.get()
        destStr = self.destVar.get()

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

    # kick off a route search using all 6 algorithms and show the top results
    def findRoutes(self):
        origin, dest, hour = self.getUserInput()
        if origin is None:
            return

        modelName = self.currentModel.get()

        self.statusVar.set(f"Finding routes from {origin} to {dest} using all algorithms "
                            f"with {modelName.upper()}...")
        self.root.update()

        self.resultsText.delete(1.0, tk.END)
        self.map_viewer.clearRoute()

        self.pathfinder.setModel(modelName)

        paths = self.pathfinder.findUniquePaths(origin, dest, hour, maxPaths=5)

        self.currentPaths = paths
        if paths:
            bestAlgos = " & ".join(a.upper() for a in paths[0][2])
            self.statusVar.set(f"Found {len(paths)} unique route(s). Best: {paths[0][1]:.1f} minutes ({bestAlgos})")
        else:
            self.statusVar.set("No routes found. Try different origin/destination.")
        self.populateRouteBtns()
        if paths:
            self.selectRoute(0)

        self.displayResults(origin, dest, hour, modelName, paths)

    # format and print all found routes into the scrollable results box
    def displayResults(self, origin, dest, hour, modelName, paths):
        SEP = "=" * 40
        self.resultsText.insert(tk.END, SEP + "\n")
        self.resultsText.insert(tk.END, "TBRGS ROUTE RESULTS\n")
        self.resultsText.insert(tk.END, SEP + "\n")
        self.resultsText.insert(tk.END, f"Origin:    SCATS {origin}\n")
        self.resultsText.insert(tk.END, f"Dest:      SCATS {dest}\n")
        self.resultsText.insert(tk.END, f"Departure: {self.time_var.get()} (Hour {hour}:00)\n")
        self.resultsText.insert(tk.END, f"ML Model:  {modelName.upper()}\n")
        self.resultsText.insert(tk.END, SEP + "\n\n")

        if not paths:
            self.resultsText.insert(tk.END, "No routes found!\n\n")
            return

        algoDisplay = {
            'astar': 'A*', 'bidirectional': 'Bidirectional A*',
            'dijkstra': "Dijkstra's", 'greedy': 'Greedy', 'bfs': 'BFS', 'dfs': 'DFS'
        }

        self.resultsText.insert(tk.END, f"Found {len(paths)} unique route(s):\n\n")

        for i, (path, total_time, algos) in enumerate(paths, 1):
            algoNames = " & ".join(algoDisplay.get(a, a) for a in algos)
            self.resultsText.insert(tk.END, "─" * 40 + "\n")
            self.resultsText.insert(tk.END, f"ROUTE {i} │ {total_time:.1f} min\n")
            self.resultsText.insert(tk.END, f"By: {algoNames}\n\n")

            # wrap node list so it fits the text box
            nodes = [str(n) for n in path]
            currLine = []
            currLen = 0
            for node in nodes:
                if currLen + len(node) + 4 > 40:
                    self.resultsText.insert(tk.END, "  " + " → ".join(currLine) + "\n")
                    currLine = [node]
                    currLen = len(node)
                else:
                    currLine.append(node)
                    currLen += len(node) + 4
            if currLine:
                self.resultsText.insert(tk.END, "  " + " → ".join(currLine) + "\n")

            self.resultsText.insert(tk.END, "\n")

        self.resultsText.insert(tk.END, SEP + "\n")

    # run all 6 algorithms on the same trip and print a side-by-side comparison
    def compareAlgos(self):
        origin, dest, hour = self.getUserInput()
        if origin is None:
            return

        modelName = self.currentModel.get()

        self.statusVar.set(f"Comparing all algorithms from {origin} to {dest}...")
        self.root.update()

        self.resultsText.delete(1.0, tk.END)

        SEP = "=" * 40
        self.resultsText.insert(tk.END, SEP + "\n")
        self.resultsText.insert(tk.END, "ALGORITHM COMPARISON\n")
        self.resultsText.insert(tk.END, SEP + "\n")
        self.resultsText.insert(tk.END, f"Origin: {origin}  Dest: {dest}\n")
        self.resultsText.insert(tk.END, f"Time: {self.time_var.get()}  Model: {modelName.upper()}\n\n")

        self.resultsText.insert(tk.END, f"{'Algorithm':<16} {'min':<8} {'Nodes':<7} \n")
        self.resultsText.insert(tk.END, "-" * 40 + "\n")

        algorithms = ['astar', 'bidirectional', 'dijkstra', 'greedy', 'bfs', 'dfs']
        algoNames = ['A*', 'Bidir A*', 'Dijkstra', 'Greedy', 'BFS', 'DFS']

        bestTime = float('inf')
        bestAlgo = None

        for algo, name in zip(algorithms, algoNames):
            self.pathfinder.setAlgorithm(algo)
            self.pathfinder.setModel(modelName)
            path, cost, nodes = self.pathfinder.findPath(origin, dest, hour)

            if path:
                self.resultsText.insert(tk.END, f"{name:<16} {cost:<8.1f} {nodes:<7} Y\n")
                if cost < bestTime:
                    bestTime = cost
                    bestAlgo = name
            else:
                self.resultsText.insert(tk.END, f"{name:<16} {'N/A':<8} {nodes:<7} N\n")

        self.resultsText.insert(tk.END, "-" * 40 + "\n")
        if bestAlgo:
            self.resultsText.insert(tk.END, f"\nBest: {bestAlgo} ({bestTime:.1f} min)\n")

        self.resultsText.insert(tk.END, "\n" + SEP + "\n")
        self.statusVar.set(f"Comparison complete. Best: {bestAlgo} ({bestTime:.1f} min)")
