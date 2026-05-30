# main.py
"""
TBRGS - Traffic-Based Route Guidance System
Main entry point - Run this file

COS30019 - Introduction to AI
Assignment 2B

Usage: python main.py
"""

import tkinter as tk
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from config import MODELS_SAVE_FOLDER
from travel_time import TravelTimeCalculator
from real_traffic_models import RealTrafficPredictor
from pathfinder import PathFinder
from map_visualization import SCATSMapViewer
from graph_builder import GraphBuilder
from gui import TBRGSGUI


class TBRGSSystem:
    """
    Main system coordinator - initializes all components
    """
    
    def __init__(self):
        print("\n" + "=" * 60)
        print("TBRGS - Traffic-Based Route Guidance System")
        print("COS30019 - Introduction to AI")
        print("Assignment 2B")
        print("=" * 60 + "\n")
        
        # Step 1: Initialize map and load coordinates
        print("Step 1: Loading map data...")
        self.map_viewer = SCATSMapViewer()
        self.coords = self.map_viewer.load_coordinates()
        
        # Step 2: Build graph from connections
        print("\nStep 2: Building road network graph...")
        connections = self.map_viewer.get_node_connections()
        self.graph = GraphBuilder.build_graph(connections, self.coords)
        graph_info = GraphBuilder.get_graph_info(self.graph)
        print(f"    Nodes: {graph_info['total_nodes']}")
        print(f"    Edges: {graph_info['total_edges']}")
        print(f"    Isolated nodes: {graph_info['isolated_nodes']}")
        
        # Step 3: Initialize travel time calculator
        print("\nStep 3: Initializing travel time calculator...")
        self.tt_calc = TravelTimeCalculator()
        
        # Step 4: Initialize traffic predictor
        print("\nStep 4: Initializing traffic prediction models...")
        self.predictor = RealTrafficPredictor()
        
        # Check if models exist, train if needed
        # if os.path.exists(f'{MODELS_SAVE_FOLDER}/lstm_model.h5'):
        #     print("    Loading pre-trained models...")
        #     self.predictor.load_models()
        # else:
        #     print("    No pre-trained models found. Training new models...")
        #     self._train_models()
        
        # FORCE LOADING - no retraining
        print("    Loading pre-trained models...")
        success = self.predictor.load_models()
        if not success:
            print("    Could not load models. Training new models...")
            self._train_models()
                
        # Step 5: Initialize pathfinder
        print("\nStep 5: Initializing pathfinder...")
        self.pathfinder = PathFinder(self.graph, self.tt_calc, self.predictor)
        
        # Step 6: Launch GUI
        print("\nStep 6: Launching GUI...\n")
        self._launch_gui()
    
    def _train_models(self):
        """Train all three ML models on actual traffic data"""
        print("\n    Loading traffic data for training...")
        data = self.predictor.load_data_from_excel()
        
        print("    Training LSTM model...")
        self.predictor.train_lstm(
            data['X_train_lstm'], data['y_train'],
            data['X_test_lstm'], data['y_test'],
            epochs=30, verbose=True
        )
        
        print("\n    Training GRU model...")
        self.predictor.train_gru(
            data['X_train_lstm'], data['y_train'],
            data['X_test_lstm'], data['y_test'],
            epochs=30, verbose=True
        )
        
        print("\n    Training XGBoost model...")
        self.predictor.train_xgboost(
            data['X_train_xgb'], data['y_train'],
            data['X_test_xgb'], data['y_test'],
            verbose=True
        )
        
        print("\n    Saving models...")
        self.predictor.save_models()
        print("    Training complete!")
    
    def _launch_gui(self):
        """Launch the Tkinter GUI"""
        root = tk.Tk()
        app = TBRGSGUI(root, self.map_viewer, self.pathfinder)
        root.mainloop()


def main():
    system = TBRGSSystem()


if __name__ == "__main__":
    main()