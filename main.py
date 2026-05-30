# main.py - entry point for TBRGS
import tkinter as tk
import warnings
warnings.filterwarnings('ignore')

from config import MODELS_SAVE_FOLDER
from real_traffic_models import RealTrafficPredictor
from pathfinder import PathFinder
from map_visualization import SCATSMapViewer
from graph_builder import build_graph, get_graph_info
from gui import TBRGSGUI


# train all three models on the SCATS data and save them to disk
def train_models(predictor):
    print("--- Loading traffic data for training ---")
    data = predictor.load_data()

    print("--- Training LSTM model ---")
    predictor.train_lstm(data['X_train_lstm'], data['y_train'],
                        data['X_test_lstm'], data['y_test'], epochs=30, verbose=True)

    print("--- Training GRU model ---")
    predictor.train_gru(data['X_train_lstm'], data['y_train'],
                       data['X_test_lstm'], data['y_test'], epochs=30, verbose=True)

    print("--- Training XGBoost model ---")
    predictor.train_xgb(data['X_train_xgb'], data['y_train'],
                       data['X_test_xgb'], data['y_test'], verbose=True)

    print("--- Saving models ---")
    predictor.save_models()
    print("--- Training complete ---")


# boot up the whole app: load data, build the graph, init models, launch the GUI
def main():
    print("--- Loading map data ---")
    map_viewer = SCATSMapViewer()
    coords = map_viewer.load_coords()

    print("--- Building road network graph ---")
    graph = build_graph(map_viewer.get_node_connections(), coords)
    info = get_graph_info(graph)
    print(f"    Nodes: {info['total_nodes']},  Edges: {info['total_edges']},  Isolated: {info['isolated_nodes']}")

    print("--- Initializing traffic prediction models ---")
    predictor = RealTrafficPredictor()
    if not predictor.load_models():
        print("--- No saved models found, training new ones ---")
        train_models(predictor)

    print("--- Initializing pathfinder ---")
    pathfinder = PathFinder(graph, predictor, coords)

    print("--- Launching GUI ---")
    root = tk.Tk()
    TBRGSGUI(root, map_viewer, pathfinder)
    root.mainloop()


if __name__ == "__main__":
    main()
