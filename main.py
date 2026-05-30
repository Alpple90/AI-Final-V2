# main.py
import tkinter as tk
import warnings
warnings.filterwarnings('ignore')

from config import MODELS_SAVE_FOLDER
from real_traffic_models import RealTrafficPredictor
from pathfinder import PathFinder
from map_visualization import SCATSMapViewer
from graph_builder import build_graph, get_graph_info
from gui import TBRGSGUI


def _train_models(predictor):
    print("\n    Loading traffic data for training...")
    data = predictor.load_data_from_excel()

    print("    Training LSTM model...")
    predictor.train_lstm(data['X_train_lstm'], data['y_train'],
                         data['X_test_lstm'], data['y_test'], epochs=30, verbose=True)

    print("\n    Training GRU model...")
    predictor.train_gru(data['X_train_lstm'], data['y_train'],
                        data['X_test_lstm'], data['y_test'], epochs=30, verbose=True)

    print("\n    Training XGBoost model...")
    predictor.train_xgboost(data['X_train_xgb'], data['y_train'],
                            data['X_test_xgb'], data['y_test'], verbose=True)

    print("\n    Saving models...")
    predictor.save_models()
    print("    Training complete!")


def main():
    print("\n" + "=" * 60)
    print("TBRGS - Traffic-Based Route Guidance System")
    print("=" * 60 + "\n")

    print("Step 1: Loading map data...")
    map_viewer = SCATSMapViewer()
    coords = map_viewer.load_coordinates()

    print("\nStep 2: Building road network graph...")
    graph = build_graph(map_viewer.get_node_connections(), coords)
    info = get_graph_info(graph)
    print(f"    Nodes: {info['total_nodes']},  Edges: {info['total_edges']},  Isolated: {info['isolated_nodes']}")

    print("\nStep 3: Initializing traffic prediction models...")
    predictor = RealTrafficPredictor()
    print("    Loading pre-trained models...")
    if not predictor.load_models():
        print("    Could not load models. Training new models...")
        _train_models(predictor)

    print("\nStep 4: Initializing pathfinder...")
    pathfinder = PathFinder(graph, predictor, coords)

    print("\nStep 5: Launching GUI...\n")
    root = tk.Tk()
    TBRGSGUI(root, map_viewer, pathfinder)
    root.mainloop()


if __name__ == "__main__":
    main()
