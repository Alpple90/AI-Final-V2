# main.py - entry point for TBRGS
import tkinter as tk
import warnings
warnings.filterwarnings('ignore')

from config import MODELS_SAVE_FOLDER
from real_traffic_models import RealTrafficPredictor
from pathfinder import PathFinder
from map_visualization import SCATSMapViewer
from graph_builder import buildGraph, getGraphInfo
from gui import TBRGSGUI


# train all three models on the SCATS data and save them to disk
def trainModels(predictor):
    print("--- Loading traffic data for training ---")
    data = predictor.loadData()

    print("--- Training LSTM model ---")
    predictor.trainLSTM(data['X_train_lstm'], data['y_train'],
                        data['X_test_lstm'], data['y_test'], epochs=30, verbose=True)

    print("--- Training GRU model ---")
    predictor.trainGRU(data['X_train_lstm'], data['y_train'],
                       data['X_test_lstm'], data['y_test'], epochs=30, verbose=True)

    print("--- Training XGBoost model ---")
    predictor.trainXGB(data['X_train_xgb'], data['y_train'],
                       data['X_test_xgb'], data['y_test'], verbose=True)

    print("--- Saving models ---")
    predictor.saveModels()
    print("--- Training complete ---")


# boot up the whole app: load data, build the graph, init models, launch the GUI
def main():
    print("--- Loading map data ---")
    mapViewer = SCATSMapViewer()
    coords = mapViewer.loadCoords()

    print("--- Building road network graph ---")
    graph = buildGraph(mapViewer.getNodeConnections(), coords)
    info = getGraphInfo(graph)
    print(f"    Nodes: {info['total_nodes']},  Edges: {info['total_edges']},  Isolated: {info['isolated_nodes']}")

    print("--- Initializing traffic prediction models ---")
    predictor = RealTrafficPredictor()
    if not predictor.loadModels():
        print("--- No saved models found, training new ones ---")
        trainModels(predictor)

    print("--- Initializing pathfinder ---")
    pathfinder = PathFinder(graph, predictor, coords)

    print("--- Launching GUI ---")
    root = tk.Tk()
    TBRGSGUI(root, mapViewer, pathfinder)
    root.mainloop()


if __name__ == "__main__":
    main()
