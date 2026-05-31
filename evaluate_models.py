# evaluate_models.py - compare LSTM, GRU and XGBoost on held-out test data
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from real_traffic_models import RealTrafficPredictor
from pathfinder import PathFinder
from graph_builder import buildGraph
from map_visualization import SCATSMapViewer


# load saved models, run predictions, print a comparison table and save a plot
def runEvaluation():
    print("--- ML Model Comparison - TBRGS ---")

    predictor = RealTrafficPredictor()
    data = predictor.loadData()

    print("--- Loading saved models ---")
    loaded = predictor.loadModels()
    if not loaded:
        print("No saved models found. Run real_traffic_models.py first to train.")
        return

    yTest = data['y_test']

    # Build test meta from prediction cache keys so we can evaluate per-window
    # For each test window we need (scatsNum, dayOfWeek, hour) — derive from cache
    results = {}
    modelKeys = [('LSTM', 'lstm'), ('GRU', 'gru'), ('XGBoost', 'xgboost')]

    # Inverse-transform y_test for evaluation
    if predictor.scaler is not None:
        yTestActual = predictor.scaler.inverse_transform(
            np.array(yTest).reshape(-1, 1)
        ).flatten()
    else:
        yTestActual = np.array(yTest, dtype=np.float32)

    # Rebuild test windows to get meta (site, day, hour) for each sample
    if os.path.exists('data_test_reference.csv'):
        refDf = pd.read_csv('data_test_reference.csv')
        _, metaDf = predictor._buildTestWindows(refDf)
        testScats = metaDf['SCATS Number'].tolist()
        testDays  = metaDf['Day of week'].tolist()
        testHours = [int(str(t)[:2]) for t in metaDf['Time'].tolist()]
    else:
        print("data_test_reference.csv not found — run loadData() first")
        return

    print("--- Generating predictions ---")
    for displayName, modelKey in modelKeys:
        if modelKey not in predictor.models:
            print(f"  {displayName}: model not found, skipping")
            continue

        nSamples = min(len(yTestActual), len(testScats))
        yPred = np.array([
            predictor.predict(modelKey, testScats[i], testHours[i], testDays[i])
            for i in range(nSamples)
        ], dtype=np.float32)
        yTestEval = yTestActual[:nSamples]

        mae  = mean_absolute_error(yTestEval, yPred)
        rmse = np.sqrt(mean_squared_error(yTestEval, yPred))
        r2   = r2_score(yTestEval, yPred)

        results[displayName] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'preds': yPred}
        print(f"  {displayName} done")

    if not results:
        print("No model results to display.")
        return

    print(f"\n{'Model':<12} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 40)
    for name, metrics in results.items():
        print(f"{name:<12} {metrics['mae']:>8.2f} {metrics['rmse']:>8.2f} {metrics['r2']:>8.4f}")
    print("-" * 40)

    bestName = min(results, key=lambda n: results[n]['mae'])
    print(f"Best model: {bestName} (lowest MAE)")

    # plot predicted vs actual for all models
    print("--- Saving comparison plot ---")
    numModels = len(results)
    fig, axes = plt.subplots(1, numModels, figsize=(6 * numModels, 5))
    if numModels == 1:
        axes = [axes]

    # only plot first 500 samples so the chart stays readable
    plotLimit = 500
    xAxis = np.arange(plotLimit)

    for ax, (name, metrics) in zip(axes, results.items()):
        yActual = yTestActual[:plotLimit]
        yPredPlot = metrics['preds'][:plotLimit]
        ax.plot(xAxis, yActual, label='Actual', alpha=0.7)
        ax.plot(xAxis, yPredPlot, label='Predicted', alpha=0.7)
        ax.set_title(f"{name}\nMAE={metrics['mae']:.2f}  R²={metrics['r2']:.4f}")
        ax.set_xlabel('Sample')
        ax.set_ylabel('Traffic volume (vehicles/15min)')
        ax.legend()

    plt.suptitle('Predicted vs Actual Traffic Volume', fontsize=14)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.close()
    print("  Plot saved to model_comparison.png")

    print("--- Summary ---")
    bestMetrics = results[bestName]
    print(f"  Best overall model: {bestName}")
    print(f"  MAE={bestMetrics['mae']:.2f}, RMSE={bestMetrics['rmse']:.2f}, R2={bestMetrics['r2']:.4f}")
    otherNames = [n for n in results if n != bestName]
    for other in otherNames:
        maeDiff = results[other]['mae'] - bestMetrics['mae']
        print(f"  {bestName} beats {other} by {maeDiff:.2f} MAE units")

    runRouteAgreement(predictor)


# check whether each ML model recommends the same best route for a set of O/D pairs
def runRouteAgreement(predictor):
    print("--- Route Agreement ---")
    print("Checking if LSTM, GRU and XGBoost recommend the same route\n")

    mapViewer = SCATSMapViewer()
    coords = mapViewer.loadCoords()
    graph = buildGraph(mapViewer.getNodeConnections(), coords)
    pathfinder = PathFinder(graph, predictor, coords)

    testPairs = [
        (970,  2000),
        (970,  4040),
        (3001, 4812),
        (2820, 4063),
        (3180, 3682),
        (4821, 2200),
        (4030, 3120),
        (2827, 4264),
        (3662, 4043),
        (4812, 4821),
        (2820, 3682),
        (3122, 3180),
        (4035, 3812),
        (2846, 4821),
        (4272, 4321),
        (970,  2825),
        (4273, 970),
        (3682, 4063),
        (3180, 3120),
        (4335, 3804),
        (4034, 2827),
    ]

    models = ['lstm', 'gru', 'xgboost']
    hour = 8  # morning peak

    agreeCount = 0
    totalPairs = 0

    for origin, dest in testPairs:
        routes = {}
        for modelName in models:
            pathfinder.setModel(modelName)
            pathfinder.setAlgorithm('astar')
            path, cost, _ = pathfinder.findPath(origin, dest, hour)
            routes[modelName] = tuple(path) if path else None

        uniqueRoutes = set(r for r in routes.values() if r is not None)
        allAgree = len(uniqueRoutes) == 1

        if allAgree:
            agreeCount += 1
        totalPairs += 1

        status = "AGREE" if allAgree else "DIFFER"
        print(f"  {origin} -> {dest}: {status}")
        if not allAgree:
            for modelName, route in routes.items():
                routeStr = ' -> '.join(str(n) for n in route) if route else 'No path'
                print(f"    {modelName.upper()}: {routeStr}")

    agreePct = (agreeCount / totalPairs) * 100 if totalPairs > 0 else 0
    print(f"\n  Agreement rate: {agreeCount}/{totalPairs} pairs ({agreePct:.0f}%)")
    if agreePct == 100:
        print("  All models recommend the same route for every pair.")
    elif agreePct >= 75:
        print("  Models mostly agree — ML model choice has minimal routing impact.")
    else:
        print("  Models disagree on several routes — ML model choice affects routing.")


if __name__ == '__main__':
    runEvaluation()
