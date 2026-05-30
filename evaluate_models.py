# evaluate_models.py - compare LSTM, GRU and XGBoost on held-out test data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from real_traffic_models import RealTrafficPredictor
from pathfinder import PathFinder
from graph_builder import build_graph, get_graph_info
from map_visualization import SCATSMapViewer


# load saved models, run predictions, print a comparison table and save a plot
def run_evaluation():
    print("--- ML Model Comparison - TBRGS ---")

    predictor = RealTrafficPredictor()
    data = predictor.load_data()

    print("--- Loading saved models ---")
    loaded = predictor.load_models()
    if not loaded:
        print("No saved models found. Run real_traffic_models.py first to train.")
        return

    x_test_lstm = data['X_test_lstm']
    x_test_xgb  = data['X_test_xgb']
    y_test      = data['y_test']

    results = {}
    model_configs = [
        ('LSTM',    'lstm',    x_test_lstm),
        ('GRU',     'gru',     x_test_lstm),
        ('XGBoost', 'xgboost', x_test_xgb),
    ]

    print("--- Generating predictions ---")
    for display_name, model_key, x_test in model_configs:
        if model_key not in predictor.models:
            print(f"  {display_name}: model not found, skipping")
            continue

        model = predictor.models[model_key]
        if model_key in ['lstm', 'gru']:
            y_pred = model.predict(x_test, verbose=0).flatten()
        else:
            y_pred = model.predict(x_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        results[display_name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'preds': y_pred}
        print(f"  {display_name} done")

    if not results:
        print("No model results to display.")
        return

    print(f"\n{'Model':<12} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 40)
    for name, metrics in results.items():
        print(f"{name:<12} {metrics['mae']:>8.2f} {metrics['rmse']:>8.2f} {metrics['r2']:>8.4f}")
    print("-" * 40)

    best_name = min(results, key=lambda n: results[n]['mae'])
    print(f"Best model: {best_name} (lowest MAE)")

    # plot predicted vs actual for all models
    print("--- Saving comparison plot ---")
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]

    # only plot first 500 samples so the chart stays readable
    plot_limit = 500
    x_axis = np.arange(plot_limit)

    for ax, (name, metrics) in zip(axes, results.items()):
        y_actual = y_test[:plot_limit]
        y_pred_plot = metrics['preds'][:plot_limit]
        ax.plot(x_axis, y_actual, label='Actual', alpha=0.7)
        ax.plot(x_axis, y_pred_plot, label='Predicted', alpha=0.7)
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
    best_metrics = results[best_name]
    print(f"  Best overall model: {best_name}")
    print(f"  MAE={best_metrics['mae']:.2f}, RMSE={best_metrics['rmse']:.2f}, R2={best_metrics['r2']:.4f}")
    other_names = [n for n in results if n != best_name]
    for other in other_names:
        mae_diff = results[other]['mae'] - best_metrics['mae']
        print(f"  {best_name} beats {other} by {mae_diff:.2f} MAE units")

    run_route_agreement(predictor)


# check whether each ML model recommends the same best route for a set of O/D pairs
def run_route_agreement(predictor):
    print("--- Route Agreement ---")
    print("Checking if LSTM, GRU and XGBoost recommend the same route\n")

    map_viewer = SCATSMapViewer()
    coords = map_viewer.load_coords()
    graph = build_graph(map_viewer.get_node_connections(), coords)
    pathfinder = PathFinder(graph, predictor, coords)

    test_pairs = [
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

    agree_count = 0
    total_pairs = 0

    for origin, dest in test_pairs:
        routes = {}
        for model_name in models:
            pathfinder.set_model(model_name)
            pathfinder.set_algorithm('astar')
            path, cost, _ = pathfinder.find_path(origin, dest, hour)
            routes[model_name] = tuple(path) if path else None

        unique_routes = set(r for r in routes.values() if r is not None)
        all_agree = len(unique_routes) == 1

        if all_agree:
            agree_count += 1
        total_pairs += 1

        status = "AGREE" if all_agree else "DIFFER"
        print(f"  {origin} -> {dest}: {status}")
        if not all_agree:
            for model_name, route in routes.items():
                route_str = ' -> '.join(str(n) for n in route) if route else 'No path'
                print(f"    {model_name.upper()}: {route_str}")

    agree_pct = (agree_count / total_pairs) * 100 if total_pairs > 0 else 0
    print(f"\n  Agreement rate: {agree_count}/{total_pairs} pairs ({agree_pct:.0f}%)")
    if agree_pct == 100:
        print("  All models recommend the same route for every pair.")
    elif agree_pct >= 75:
        print("  Models mostly agree — ML model choice has minimal routing impact.")
    else:
        print("  Models disagree on several routes — ML model choice affects routing.")


if __name__ == '__main__':
    run_evaluation()
