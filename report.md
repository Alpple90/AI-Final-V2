# COS30019 – Introduction to AI
## Assignment 2B – Traffic-Based Route Guidance System (TBRGS)

---

## Cover Page

**Unit:** COS30019 Introduction to AI
**Assignment:** Assignment 2 – Part B
**Semester:** Semester 1, 2026

| Full Name | Student ID | Signature |
|-----------|-----------|-----------|
| [Name 1]  | [ID 1]    |           |
| [Name 2]  | [ID 2]    |           |
| [Name 3]  | [ID 3]    |           |
| [Name 4]  | [ID 4]    |           |

### Statement of Contribution

| Team Member | Contributions |
|-------------|--------------|
| [Name 1] | Data processing (`real_traffic_models.py`), LSTM and GRU model implementation, model training pipeline |
| [Name 2] | XGBoost implementation, model evaluation (`evaluate_models.py`), route agreement analysis |
| [Name 3] | Search algorithm integration (`pathfinder.py`), travel time model (`travel_time.py`), Yen's K-shortest paths |
| [Name 4] | GUI (`gui.py`), map visualisation (`map_visualization.py`), unit testing (`tests.py`), report |

---

## Table of Contents

1. Instructions
2. Introduction
3. Features / Bugs / Missing
4. Testing
5. Insights
6. Research
7. Conclusion
8. Acknowledgements / Resources
9. References

---

## 1. Instructions

### Requirements

Install dependencies before running:

```
pip install tensorflow xgboost scikit-learn pandas openpyxl xlrd
pip install tkintermapview requests pillow
```

The following data files must be present in the project root:

- `Scats Data October 2006.xls` — traffic volume data (provided by VicRoads)
- `scatsTrueLongLat.xlsx` — corrected latitude/longitude for each SCATS site

### Training the Models

Run the training script to produce saved models (only required once):

```
python real_traffic_models.py
```

Trained models are saved to the `saved_models/` folder. Training takes several minutes on CPU.

### Running the Application

```
python main.py
```

This loads saved models, builds the road graph, and opens the GUI. If no saved models are found, a fallback predictor is used with a fixed flow estimate.

### Using the GUI

1. Select an **Origin** and **Destination** SCATS site from the dropdowns (or type a site number).
2. Select the **hour of day** to use for traffic prediction (0–23).
3. Choose an **ML model** (LSTM, GRU, or XGBoost).
4. Click **Find Routes** — up to 5 routes are returned, sorted by estimated travel time.
5. Click any route button to highlight that route on the map.
6. Click **Compare Algorithms** to see a side-by-side comparison of all six search algorithms.

### Running the Tests

```
python tests.py
```

### Running Model Evaluation

```
python evaluate_models.py
```

Outputs a metric comparison table, saves `model_comparison.png`, and prints route agreement results.

---

## 2. Introduction

The Traffic-Based Route Guidance System (TBRGS) is an end-to-end application that combines machine learning traffic prediction with graph-based search to recommend optimal driving routes through the Boroondara area of Melbourne.

The Boroondara road network is represented as a weighted directed graph of 39 SCATS (Sydney Coordinated Adaptive Traffic System) intersections. The edge weight between two adjacent intersections is not a fixed distance — it is a predicted travel time that changes based on the traffic volume forecast produced by a trained ML model.

**Traffic prediction** is performed by three ML models trained on one month of 15-minute interval vehicle count data from VicRoads (October 2006):

- **LSTM (Long Short-Term Memory)** — a recurrent neural network architecture well suited to time series data due to its ability to retain long-range dependencies through gating mechanisms.
- **GRU (Gated Recurrent Unit)** — a lighter variant of LSTM with fewer parameters, offering comparable accuracy with faster training.
- **XGBoost** — a gradient-boosted decision tree ensemble that treats traffic prediction as a tabular regression problem using lag features derived from the same sequence windows.

**Route search** is handled by six algorithms carried over from Assignment 2A: BFS, DFS, Greedy Best-First, A\*, Dijkstra, and Bidirectional A\*. Yen's K-shortest paths algorithm is applied on top of each search method so that each algorithm can propose multiple distinct routes. All candidate routes from all six algorithms are pooled, deduplicated, and the top 5 by predicted travel time are returned to the user.

---

## 3. Features / Bugs / Missing

### Implemented Features

- **Data processing:** Reads `Scats Data October 2006.xls`, extracts 15-minute volume columns (V00–V95) for each site and date, flattens into a single time series, applies sliding window sequencing (window length = 12 intervals = 3 hours).
- **LSTM model:** Two-layer LSTM with dropout, trained with early stopping on 80/20 train-test split.
- **GRU model:** Two-layer GRU with dropout, same training setup as LSTM.
- **XGBoost model:** Trained on the same sequence windows as flat feature vectors; no temporal structure assumed.
- **Travel time model:** Implements the quadratic flow-speed relationship from the Traffic Flow to Travel Time Conversion document. Assumes 60 km/h speed limit and a 30-second intersection delay per node.
- **Graph construction:** Builds a weighted graph from `NODE_CONNECTIONS` using Haversine distance. Edges outside the 0.1–10 km range are filtered.
- **Six search algorithms:** BFS, DFS, Greedy, A\*, Dijkstra, Bidirectional A\* — all returning (path, cost, nodes\_explored).
- **Yen's K-shortest paths:** Each algorithm runs Yen's spur method to find up to 5 candidate routes; all results are pooled across algorithms, deduplicated, and sorted by cost.
- **GUI:** Tkinter interface with site dropdowns, hour selector, model selector, route buttons, result panel, and interactive map.
- **Map visualisation:** TkinterMapView with OpenStreetMap tiles displays the road network and highlights selected routes.
- **Model evaluation:** Computes MAE, RMSE, and R² for each model on the held-out test set, prints a comparison table, and saves a plot.
- **Route agreement analysis:** Runs A\* with each ML model across 21 origin-destination pairs and reports how often all three models recommend the same route.
- **18 automated unit tests** covering graph building, travel time, all four search algorithms, path validity, and route uniqueness.
- **Configuration file (`config.py`):** Centralises speed limit, intersection delay, capacity, quadratic coefficients, and GUI dimensions.

### Known Bugs

- Bidirectional A\* occasionally finds a suboptimal path on small graphs due to the meeting-point cost estimation. On the full 39-node graph this has not been observed.
- The `findUniquePaths` call can be slow when all 6 algorithms each run Yen's with k=5, particularly for long routes. On the 39-node graph, response time is typically under 2 seconds.

### Missing / Not Implemented

- No features are missing relative to the base requirements. The 30-second intersection delay is included. The speed limit is fixed at 60 km/h as specified.

---

## 4. Testing

Testing was carried out at two levels: automated unit testing of the software modules, and empirical evaluation of the ML models.

### 4.1 Unit Tests (`tests.py`)

A total of **18 test cases** were written using Python's `unittest` framework. A lightweight 5-node mock graph (nodes 970, 3685, 2000, 2846, 4043) is used so tests run without requiring the Excel data files or trained models. A `MockPredictor` that always returns 100 vehicles per 15 minutes is used in place of the real ML models.

**Table 1 — Unit Test Summary**

| # | Class | Test | Verifies |
|---|-------|------|---------|
| 1 | TestBuildGraph | test\_buildGraph\_returnsNonEmpty | `buildGraph` returns a non-empty dict |
| 2 | TestBuildGraph | test\_haversineDistance\_positive | Haversine distance is > 0 for two known coords |
| 3 | TestBuildGraph | test\_calcTravelTime\_positive | Travel time is positive for normal inputs |
| 4 | TestBuildGraph | test\_calcTravelTime\_higherFlowSlower | Higher traffic flow produces a longer travel time |
| 5 | TestPathFinderBasic | test\_astar\_findsPath | A\* finds a path between connected nodes |
| 6 | TestPathFinderBasic | test\_bfs\_findsPath | BFS finds a path between connected nodes |
| 7 | TestPathFinderBasic | test\_dfs\_findsPath | DFS finds a path between connected nodes |
| 8 | TestPathFinderBasic | test\_dijkstra\_findsPath | Dijkstra finds a path between connected nodes |
| 9 | TestPathFinderBasic | test\_sameOriginDest\_returnNoneOrZero | Same origin and destination returns None or cost 0 |
| 10 | TestPathFinderEdgeCases | test\_invalidNode\_returnsNone | Unknown destination node returns None path |
| 11 | TestPathFinderEdgeCases | test\_pathCost\_greaterThanZero | Valid path has a positive cost |
| 12 | TestPathFinderEdgeCases | test\_path\_isListOfInts | Returned path is a list of integers |
| 13 | TestFindUniquePaths | test\_findUniquePaths\_atMostFive | `findUniquePaths` returns ≤ 5 routes |
| 14 | TestFindUniquePaths | test\_findUniquePaths\_sortedByCost | Routes are sorted ascending by travel time |
| 15 | TestFindUniquePaths | test\_findUniquePaths\_noDuplicates | No two returned routes are identical |
| 16 | TestRouteValidity | test\_path\_starts\_at\_origin\_ends\_at\_dest | Path begins at origin node and ends at destination node |
| 17 | TestRouteValidity | test\_path\_has\_no\_cycles | Path contains no repeated nodes |
| 18 | TestRouteValidity | test\_path\_edges\_exist\_in\_graph | Every consecutive pair of nodes in the path is connected in the graph |

**Result: 18/18 tests passed.**

*Figure 1 — Unit test output showing all 18 tests passing.*

![Test output showing 18 tests OK]

### 4.2 ML Model Evaluation (`evaluate_models.py`)

All three models were evaluated on the held-out 20% test split of the October 2006 SCATS dataset. The test set covers approximately 60,000 15-minute volume readings aggregated across all sites.

**Table 2 — ML Model Performance Metrics**

| Model   | MAE   | RMSE  | R²     |
|---------|-------|-------|--------|
| LSTM    | 11.17 | 16.19 | 0.9492 |
| GRU     | 11.18 | 16.19 | 0.9492 |
| XGBoost | 10.22 | 14.90 | 0.9570 |

*Figure 2 — Predicted vs Actual traffic volume for 500 test samples (LSTM, GRU, XGBoost).*

![model_comparison.png]

All three models achieve R² > 0.94, demonstrating a strong fit to the traffic volume time series. XGBoost achieves the lowest MAE (10.22) and highest R² (0.9570), outperforming LSTM by 0.95 MAE units and GRU by 0.96 MAE units.

### 4.3 Route Agreement

To verify that the choice of ML model does not introduce inconsistency in routing recommendations, A\* was run with each of the three models across 21 origin-destination pairs at hour 8 (morning peak).

**Result: 21/21 pairs (100%) — all models recommended the same route.**

This indicates that at the Boroondara network scale, all three models produce sufficiently similar flow predictions that the travel time ordering of edges is unchanged, leading to identical optimal paths.

---

## 5. Insights

### 5.1 LSTM vs GRU

LSTM and GRU produced nearly identical results (MAE 11.17 vs 11.18, R² 0.9492 vs 0.9492). This is consistent with the literature — GRU was designed as a simplified LSTM, and on datasets of this size and regularity, the additional gates in LSTM do not provide a measurable advantage. GRU trains faster due to fewer parameters, making it the preferred choice when computational cost matters.

### 5.2 XGBoost Performance

XGBoost outperformed both recurrent models on every metric. This is a well-known result for structured time series data at moderate scale — gradient boosted trees handle feature interactions efficiently and do not require the careful tuning of sequence length, learning rate, and dropout that neural models demand. The October 2006 dataset has a strong weekly and daily periodicity which is naturally captured through lag features, giving XGBoost a straightforward signal to exploit.

However, XGBoost has a key limitation: it treats each prediction independently. It cannot generalise across unseen traffic patterns in the way a recurrent network can, and would likely degrade more sharply on data from a different month or under unusual conditions (events, road closures). LSTM and GRU are theoretically more robust to such distributional shifts.

### 5.3 Search Algorithm Behaviour

Each of the six search algorithms finds routes with different trade-offs:

- **BFS** guarantees the fewest hops (not lowest travel time) and is predictable, but does not account for edge weights.
- **DFS** explores deep paths quickly and is suitable for detecting whether a path exists, but produces longer routes by nature of its LIFO expansion.
- **Greedy Best-First** moves toward the goal quickly using the haversine heuristic but can miss globally cheaper routes.
- **A\*** balances actual cost and heuristic estimate, consistently finding the lowest-cost path.
- **Dijkstra** always finds the globally optimal path but explores more nodes than A\* since it has no heuristic.
- **Bidirectional A\*** searches from both ends simultaneously, which is effective on larger graphs but adds implementation complexity.

In the pooled Yen's approach, the differences between algorithms are minimised — all routes from all algorithms are merged and re-ranked by cost, so even if DFS finds a suboptimal spur path, it is pushed down the list by the cheaper routes found by A\* and Dijkstra.

### 5.4 Route Diversity

On the 39-node Boroondara graph, there are typically 3–5 structurally distinct routes between any two distant nodes. Pooling all six algorithms' Yen outputs consistently identifies the full set of meaningful routes, whereas running a single algorithm in isolation (especially DFS or Greedy) would occasionally miss a cheaper alternative.

---

## 6. Research

### 6.1 Map Visualisation with OpenStreetMap

The system includes an interactive map built on TkinterMapView, using CartoCDN's Voyager tile layer (a styled OpenStreetMap derivative). Each SCATS node is rendered as a coloured marker and the road network edges are drawn as grey path segments. When a route is selected, the path is drawn in a distinct colour over the network.

The latitude/longitude data provided in the SCATS dataset did not align precisely with the real intersection locations on OpenStreetMap. A corrected coordinates file (`scatsTrueLongLat.xlsx`) was prepared by cross-referencing each intersection name with its actual coordinates, allowing the markers to appear at the correct map positions.

### 6.2 Travel Time Conversion

The quadratic flow-speed model from the specification document was implemented in `travel_time.py`. The model uses two regimes:

- **Free flow** (flow ≤ 351 veh/hr): speed = 60 km/h
- **Congested** (flow > 351 veh/hr): speed derived from the quadratic relationship
  `V = -1.4648375·S² + 93.75·S`
  where V is flow (veh/hr) and S is speed (km/h).

Travel time for an edge of length d is then:

`t = (d / S) × 60 + 0.5 minutes`

where 0.5 minutes is the 30-second intersection delay. This model produces realistic travel times that increase non-linearly as traffic volume approaches and exceeds road capacity (1500 veh/hr).

---

## 7. Conclusion

The TBRGS successfully integrates machine learning traffic prediction with graph-based route search to produce a functional end-to-end system for the Boroondara network. All three ML models achieve high predictive accuracy (R² > 0.94) and — at this network scale — produce identical routing recommendations, confirming that the route guidance component is robust to model choice.

XGBoost is the best-performing model by MAE and RMSE, benefiting from the strong periodicity of the October 2006 dataset. For deployment on a live or multi-year dataset, LSTM or GRU would be preferable due to their ability to generalise across shifting traffic patterns without retraining.

A\* and Dijkstra are the most reliable search algorithms for cost-optimal routing. The pooled Yen's approach adds significant value by aggregating diverse candidate routes from all six algorithms, consistently delivering 5 distinct alternatives to the user.

Possible improvements include:

- Training on a larger, multi-year dataset to improve generalisation.
- Adding time-of-day and day-of-week as explicit input features to the ML models.
- Extending the network beyond Boroondara using the full VicRoads SCATS dataset.
- Replacing the static travel time model with real-time VicRoads API data.
- Adding turn restrictions and traffic signal timing to the edge cost model.

---

## 8. Acknowledgements / Resources

| Resource | How it was used |
|----------|----------------|
| VicRoads SCATS dataset (`Scats Data October 2006.xls`) | Source of all traffic volume data for model training and evaluation |
| `scatsTrueLongLat.xlsx` (provided via Canvas) | Corrected latitude/longitude for all 39 SCATS sites used to build the map |
| Traffic Flow to Travel Time Conversion v1.0 (Canvas document) | Provided the quadratic flow-speed relationship and intersection delay assumption used in `travel_time.py` |
| TensorFlow / Keras documentation (tensorflow.org) | Reference for LSTM and GRU layer configuration, EarlyStopping callback, and model serialisation |
| XGBoost documentation (xgboost.readthedocs.io) | Reference for `XGBRegressor` parameters and feature importance API |
| Scikit-learn documentation (scikit-learn.org) | Used for `StandardScaler`, `train_test_split`, MAE, RMSE, and R² metric functions |
| TkinterMapView (github.com/TomSchimansky/TkinterMapView) | Provided the interactive map widget embedded in the GUI |
| Yen, J.Y. (1971). "Finding the K Shortest Loopless Paths in a Network." *Management Science*, 17(11), 712–716. | Original paper describing Yen's K-shortest paths algorithm used in `pathfinder.py` |
| Haversine formula (Wikipedia) | Used to compute great-circle distances between SCATS site coordinates in `graph_builder.py` |

---

## 9. References

1. Yen, J. Y. (1971). Finding the K Shortest Loopless Paths in a Network. *Management Science*, 17(11), 712–716.

2. Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *arXiv:1406.1078*.

3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

5. TensorFlow Developers. (2024). TensorFlow (v2.x). Retrieved from https://www.tensorflow.org

6. XGBoost Contributors. (2024). XGBoost Documentation. Retrieved from https://xgboost.readthedocs.io

7. VicRoads. (2006). SCATS Traffic Volume Data — Boroondara, October 2006. Provided via Swinburne University Canvas.
