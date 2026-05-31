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
pip install -r requirements.txt
```

The following data files must be present in the project root:

- `TrafficDataCopy.xlsx` — traffic volume data (October 2006, provided via Canvas)
- `scatsTrueLongLat.xlsx` — corrected latitude/longitude for each SCATS site

### Training the Models

If the `saved_models/` folder is empty, the models must be trained before the application can run. Execute the following command from the project root:

```
python real_traffic_models.py
```

Trained models are saved to the `saved_models/` folder. Training all three models may take several minutes on CPU. To train a single model only:

```
python real_traffic_models.py --model lstm
python real_traffic_models.py --model gru
python real_traffic_models.py --model xgboost
```

### Running the Application

```
python main.py
```

This loads the saved models, builds the road graph, and opens the GUI.

### Using the GUI

1. Select an **Origin** and **Destination** SCATS site from the dropdowns (or type a site number directly).
2. Select the **hour of day** to use for traffic prediction (00:00 – 23:59).
3. Select the **day of the week** for traffic prediction.
4. Choose an **ML model** (LSTM, GRU, or XGBoost).
5. Click **Find Routes** — up to 5 routes are returned, sorted by estimated travel time.
6. Click any route button to highlight the corresponding route on the map.
7. Click **Compare Algorithms** to see a side-by-side comparison of all six search algorithms.

### Running the Tests

```
python tests.py
```

### Running Model Evaluation

```
python evaluate_models.py
```

Outputs a metric comparison table, saves `model_comparison.png`, and prints route recommendation agreement results.

---

## 2. Introduction

The Traffic-Based Route Guidance System (TBRGS) is an application that combines machine learning (ML) traffic prediction with graph-based search algorithms to recommend optimal driving routes through the Boroondara area of south-eastern Melbourne.

The road network is represented as a weighted directed graph of 39 SCATS (Sydney Coordinated Adaptive Traffic System) intersections. The edge weight between two adjacent intersections is not a fixed distance — it is a predicted travel time that varies based on the traffic volume forecast produced by the selected ML model.

**Traffic prediction** is performed by three ML models trained on one month of 15-minute interval vehicle count data (October 2006):

- **LSTM (Long Short-Term Memory)** — a recurrent neural network well suited to sequential data, capable of capturing long-range temporal dependencies through its gating mechanisms.
- **GRU (Gated Recurrent Unit)** — a streamlined variant of LSTM with fewer parameters, offering comparable predictive accuracy with faster training times.
- **XGBoost** — a gradient-boosted decision tree ensemble that treats traffic prediction as a tabular regression problem, using flattened sequence windows as input features.

Each model is trained on the first three weeks of October (1–24) and evaluated on the final week (25–31). Input windows consist of the 12 most recent 15-minute readings (3 hours) for a given site, encoded with cyclical features for time of day (period 96) and day of week (period 7) using sine and cosine transformations.

**Route search** is handled by six algorithms carried over from Assignment 2A: BFS, DFS, Greedy Best-First, A\*, Dijkstra, and Bidirectional A\*. Yen's K-shortest paths algorithm is applied on top of each search method so that each can propose multiple distinct routes. All candidate routes from all six algorithms are pooled, deduplicated, and the top 5 by estimated travel time are returned to the user.

---

## 3. Features / Bugs / Missing

### Implemented Features

- **Data processing:** Reads `TrafficDataCopy.xlsx`, extracts 15-minute volume columns for each site and date, melts to long format, applies cyclical encoding, and builds sliding sequence windows (length = 12 intervals = 3 hours).
- **LSTM model:** Two-layer LSTM (64 units each) with dropout, trained with early stopping and learning rate reduction.
- **GRU model:** Two-layer GRU (64 units each) with dropout, same training configuration as LSTM.
- **XGBoost model:** Trained on the same sequence windows as flat feature vectors.
- **Travel time model:** Implements the quadratic flow-speed relationship from the Traffic Flow to Travel Time Conversion document. Assumes a 60 km/h speed limit and a 30-second intersection delay per node.
- **Graph construction:** Builds a weighted directed graph from `NODE_CONNECTIONS` using Haversine distance. Edges outside the 0.1–10 km range are filtered.
- **Six search algorithms:** BFS, DFS, Greedy, A\*, Dijkstra, Bidirectional A\* — all returning a (path, cost, nodes\_explored) tuple.
- **Yen's K-shortest paths:** Each algorithm runs Yen's spur method to find up to 5 candidate routes; results are pooled across all six algorithms, deduplicated, and sorted by cost.
- **GUI:** Tkinter interface with site dropdowns, hour and day-of-week selectors, model selector, route buttons, result panel, and an interactive map.
- **Map visualisation:** TkinterMapView with CartoCDN tiles displays the road network and highlights selected routes.
- **Model evaluation:** Computes MAE, RMSE, and R² for each model on the held-out test set, prints a comparison table, and saves `model_comparison.png`.
- **Route agreement analysis:** Runs A\* with each ML model across 21 origin-destination pairs and reports the proportion of pairs where all three models recommend the same route.
- **18 automated unit tests** covering graph building, travel time, search algorithms, path validity, and route uniqueness.
- **Configuration file (`config.py`):** Centralises speed limit, intersection delay, road capacity, quadratic coefficients, and GUI dimensions.

### Known Bugs

- Bidirectional A\* occasionally finds a suboptimal path on small graphs due to meeting-point cost estimation. This has not been observed on the full 39-node graph.
- `findUniquePaths` can be slow when all six algorithms each run Yen's with k=5 on longer routes. On the 39-node graph, response time is typically under two seconds.

### Missing / Not Implemented

- No features are missing relative to the base requirements. The 30-second intersection delay is included and the speed limit is fixed at 60 km/h as specified.

---

## 4. Testing

Testing was carried out at two levels: automated unit testing of the software modules, and empirical evaluation of the ML models.

### 4.1 Unit Tests (`tests.py`)

A total of **20 test cases** were written using Python's `unittest` framework. A lightweight 5-node mock graph and a `MockPredictor` that always returns 100 vehicles per 15 minutes are used in place of the real graph and ML models, so tests run without requiring the data files or trained models.

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
| 9 | TestPathFinderBasic | test\_greedy\_findsPath | Greedy finds a path between connected nodes |
| 10 | TestPathFinderBasic | test\_bidirectionalAstar\_findsPath | Bidirectional A\* finds a path between connected nodes |
| 11 | TestPathFinderBasic | test\_sameOriginDest\_returnNoneOrZero | Same origin and destination returns None or cost 0 |
| 12 | TestPathFinderEdgeCases | test\_invalidNode\_returnsNone | Unknown destination node returns None path |
| 13 | TestPathFinderEdgeCases | test\_pathCost\_greaterThanZero | Valid path has a positive cost |
| 14 | TestPathFinderEdgeCases | test\_path\_isListOfInts | Returned path is a list of integers |
| 15 | TestFindUniquePaths | test\_findUniquePaths\_atMostFive | `findUniquePaths` returns ≤ 5 routes |
| 16 | TestFindUniquePaths | test\_findUniquePaths\_sortedByCost | Routes are sorted ascending by travel time |
| 17 | TestFindUniquePaths | test\_findUniquePaths\_noDuplicates | No two returned routes are identical |
| 18 | TestRouteValidity | test\_path\_starts\_at\_origin\_ends\_at\_dest | Path begins at origin and ends at destination |
| 19 | TestRouteValidity | test\_path\_has\_no\_cycles | Path contains no repeated nodes |
| 20 | TestRouteValidity | test\_path\_edges\_exist\_in\_graph | Every consecutive node pair is connected in the graph |

**Result: 20/20 tests passed.**

*Figure 1 — Unit test output showing all 20 tests passing.*

![Test output showing 20 tests OK]

### 4.2 ML Model Evaluation (`evaluate_models.py`)

All three models were evaluated on the held-out test split (October 25–31) of the SCATS dataset. Figure 2 shows the first 500 samples of the test set for visual comparison; MAE, RMSE, and R² are computed across the full test set.

**Table 2 — ML Model Performance Metrics**

| Model   | MAE   | RMSE  | R²     |
|---------|-------|-------|--------|
| LSTM    | 32.76 | 50.84 | 0.9660 |
| GRU     | 32.64 | 50.66 | 0.9662 |
| XGBoost | 31.00 | 49.55 | 0.9677 |

*Figure 2 — Predicted vs actual traffic volume for the first 500 test samples. MAE and R² are calculated over the full test set.*

![model_comparison.png]

All three models achieve R² > 0.96, demonstrating a strong fit to the traffic volume time series. XGBoost achieves the lowest MAE (31.00) and highest R² (0.9677), outperforming LSTM by 1.76 MAE units and GRU by 1.63 MAE units.

### 4.3 Route Recommendation Agreement

To verify that the choice of ML model does not introduce inconsistency in route recommendations, A\* was run with each of the three models across 21 origin-destination pairs at hour 8 (morning peak).

**Result: 21/21 pairs (100%) — all three models recommended the same route for every pair.**

This test was also run using the other heuristic-based search algorithms with the same outcome. This confirms that at the Boroondara network scale, all three models produce sufficiently similar flow predictions that the ordering of edge costs — and therefore the optimal path — is unchanged regardless of which model is selected.

---

## 5. Insights

### 5.1 LSTM vs GRU

LSTM and GRU produced nearly identical results (MAE 32.76 vs 32.64, R² 0.9660 vs 0.9662). This is consistent with the literature — GRU was designed as a simplified LSTM, and on datasets of this size and regularity the additional gating in LSTM provides no measurable advantage. GRU trains slightly faster due to its lower parameter count, making it the more practical choice when compute time is a concern.

### 5.2 XGBoost Performance

XGBoost outperformed both recurrent models on every metric. This is a well-known result for structured time series data at moderate scale — gradient boosted trees handle feature interactions efficiently and do not require the careful tuning of sequence length, learning rate, and dropout that neural networks demand. The October 2006 dataset has strong weekly and daily periodicity, which is naturally captured through the lag features in each window, giving XGBoost a straightforward signal to exploit.

XGBoost does have a key limitation: it treats each prediction independently and cannot generalise across unseen traffic patterns the way a recurrent network can. It would likely degrade more sharply on data from a different month or under unusual conditions such as events or road closures. LSTM and GRU are theoretically more robust to such distributional shifts.

### 5.3 Search Algorithm Behaviour

Each of the six search algorithms finds routes with different trade-offs:

- **BFS** — guarantees the fewest hops but does not account for edge weights, so the fewest-hop route is not necessarily the fastest.
- **DFS** — explores deep paths quickly and is useful for detecting whether a path exists, but tends to produce longer routes due to its LIFO expansion order.
- **Greedy Best-First** — moves toward the goal quickly using the haversine heuristic but can miss globally cheaper routes.
- **A\*** — balances actual accumulated cost and heuristic estimate, consistently finding the lowest-cost path.
- **Dijkstra** — always finds the globally optimal path but explores more nodes than A\* since it has no heuristic.
- **Bidirectional A\*** — searches simultaneously from both endpoints, effective on larger graphs but adds implementation complexity.

In the pooled Yen's approach, differences between algorithms are minimised — all routes from all six algorithms are merged and re-ranked by cost, so even if DFS or Greedy produces a suboptimal spur path, it is pushed down the list by cheaper routes found by A\* and Dijkstra.

### 5.4 Route Diversity

On the 39-node Boroondara graph there are typically 3–5 structurally distinct routes between any two distant sites. Pooling all six algorithms' Yen outputs consistently surfaces the full set of meaningful alternatives, whereas running a single algorithm in isolation — especially DFS or Greedy — would occasionally miss a cheaper option.

---

## 6. Research

### 6.1 Map Visualisation with OpenStreetMap

The application includes an interactive map built on TkinterMapView, using CartoCDN's Voyager tile layer — a styled OpenStreetMap derivative. Each SCATS node is rendered as a coloured marker and the road network edges are drawn as grey path segments. When a route is selected, the path is drawn in a distinct colour overlaid on the network.

The latitude and longitude values provided in the SCATS dataset did not align precisely with the actual intersection locations visible on OpenStreetMap. A corrected coordinates file (`scatsTrueLongLat.xlsx`) was prepared by manually identifying the true location of each intersection, allowing markers to appear at the correct map positions.

### 6.2 Travel Time Conversion

The quadratic flow-speed model from the specification document was implemented in `travel_time.py`. The model uses two regimes:

- **Free flow** (flow ≤ 351 veh/hr): speed = 60 km/h
- **Congested** (flow > 351 veh/hr): speed is derived from the quadratic relationship
  `V = -1.4648375·S² + 93.75·S`
  where V is flow (veh/hr) and S is speed (km/h).

Travel time for an edge of length d is:

`t = (d / S) × 60 + 0.5 minutes`

where 0.5 minutes accounts for the 30-second intersection delay. This produces realistic travel times that increase non-linearly as traffic volume approaches and exceeds road capacity.

---

## 7. Conclusion

The TBRGS successfully integrates machine learning traffic prediction with graph-based route search to produce a functional end-to-end system for the Boroondara network. All three ML models achieve high predictive accuracy (R² > 0.96) and produce identical routing recommendations across all tested origin-destination pairs, confirming that route guidance is robust to the choice of model.

XGBoost is the best-performing model by MAE and RMSE, benefiting from the strong periodicity of the October 2006 dataset. For deployment on a live or multi-year dataset, LSTM or GRU would be preferable due to their capacity to generalise across shifting traffic patterns without full retraining.

A\* and Dijkstra are the most reliable search algorithms for cost-optimal routing. The pooled Yen's approach adds significant value by aggregating diverse candidate routes from all six algorithms, consistently delivering up to 5 distinct alternatives ranked by predicted travel time.

Possible improvements include:

- Training on a larger, multi-year dataset to improve generalisation across seasons and special events.
- Extending the network beyond Boroondara using the full VicRoads SCATS dataset.
- Replacing the static travel time model with real-time VicRoads API data.
- Adding turn restrictions and traffic signal timing to the edge cost model.

---

## 8. Acknowledgements / Resources

| Resource | How it was used |
|----------|----------------|
| VicRoads SCATS dataset (`TrafficDataCopy.xlsx`) | Source of all traffic volume data for model training and evaluation |
| `scatsTrueLongLat.xlsx` (provided via Canvas) | Corrected latitude/longitude for all 39 SCATS sites used to place markers on the map |
| Traffic Flow to Travel Time Conversion v1.0 (Canvas document) | Provided the quadratic flow-speed relationship and intersection delay used in `travel_time.py` |
| TensorFlow / Keras documentation (tensorflow.org) | Reference for LSTM and GRU layer configuration, EarlyStopping, ReduceLROnPlateau, and model serialisation |
| XGBoost documentation (xgboost.readthedocs.io) | Reference for `XGBRegressor` parameters |
| Scikit-learn documentation (scikit-learn.org) | Used for `MinMaxScaler`, MAE, RMSE, and R² metric functions |
| TkinterMapView (github.com/TomSchimansky/TkinterMapView) | Provided the interactive map widget embedded in the GUI |
| Yen, J.Y. (1971) | Original paper describing Yen's K-shortest paths algorithm used in `pathfinder.py` |
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
