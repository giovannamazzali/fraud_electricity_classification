from numpy import array, ndarray
from pandas import read_csv, DataFrame, concat
from sklearn.model_selection import train_test_split
from dslabs.dslabs_functions import plot_multibar_chart, read_train_test_from_files, CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart, plot_evaluation_results
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from matplotlib.pyplot import figure, savefig, show


# --- Settings ---
file_tag = "client"
index_col = "client_id"
target = "target"
train_size = 0.7

# --- Load data ---
data: DataFrame = read_csv("client_pos_selected.csv", index_col=index_col)
labels: list = list(map(int, data[target].unique()))
labels.sort()
print(f"Labels={labels}")

# --- Summary of class distribution ---
positive: int = 1
negative: int = 0
values: dict[str, list[int]] = {
    "Original": [
        len(data[data[target] == negative]),
        len(data[data[target] == positive]),
    ]
}

# --- Split X and y ---
# y: array = data.pop(target).to_list()
# X: ndarray = data.values

# # --- Stratified Hold-Out Split ---
# trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=train_size, stratify=y, random_state=42)

# # --- Recombine and save ---
# train: DataFrame = concat(
#     [DataFrame(trnX, columns=data.columns), DataFrame(trnY, columns=[target])],
#     axis=1
# )
# train.to_csv(f"data/{file_tag}_train.csv", index_label=index_col)

# test: DataFrame = concat(
#     [DataFrame(tstX, columns=data.columns), DataFrame(tstY, columns=[target])],
#     axis=1
# )
# test.to_csv(f"data/{file_tag}_test.csv", index_label=index_col)

# # --- Class balance report ---
# values["Train"] = [
#     len(train[train[target] == negative]),
#     len(train[train[target] == positive]),
# ]
# values["Test"] = [
#     len(test[test[target] == negative]),
#     len(test[test[target] == positive]),
# ]

# --- Plot distribution ---
# figure(figsize=(6, 4))
# plot_multibar_chart(labels, values, title="Data distribution per dataset")
# show()

# ---------------------------------------
# Naive Bayes Classifier

file_tag = "client"
train_filename = "data/client_train.csv"
test_filename = "data/client_test.csv"
target = "target"
eval_metric = "accuracy"


trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
    train_filename, test_filename, target
)
print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")

# def naive_Bayes_study(
#     trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, metric: str = "accuracy"
# ) -> tuple:
#     estimators: dict = {
#         "GaussianNB": GaussianNB(),
#         "MultinomialNB": MultinomialNB(),
#         "BernoulliNB": BernoulliNB(),
#     }

#     xvalues: list = []
#     yvalues: list = []
#     best_model = None
#     best_params: dict = {"name": "", "metric": metric, "params": ()}
#     best_performance = 0
#     for clf in estimators:
#         xvalues.append(clf)
#         estimators[clf].fit(trnX, trnY)
#         prdY: array = estimators[clf].predict(tstX)
#         eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
#         if eval - best_performance > DELTA_IMPROVE:
#             best_performance: float = eval
#             best_params["name"] = clf
#             best_params[metric] = eval
#             best_model = estimators[clf]
#         yvalues.append(eval)
#         # print(f'NB {clf}')
#     plot_bar_chart(
#         xvalues,
#         yvalues,
#         title=f"Naive Bayes Models ({metric})",
#         ylabel=metric,
#         percentage=True,
#     )

#     return best_model, best_params


# figure()
# best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, eval_metric)
# savefig(f"images/{file_tag}_nb_{eval_metric}_study.png")
# show()

# figure()
# best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, "recall")
# savefig(f"images/{file_tag}_nb_recall_study.png")
# show()

# prd_trn: array = best_model.predict(trnX)
# prd_tst: array = best_model.predict(tstX)
# figure()
# plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
# savefig(f'images/{file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')
# show()

# KNN Classifier

# from typing import Literal
# from numpy import array, ndarray
# from sklearn.neighbors import KNeighborsClassifier
# from matplotlib.pyplot import figure, savefig, show
# from dslabs.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_multiline_chart
# from dslabs.dslabs_functions import read_train_test_from_files, plot_evaluation_results

# def knn_study(
#         trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, k_max: int=19, lag: int=2, metric='accuracy'
#         ) -> tuple[KNeighborsClassifier | None, dict]:
#     dist: list[Literal['manhattan', 'euclidean', 'chebyshev']] = ['manhattan', 'euclidean', 'chebyshev']

#     kvalues: list[int] = [i for i in range(1, k_max+1, lag)]
#     best_model: KNeighborsClassifier | None = None
#     best_params: dict = {'name': 'KNN', 'metric': metric, 'params': ()}
#     best_performance: float = 0.0

#     values: dict[str, list] = {}
#     for d in dist:
#         y_tst_values: list = []
#         for k in kvalues:
#             clf = KNeighborsClassifier(n_neighbors=k, metric=d)
#             clf.fit(trnX, trnY)
#             prdY: array = clf.predict(tstX)
#             eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
#             y_tst_values.append(eval)
#             if eval - best_performance > DELTA_IMPROVE:
#                 best_performance: float = eval
#                 best_params['params'] = (k, d)
#                 best_model = clf
#             # print(f'KNN {d} k={k}')
#         values[d] = y_tst_values
#     print(f'KNN best with k={best_params['params'][0]} and {best_params['params'][1]}')
#     plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric})', xlabel='k', ylabel=metric, percentage=True)

#     return best_model, best_params


# eval_metric = 'accuracy'

# trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(train_filename, test_filename, target)
# print(f'Train#={len(trnX)} Test#={len(tstX)}')
# print(f'Labels={labels}')

# figure()
# best_model, params = knn_study(trnX, trnY, tstX, tstY, k_max=25, metric=eval_metric)
# savefig(f'images/{file_tag}_knn_{eval_metric}_study.png')

# prd_trn: array = best_model.predict(trnX)
# prd_tst: array = best_model.predict(tstX)
# figure()
# plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
# savefig(f'images/{file_tag}_knn_{params["name"]}_best_{params["metric"]}_eval.png')
# show()

# from matplotlib.pyplot import figure, savefig

# distance: Literal["manhattan", "euclidean", "chebyshev"] = params["params"][1]
# K_MAX = 25
# kvalues: list[int] = [i for i in range(1, K_MAX, 2)]
# y_tst_values: list = []
# y_trn_values: list = []
# acc_metric: str = "accuracy"
# for k in kvalues:
#     clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
#     clf.fit(trnX, trnY)
#     prd_tst_Y: array = clf.predict(tstX)
#     prd_trn_Y: array = clf.predict(trnX)
#     y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
#     y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

# figure()
# plot_multiline_chart(
#     kvalues,
#     {"Train": y_trn_values, "Test": y_tst_values},
#     title=f"KNN overfitting study for {distance}",
#     xlabel="K",
#     ylabel=str(eval_metric),
#     percentage=True,
# )
# savefig(f"images/{file_tag}_knn_overfitting.png")
# show()

#------------------------------------------------------
# Decision Tree Classifier

# from typing import Literal
# from numpy import array, ndarray
# from matplotlib.pyplot import figure, savefig, show
# from sklearn.tree import DecisionTreeClassifier
# from dslabs.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, read_train_test_from_files
# from dslabs.dslabs_functions import plot_evaluation_results, plot_multiline_chart


# def trees_study(
#         trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, d_max: int=10, lag:int=2, metric='accuracy'
#         ) -> tuple:
#     criteria: list[Literal['entropy', 'gini']] = ['entropy', 'gini']
#     depths: list[int] = [i for i in range(2, d_max+1, lag)]

#     best_model: DecisionTreeClassifier | None = None
#     best_params: dict = {'name': 'DT', 'metric': metric, 'params': ()}
#     best_performance: float = 0.0

#     values: dict = {}
#     for c in criteria:
#         y_tst_values: list[float] = []
#         for d in depths:
#             clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=0)
#             clf.fit(trnX, trnY)
#             prdY: array = clf.predict(tstX)
#             eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
#             y_tst_values.append(eval)
#             if eval - best_performance > DELTA_IMPROVE:
#                 best_performance = eval
#                 best_params['params'] = (c, d)
#                 best_model = clf
#             # print(f'DT {c} and d={d}')
#         values[c] = y_tst_values
#     print(f'DT best with {best_params['params'][0]} and d={best_params['params'][1]}')
#     plot_multiline_chart(depths, values, title=f'DT Models ({metric})', xlabel='d', ylabel=metric, percentage=True)

#     return best_model, best_params


# trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(train_filename, test_filename, target)
# print(f'Train#={len(trnX)} Test#={len(tstX)}')
# print(f'Labels={labels}')

# figure()
# best_model, params = trees_study(trnX, trnY, tstX, tstY, d_max=25, metric=eval_metric)
# savefig(f'images/{file_tag}_dt_{eval_metric}_study.png')
# show()

# prd_trn: array = best_model.predict(trnX)
# prd_tst: array = best_model.predict(tstX)
# figure()
# plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
# savefig(f'images/{file_tag}_dt_{params["name"]}_best_{params["metric"]}_eval.png')
# show()

# from sklearn.tree import export_graphviz
# from matplotlib.pyplot import imread, imshow, axis
# from subprocess import call

# tree_filename: str = f"images/{file_tag}_dt_{eval_metric}_best_tree"
# max_depth2show = 3
# st_labels: list[str] = [str(value) for value in labels]


# from sklearn.tree import plot_tree

# figure(figsize=(14, 6))
# plot_tree(
#     best_model,
#     max_depth=max_depth2show,
#     feature_names=vars,
#     class_names=st_labels,
#     filled=True,
#     rounded=True,
#     impurity=False,
#     precision=2,
# )
# savefig(tree_filename + ".png")

# from numpy import argsort
# from dslabs.dslabs_functions import plot_horizontal_bar_chart

# importances = best_model.feature_importances_
# indices: list[int] = argsort(importances)[::-1]
# elems: list[str] = []
# imp_values: list[float] = []
# for f in range(len(vars)):
#     elems += [vars[indices[f]]]
#     imp_values += [importances[indices[f]]]
#     print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

# figure()
# plot_horizontal_bar_chart(
#     elems,
#     imp_values,
#     title="Decision Tree variables importance",
#     xlabel="importance",
#     ylabel="variables",
#     percentage=True,
# )
# savefig(f"images/{file_tag}_dt_{eval_metric}_vars_ranking.png")

# crit: Literal["entropy", "gini"] = params["params"][0]
# d_max = 25
# depths: list[int] = [i for i in range(2, d_max + 1, 1)]
# y_tst_values: list[float] = []
# y_trn_values: list[float] = []
# acc_metric = "accuracy"
# for d in depths:
#     clf = DecisionTreeClassifier(max_depth=d, criterion=crit, min_impurity_decrease=0)
#     clf.fit(trnX, trnY)
#     prd_tst_Y: array = clf.predict(tstX)
#     prd_trn_Y: array = clf.predict(trnX)
#     y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
#     y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

# figure()
# plot_multiline_chart(
#     depths,
#     {"Train": y_trn_values, "Test": y_tst_values},
#     title=f"DT overfitting study for {crit}",
#     xlabel="max_depth",
#     ylabel=str(eval_metric),
#     percentage=True,
# )
# savefig(f"images/{file_tag}_dt_{eval_metric}_overfitting.png")

# Multilayer Perceptron Classifier

from typing import Literal
from numpy import array, ndarray
from matplotlib.pyplot import subplots, figure, savefig, show
from sklearn.neural_network import MLPClassifier
from dslabs.dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from dslabs.dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart

# LAG: int = 500
# NR_MAX_ITER: int = 5000


# def mlp_study(
#     trnX: ndarray,
#     trnY: array,
#     tstX: ndarray,
#     tstY: array,
#     nr_max_iterations: int = 2500,
#     lag: int = 500,
#     metric: str = "accuracy",
# ) -> tuple[MLPClassifier | None, dict]:
#     nr_iterations: list[int] = [lag] + [
#         i for i in range(2 * lag, nr_max_iterations + 1, lag)
#     ]

#     lr_types: list[Literal["constant", "invscaling", "adaptive"]] = [
#         "constant",
#         "invscaling",
#         "adaptive",
#     ]  # only used if optimizer='sgd'
#     learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]

#     best_model: MLPClassifier | None = None
#     best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
#     best_performance: float = 0.0

#     values: dict = {}
#     _, axs = subplots(
#         1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False
#     )
#     for i in range(len(lr_types)):
#         type: str = lr_types[i]
#         values = {}
#         for lr in learning_rates:
#             warm_start: bool = False
#             y_tst_values: list[float] = []
#             for j in range(len(nr_iterations)):
#                 clf = MLPClassifier(
#                     learning_rate=type,
#                     learning_rate_init=lr,
#                     max_iter=lag,
#                     warm_start=warm_start,
#                     activation="logistic",
#                     solver="sgd",
#                     verbose=False,
#                 )
#                 clf.fit(trnX, trnY)
#                 prdY: array = clf.predict(tstX)
#                 eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
#                 y_tst_values.append(eval)
#                 warm_start = True
#                 if eval - best_performance > DELTA_IMPROVE:
#                     best_performance = eval
#                     best_params["params"] = (type, lr, nr_iterations[j])
#                     best_model = clf
#                 # print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
#             values[lr] = y_tst_values
#         plot_multiline_chart(
#             nr_iterations,
#             values,
#             ax=axs[0, i],
#             title=f"MLP with {type}",
#             xlabel="nr iterations",
#             ylabel=metric,
#             percentage=True,
#         )
#     print(
#         f'MLP best for {best_params["params"][2]} iterations (lr_type={best_params["params"][0]} and lr={best_params["params"][1]}'
#     )

#     return best_model, best_params


# figure()
# best_model, params = mlp_study(
#     trnX,
#     trnY,
#     tstX,
#     tstY,
#     nr_max_iterations=NR_MAX_ITER,
#     lag=LAG,
#     metric=eval_metric,
# )
# savefig(f"images/{file_tag}_mlp_{eval_metric}_study.png")
# show()

# prd_trn: array = best_model.predict(trnX)
# prd_tst: array = best_model.predict(tstX)
# figure()
# plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
# savefig(f'images/{file_tag}_mlp_{params["name"]}_best_{params["metric"]}_eval.png')
# show()

# lr_type: Literal["constant", "invscaling", "adaptive"] = params["params"][0]
# lr: float = params["params"][1]

# nr_iterations: list[int] = [LAG] + [i for i in range(2 * LAG, NR_MAX_ITER + 1, LAG)]

# y_tst_values: list[float] = []
# y_trn_values: list[float] = []
# acc_metric = "accuracy"

# warm_start: bool = False
# for n in nr_iterations:
#     clf = MLPClassifier(
#         warm_start=warm_start,
#         learning_rate=lr_type,
#         learning_rate_init=lr,
#         max_iter=n,
#         activation="logistic",
#         solver="sgd",
#         verbose=False,
#     )
#     clf.fit(trnX, trnY)
#     prd_tst_Y: array = clf.predict(tstX)
#     prd_trn_Y: array = clf.predict(trnX)
#     y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
#     y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))
#     warm_start = True

# figure()
# plot_multiline_chart(
#     nr_iterations,
#     {"Train": y_trn_values, "Test": y_tst_values},
#     title=f"MLP overfitting study for lr_type={lr_type} and lr={lr}",
#     xlabel="nr_iterations",
#     ylabel=str(eval_metric),
#     percentage=True,
# )
# savefig(f"images/{file_tag}_mlp_{eval_metric}_overfitting.png")

# from numpy import arange
# from dslabs.dslabs_functions import plot_line_chart


# figure()
# plot_line_chart(
#     arange(len(best_model.loss_curve_)),
#     best_model.loss_curve_,
#     title="Loss curve for MLP best model training",
#     xlabel="iterations",
#     ylabel="loss",
#     percentage=False,
# )
# savefig(f"images/{file_tag}_mlp_{eval_metric}_loss_curve.png")

# -----------------------------------------------
# Random Forest Classifier

# from numpy import array, ndarray
# from matplotlib.pyplot import subplots, figure, savefig, show
# from sklearn.ensemble import RandomForestClassifier
# from dslabs.dslabs_functions import (
#     CLASS_EVAL_METRICS,
#     DELTA_IMPROVE,
#     read_train_test_from_files,
# )
# from dslabs.dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart


# def random_forests_study(
#     trnX: ndarray,
#     trnY: array,
#     tstX: ndarray,
#     tstY: array,
#     nr_max_trees: int = 2500,
#     lag: int = 500,
#     metric: str = "accuracy",
# ) -> tuple[RandomForestClassifier | None, dict]:
#     n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
#     max_depths: list[int] = [2, 5, 7]
#     max_features: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

#     best_model: RandomForestClassifier | None = None
#     best_params: dict = {"name": "RF", "metric": metric, "params": ()}
#     best_performance: float = 0.0

#     values: dict = {}

#     cols: int = len(max_depths)
#     _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
#     for i in range(len(max_depths)):
#         d: int = max_depths[i]
#         values = {}
#         for f in max_features:
#             y_tst_values: list[float] = []
#             for n in n_estimators:
#                 clf = RandomForestClassifier(
#                     n_estimators=n, max_depth=d, max_features=f
#                 )
#                 clf.fit(trnX, trnY)
#                 prdY: array = clf.predict(tstX)
#                 eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
#                 y_tst_values.append(eval)
#                 if eval - best_performance > DELTA_IMPROVE:
#                     best_performance = eval
#                     best_params["params"] = (d, f, n)
#                     best_model = clf
#                 # print(f'RF d={d} f={f} n={n}')
#             values[f] = y_tst_values
#         plot_multiline_chart(
#             n_estimators,
#             values,
#             ax=axs[0, i],
#             title=f"Random Forests with max_depth={d}",
#             xlabel="nr estimators",
#             ylabel=metric,
#             percentage=True,
#         )
#     print(
#         f'RF best for {best_params["params"][2]} trees (d={best_params["params"][0]} and f={best_params["params"][1]})'
#     )
#     return best_model, best_params


# figure()
# best_model, params = random_forests_study(
#     trnX,
#     trnY,
#     tstX,
#     tstY,
#     nr_max_trees=1000,
#     lag=250,
#     metric=eval_metric,
# )
# savefig(f"images/{file_tag}_rf_{eval_metric}_study.png")
# show()

# prd_trn: array = best_model.predict(trnX)
# prd_tst: array = best_model.predict(tstX)
# figure()
# plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
# savefig(f'images/{file_tag}_rf_{params["name"]}_best_{params["metric"]}_eval.png')
# show()

# from numpy import std, argsort
# from dslabs.dslabs_functions import plot_horizontal_bar_chart

# stdevs: list[float] = list(
#     std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
# )
# importances = best_model.feature_importances_
# indices: list[int] = argsort(importances)[::-1]
# elems: list[str] = []
# imp_values: list[float] = []
# for f in range(len(vars)):
#     elems += [vars[indices[f]]]
#     imp_values.append(importances[indices[f]])
#     print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

# figure()
# plot_horizontal_bar_chart(
#     elems,
#     imp_values,
#     error=stdevs,
#     title="RF variables importance",
#     xlabel="importance",
#     ylabel="variables",
#     percentage=True,
# )
# savefig(f"images/{file_tag}_rf_{eval_metric}_vars_ranking.png")

# d_max: int = params["params"][0]
# feat: float = params["params"][1]
# nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

# y_tst_values: list[float] = []
# y_trn_values: list[float] = []
# acc_metric: str = "accuracy"

# for n in nr_estimators:
#     clf = RandomForestClassifier(n_estimators=n, max_depth=d_max, max_features=feat)
#     clf.fit(trnX, trnY)
#     prd_tst_Y: array = clf.predict(tstX)
#     prd_trn_Y: array = clf.predict(trnX)
#     y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
#     y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

# figure()
# plot_multiline_chart(
#     nr_estimators,
#     {"Train": y_trn_values, "Test": y_tst_values},
#     title=f"RF overfitting study for d={d_max} and f={feat}",
#     xlabel="nr_estimators",
#     ylabel=str(eval_metric),
#     percentage=True,
# )
# savefig(f"images/{file_tag}_rf_{eval_metric}_overfitting.png")

# -------------------------------------------------------------------
# Gradient Boosting Classifier

from numpy import array, ndarray
from matplotlib.pyplot import subplots, figure, savefig, show
from sklearn.ensemble import GradientBoostingClassifier
from dslabs.dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from dslabs.dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart


def gradient_boosting_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[GradientBoostingClassifier | None, dict]:
    n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    learning_rates: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model: GradientBoostingClassifier | None = None
    best_params: dict = {"name": "GB", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for lr in learning_rates:
            y_tst_values: list[float] = []
            for n in n_estimators:
                clf = GradientBoostingClassifier(
                    n_estimators=n, max_depth=d, learning_rate=lr
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (d, lr, n)
                    best_model = clf
                # print(f'GB d={d} lr={lr} n={n}')
            values[lr] = y_tst_values
        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"Gradient Boosting with max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'GB best for {best_params["params"][2]} trees (d={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params


trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
    train_filename, test_filename, target
)
print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")

figure()
best_model, params = gradient_boosting_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_trees=1000,
    lag=250,
    metric=eval_metric,
)
savefig(f"images/{file_tag}_gb_{eval_metric}_study.png")
show()

prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'images/{file_tag}_gb_{params["name"]}_best_{params["metric"]}_eval.png')
show()

from numpy import std, argsort
from dslabs.dslabs_functions import plot_horizontal_bar_chart

trees_importances: list[float] = []
for lst_trees in best_model.estimators_:
    for tree in lst_trees:
        trees_importances.append(tree.feature_importances_)

stdevs: list[float] = list(std(trees_importances, axis=0))
importances = best_model.feature_importances_
indices: list[int] = argsort(importances)[::-1]
elems: list[str] = []
imp_values: list[float] = []
for f in range(len(vars)):
    elems += [vars[indices[f]]]
    imp_values.append(importances[indices[f]])
    print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

figure()
plot_horizontal_bar_chart(
    elems,
    imp_values,
    error=stdevs,
    title="GB variables importance",
    xlabel="importance",
    ylabel="variables",
    percentage=True,
)
savefig(f"images/{file_tag}_gb_{eval_metric}_vars_ranking.png")

d_max: int = params["params"][0]
lr: float = params["params"][1]
nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric: str = "accuracy"

for n in nr_estimators:
    clf = GradientBoostingClassifier(n_estimators=n, max_depth=d_max, learning_rate=lr)
    clf.fit(trnX, trnY)
    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

figure()
plot_multiline_chart(
    nr_estimators,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"GB overfitting study for d={d_max} and lr={lr}",
    xlabel="nr_estimators",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"images/{file_tag}_gb_{eval_metric}_overfitting.png")