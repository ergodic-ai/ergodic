*[confounder]: A common cause behind both treatment and outcome that fakes an effect.
*[confounders]: Common causes behind both treatment and outcome that fake an effect.
*[confounding]: Bias from a common cause of both treatment and outcome.
*[collider]: A node two arrows point into (X to Z from Y); a shared effect that blocks a path until you condition on it.
*[mediator]: A variable on the causal path from treatment to outcome, carrying part of the effect.
*[estimand]: The formula identification produces: the effect written in terms of observable quantities.
*[identification]: Deciding whether an effect can be computed from the data, and writing the formula, before touching numbers.
*[skeleton]: A graph's adjacencies with all arrow directions stripped off.
*[face]: Which graph kind a result presents as: dag, cpdag, pag, or mixed.
*[propensity]: The modeled probability of treatment given the covariates.
*[nuisance]: A helper model an estimator needs (outcome, treatment) but does not report as the answer.
*[positivity]: Every kind of unit appears both treated and untreated, so there is somebody to compare with.
*[overlap]: Every kind of unit appears both treated and untreated, so there is somebody to compare with.
*[selection bias]: Distortion from only keeping rows where some condition held, which can correlate that condition's causes.
*[counterfactual]: What would have happened to a unit under the treatment it did not actually get.
*[exclusion restriction]: The assumption that an instrument reaches the outcome only through the treatment.
*[exogenous]: Not caused by the other variables; a root of the causal order.
*[DAG]: Directed acyclic graph: one-way arrows, no loops; a fully known causal model.
*[ADMG]: A graph that adds two-headed arrows for hidden common causes.
*[CPDAG]: What constraint-based discovery returns: arrows where the data decides, dashes where several DAGs fit equally well.
*[MAG]: A graph type that handles hidden confounders and selection bias together.
*[PAG]: What FCI returns: a class of MAGs, with circle marks where the class cannot decide an endpoint.
*[v-structure]: A collider whose two parents are not adjacent; the fingerprint discovery can orient.
*[ATE]: Average treatment effect, over everyone.
*[ATT]: Average treatment effect on the treated.
*[LATE]: Local average treatment effect, for the compliers an instrument moves.
*[CATE]: Conditional average treatment effect: the effect as a function of who the unit is.
*[AIPW]: Augmented inverse-probability weighting; a doubly robust estimator correct if either model is.
*[IPW]: Inverse-probability weighting; reweight units by one over their treatment probability.
*[DML]: Double machine learning; a cross-fit estimator that uses flexible models while keeping valid error bars.
*[doubly robust]: Correct if at least one of the two models (outcome, treatment) is correct.
*[cross-fitting]: Fit nuisance models on one slice of data and score on another, so flexible models cannot bias the estimate.
*[propensity score]: The modeled probability of treatment given the covariates.
*[IV]: Instrumental variable; moves the treatment but reaches the outcome only through it.
*[2SLS]: Two-stage least squares, the linear instrumental-variable estimator.
*[PC]: A constraint-based discovery algorithm; returns a CPDAG.
*[FCI]: A constraint-based discovery algorithm that tolerates hidden confounders; returns a PAG.
*[RFCI]: A faster relaxation of FCI.
*[GES]: Greedy Equivalence Search, a score-based discovery algorithm.
*[PCMCI]: A time-series discovery method; returns a window graph over lagged variables.
*[DiD]: Difference-in-differences; nets out fixed differences and the shared calendar across time.
*[ITS]: Interrupted time series; fit level and slope before and after a dated change.
*[RMSPE]: Root-mean-square prediction error; how closely a synthetic control tracked before treatment.
*[parallel trends]: The DiD assumption that, without treatment, the groups would have moved together over time.
*[SUTVA]: The assumption that one unit's treatment does not affect another's outcome.
*[event log]: A process record with one row per event: case id, activity, timestamp.
*[DFG]: Directly-follows graph; counts of how often one activity directly follows another.
*[conformance]: Measuring how well a log's real traces fit a process model.
*[soundness]: A process model where every run can finish, no token is stranded, and every step is reachable.
*[estimator]: The object encoding how to turn an estimand into a number.
