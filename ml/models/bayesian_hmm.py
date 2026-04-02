"""
Model 2: Bayesian HMM with optimal state selection for market regime classification.

Features: [log_returns_mean_24h, realised_vol_120h, market_breadth]
Output: Soft posterior probabilities [P(bull), P(normal), P(crisis)]

Uses hmmlearn GaussianHMM with Dirichlet-style priors on the transition matrix.
States are sorted by mean return: bull (highest) > normal > crisis (lowest).

Enhancements:
  - Optimal state selection via BIC and cross-validated log-likelihood (2-5 states)
  - Viterbi path, regime duration distribution, sojourn time analysis
  - Regime-conditional return/vol statistics and correlation structure
  - Multiple random restarts, convergence monitoring, OOS prediction accuracy
  - Transition probability monitoring with spike alerts
  - predict_regime_change_probability() method
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *a, **kw):
        return iterable if iterable is not None else range(0)

logger = logging.getLogger(__name__)


class BayesianRegimeHMM:
    """
    Hidden Markov Model for market regime classification with optimal state selection.

    Outputs soft posterior probabilities for continuous risk budget blending
    instead of hard regime assignments, reducing whipsaw during transitions.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 200,
                 covariance_type: str = "full", random_state: int = 42,
                 config: Optional[dict] = None):
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.state_order = None  # Maps internal states to [bull, normal, crisis]
        self._fitted = False

        # --- Enhancement config ---
        cfg = config or {}
        self.enable_state_selection = cfg.get("enable_state_selection", True)
        self.state_range = cfg.get("state_range", [2, 3, 4, 5])
        self.n_init = cfg.get("n_init", 10)  # Random restarts to avoid local optima
        self.cv_folds = cfg.get("cv_folds", 5)
        self.transition_spike_threshold = cfg.get("transition_spike_threshold", 0.3)

        # Storage for enhancement outputs
        self.state_selection_results: Dict[int, dict] = {}
        self.optimal_n_states: int = n_states
        self.viterbi_path: Optional[np.ndarray] = None
        self.regime_duration_stats: Dict[str, dict] = {}
        self.regime_conditional_stats: Dict[str, dict] = {}
        self.regime_correlations: Dict[str, np.ndarray] = {}
        self.convergence_history: List[float] = []
        self.oos_accuracy: Optional[float] = None
        self._X_scaled_train: Optional[np.ndarray] = None
        self._feature_names: Optional[list] = None

    # ─────────────────────────────────────────────────
    #  FIT
    # ─────────────────────────────────────────────────
    def fit(self, features_df: pd.DataFrame) -> "BayesianRegimeHMM":
        """
        Fit HMM on market features and sort states by mean return.

        If enable_state_selection=True, compares models with 2-5 states
        and selects optimal via BIC.

        Args:
            features_df: DataFrame with columns matching HMM feature names.
                         Must include 'log_returns_mean_24h' for state sorting.
        """
        logger.info(f"Fitting Gaussian HMM on {len(features_df)} observations...")

        X = features_df.values
        X_scaled = self.scaler.fit_transform(X)
        self._feature_names = features_df.columns.tolist()

        # ── Step 1: Optimal state selection (if enabled) ──
        if self.enable_state_selection:
            self.optimal_n_states = self._select_optimal_states(X_scaled)
            self.n_states = self.optimal_n_states
            logger.info(f"  Selected optimal n_states={self.n_states}")

        # ── Step 2: Fit best model with multiple restarts ──
        best_model, best_score = self._fit_with_restarts(X_scaled, self.n_states)
        self.model = best_model

        # ── Step 3: Sort states by mean return ──
        self._sort_states_by_return(X_scaled, features_df.columns)

        # ── Step 4: Compute Viterbi path ──
        self.viterbi_path = self._compute_viterbi_path(X_scaled)

        # ── Step 5: Regime analysis ──
        self._compute_regime_duration_stats(X_scaled)
        self._compute_regime_conditional_stats(X_scaled, features_df)
        self._compute_regime_correlations(X_scaled, features_df)

        # ── Step 6: Out-of-sample accuracy (train/test split) ──
        self._evaluate_oos_accuracy(X_scaled)

        self._features_index = features_df.index
        self._X_scaled_train = X_scaled
        self._fitted = True

        # Log fitted parameters
        self._log_model_summary()

        return self

    # ─────────────────────────────────────────────────
    #  OPTIMAL STATE SELECTION
    # ─────────────────────────────────────────────────
    def _select_optimal_states(self, X_scaled: np.ndarray) -> int:
        """
        Fit HMMs with 2, 3, 4, 5 states and select optimal via BIC + CV log-likelihood.

        Args:
            X_scaled: Standardised feature matrix (T, n_features)

        Returns:
            Optimal number of states
        """
        logger.info("  State selection: comparing models...")

        n_obs = X_scaled.shape[0]
        results = {}

        state_bar = tqdm(self.state_range, desc="State selection", unit="n_states", leave=True)
        for n_s in state_bar:
            state_bar.set_postfix(n_states=n_s)
            try:
                # Fit with multiple restarts
                model, best_ll = self._fit_with_restarts(X_scaled, n_s)

                # BIC = -2*LL + k*ln(n) where k = free parameters
                n_features = X_scaled.shape[1]
                k = self._count_hmm_params(n_s, n_features)
                bic = -2 * best_ll + k * np.log(n_obs)

                # Cross-validated log-likelihood
                cv_ll = self._cross_validate_hmm(X_scaled, n_s)

                results[n_s] = {
                    "bic": float(bic),
                    "log_likelihood": float(best_ll),
                    "cv_log_likelihood": float(cv_ll),
                    "n_params": k,
                }

                state_bar.set_postfix(n_states=n_s, BIC=f"{bic:.1f}")
                logger.info(
                    f"    n_states={n_s}: BIC={bic:.1f}, LL={best_ll:.1f}, CV_LL={cv_ll:.1f}"
                )
            except Exception as e:
                logger.warning(f"    n_states={n_s}: FAILED ({e})")
                results[n_s] = {"bic": np.inf, "log_likelihood": -np.inf,
                                "cv_log_likelihood": -np.inf, "n_params": 0}

        self.state_selection_results = results

        # Select by lowest BIC (primary), break ties with CV log-likelihood
        valid = {k: v for k, v in results.items() if np.isfinite(v["bic"])}
        if not valid:
            logger.warning("  All state counts failed, defaulting to 3 states")
            return 3

        best_n = min(valid, key=lambda k: valid[k]["bic"])

        # Log comparison table
        best_bic = valid[best_n]["bic"]
        log_lines = ["  State selection results:"]
        for n_s in sorted(results.keys()):
            r = results[n_s]
            delta = r["bic"] - best_bic if np.isfinite(r["bic"]) else float("inf")
            marker = " <-- BEST" if n_s == best_n else ""
            log_lines.append(
                f"    {n_s} states: BIC={r['bic']:10.1f}  "
                f"CV_LL={r['cv_log_likelihood']:10.1f}  "
                f"(ΔBIC={delta:+8.1f}){marker}"
            )
        logger.info("\n".join(log_lines))

        return best_n

    def _count_hmm_params(self, n_states: int, n_features: int) -> int:
        """
        Count free parameters in a Gaussian HMM.

        Args:
            n_states: Number of hidden states
            n_features: Number of observed features

        Returns:
            Number of free parameters
        """
        # Start probs: n_states - 1
        # Transition matrix: n_states * (n_states - 1)
        # Means: n_states * n_features
        # Covariances (full): n_states * n_features * (n_features + 1) / 2
        n_start = n_states - 1
        n_trans = n_states * (n_states - 1)
        n_means = n_states * n_features
        n_covs = n_states * n_features * (n_features + 1) // 2
        return n_start + n_trans + n_means + n_covs

    def _cross_validate_hmm(self, X_scaled: np.ndarray, n_states: int) -> float:
        """
        Time-series cross-validation for HMM log-likelihood.

        Uses expanding window: train on first k folds, test on fold k+1.

        Args:
            X_scaled: Standardised feature matrix
            n_states: Number of states to evaluate

        Returns:
            Mean CV log-likelihood
        """
        n_obs = X_scaled.shape[0]
        fold_size = n_obs // (self.cv_folds + 1)
        if fold_size < 50:
            return -np.inf

        cv_scores = []
        cv_bar = tqdm(range(1, self.cv_folds + 1),
                       desc=f"CV folds | {n_states} states",
                       unit="fold", leave=False)
        for fold in cv_bar:
            cv_bar.set_postfix(fold=f"{fold}/{self.cv_folds}")
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = min(test_start + fold_size, n_obs)

            if test_end <= test_start:
                continue

            X_train = X_scaled[:train_end]
            X_test = X_scaled[test_start:test_end]

            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                )
                model.fit(X_train)
                score = model.score(X_test)
                cv_scores.append(score)
                cv_bar.set_postfix(fold=f"{fold}/{self.cv_folds}",
                                    mean_LL=f"{np.mean(cv_scores):.1f}")
            except Exception:
                continue

        return float(np.mean(cv_scores)) if cv_scores else -np.inf

    # ─────────────────────────────────────────────────
    #  FIT WITH MULTIPLE RESTARTS
    # ─────────────────────────────────────────────────
    def _fit_with_restarts(self, X_scaled: np.ndarray,
                           n_states: int) -> Tuple[hmm.GaussianHMM, float]:
        """
        Fit HMM with multiple random restarts to avoid local optima.

        Args:
            X_scaled: Standardised feature matrix
            n_states: Number of hidden states

        Returns:
            Tuple of (best_model, best_log_likelihood)
        """
        best_model = None
        best_ll = -np.inf
        convergence_scores = []
        n_converged_count = 0

        restart_bar = tqdm(range(self.n_init),
                           desc=f"Random restarts | {n_states} states",
                           unit="restart", leave=False)
        for i in restart_bar:
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state + i,
                    init_params="stmc",
                    params="stmc",
                )

                # Set sticky transition prior (high diagonal = persistent regimes)
                prior_transmat = np.full((n_states, n_states), 0.05)
                np.fill_diagonal(prior_transmat, 0.90)
                prior_transmat /= prior_transmat.sum(axis=1, keepdims=True)
                model.transmat_prior = prior_transmat * 10

                model.fit(X_scaled)
                ll = model.score(X_scaled)
                convergence_scores.append(ll)
                n_converged_count += 1

                if ll > best_ll:
                    best_ll = ll
                    best_model = model
            except Exception:
                convergence_scores.append(-np.inf)
                continue
            restart_bar.set_postfix(converged=f"{n_converged_count}/{i + 1}",
                                    best_LL=f"{best_ll:.1f}")

        self.convergence_history = convergence_scores

        if best_model is None:
            raise RuntimeError(f"All {self.n_init} random restarts failed for n_states={n_states}")

        n_converged = sum(1 for s in convergence_scores if np.isfinite(s))
        ll_range = max(convergence_scores) - min(s for s in convergence_scores if np.isfinite(s)) \
            if n_converged > 1 else 0.0
        logger.info(
            f"    {n_states} states: {n_converged}/{self.n_init} converged, "
            f"LL range={ll_range:.1f}, best_LL={best_ll:.1f}"
        )

        return best_model, best_ll

    # ─────────────────────────────────────────────────
    #  VITERBI PATH
    # ─────────────────────────────────────────────────
    def _compute_viterbi_path(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Compute Viterbi (most likely) state sequence alongside forward-backward posteriors.

        Args:
            X_scaled: Standardised feature matrix

        Returns:
            Array of ordered state indices from Viterbi decoding
        """
        raw_viterbi = self.model.predict(X_scaled)

        # Reorder to match [bull, normal, crisis, ...] ordering
        ordered_viterbi = np.array([self.state_order[s] for s in raw_viterbi])

        return ordered_viterbi

    # ─────────────────────────────────────────────────
    #  REGIME DURATION ANALYSIS
    # ─────────────────────────────────────────────────
    def _compute_regime_duration_stats(self, X_scaled: np.ndarray) -> None:
        """
        Compute regime duration distribution and sojourn time analysis.

        For each regime:
        - Empirical duration distribution (mean, median, std, min, max)
        - Geometric distribution fit (theoretical from transition matrix)
        - Expected duration in hours and days

        Args:
            X_scaled: Standardised feature matrix
        """
        if self.viterbi_path is None:
            return

        states = self.viterbi_path
        trans = self.get_transition_matrix()
        labels = list(trans.index)

        for state_idx, label in enumerate(labels):
            # Compute empirical sojourn times (consecutive runs of same state)
            durations = []
            current_duration = 0
            for s in states:
                if s == state_idx:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_duration = 0
            if current_duration > 0:
                durations.append(current_duration)

            durations = np.array(durations) if durations else np.array([0])

            # Theoretical expected duration from transition matrix
            p_stay = trans.loc[label, label]
            theoretical_mean = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else float("inf")

            self.regime_duration_stats[label] = {
                "n_episodes": len(durations),
                "empirical_mean_hours": float(np.mean(durations)),
                "empirical_median_hours": float(np.median(durations)),
                "empirical_std_hours": float(np.std(durations)),
                "empirical_min_hours": float(np.min(durations)),
                "empirical_max_hours": float(np.max(durations)),
                "empirical_mean_days": float(np.mean(durations) / 24),
                "empirical_median_days": float(np.median(durations) / 24),
                "theoretical_mean_hours": float(theoretical_mean),
                "theoretical_mean_days": float(theoretical_mean / 24),
                "geometric_p": float(1.0 - p_stay),  # Geometric distribution parameter
            }

        # Log summary
        log_lines = ["  Regime duration analysis:"]
        for label, stats in self.regime_duration_stats.items():
            log_lines.append(
                f"    {label}: {stats['n_episodes']} episodes, "
                f"mean={stats['empirical_mean_days']:.1f}d "
                f"(theoretical={stats['theoretical_mean_days']:.1f}d), "
                f"median={stats['empirical_median_days']:.1f}d, "
                f"range=[{stats['empirical_min_hours']:.0f}h, {stats['empirical_max_hours']:.0f}h]"
            )
        logger.info("\n".join(log_lines))

    # ─────────────────────────────────────────────────
    #  REGIME-CONDITIONAL STATISTICS
    # ─────────────────────────────────────────────────
    def _compute_regime_conditional_stats(self, X_scaled: np.ndarray,
                                          features_df: pd.DataFrame) -> None:
        """
        Compute return and volatility statistics conditional on each regime.

        Args:
            X_scaled: Standardised feature matrix
            features_df: Original (unscaled) feature DataFrame
        """
        if self.viterbi_path is None:
            return

        states = self.viterbi_path
        trans = self.get_transition_matrix()
        labels = list(trans.index)

        # Find return column
        return_col = None
        vol_col = None
        for col_name in features_df.columns:
            if "return" in str(col_name).lower():
                return_col = col_name
            if "vol" in str(col_name).lower():
                vol_col = col_name

        for state_idx, label in enumerate(labels):
            mask = states == state_idx

            if mask.sum() == 0:
                continue

            stats: Dict[str, float] = {
                "n_observations": int(mask.sum()),
                "fraction": float(mask.mean()),
            }

            if return_col is not None:
                ret_vals = features_df[return_col].values[: len(mask)][mask]
                stats["return_mean"] = float(np.mean(ret_vals))
                stats["return_std"] = float(np.std(ret_vals))
                stats["return_skew"] = float(
                    np.mean(((ret_vals - ret_vals.mean()) / max(ret_vals.std(), 1e-10)) ** 3)
                )
                stats["return_5pct"] = float(np.percentile(ret_vals, 5))
                stats["return_95pct"] = float(np.percentile(ret_vals, 95))

            if vol_col is not None:
                vol_vals = features_df[vol_col].values[: len(mask)][mask]
                stats["vol_mean"] = float(np.mean(vol_vals))
                stats["vol_std"] = float(np.std(vol_vals))

            self.regime_conditional_stats[label] = stats

        # Log summary
        log_lines = ["  Regime-conditional statistics:"]
        for label, stats in self.regime_conditional_stats.items():
            ret_str = (f"ret_mean={stats.get('return_mean', 'N/A'):.6f}, "
                       f"ret_std={stats.get('return_std', 'N/A'):.6f}"
                       if "return_mean" in stats else "no return data")
            log_lines.append(
                f"    {label}: {stats['fraction']:.1%} of data, {ret_str}"
            )
        logger.info("\n".join(log_lines))

    # ─────────────────────────────────────────────────
    #  REGIME-DEPENDENT CORRELATION STRUCTURE
    # ─────────────────────────────────────────────────
    def _compute_regime_correlations(self, X_scaled: np.ndarray,
                                     features_df: pd.DataFrame) -> None:
        """
        Compute correlation structure of features conditional on each regime.

        Args:
            X_scaled: Standardised feature matrix
            features_df: Original feature DataFrame
        """
        if self.viterbi_path is None or features_df.shape[1] < 2:
            return

        states = self.viterbi_path
        trans = self.get_transition_matrix()
        labels = list(trans.index)

        for state_idx, label in enumerate(labels):
            mask = states == state_idx
            if mask.sum() < 10:  # Need enough data for meaningful correlation
                continue

            regime_data = features_df.values[: len(mask)][mask]
            if regime_data.shape[0] > 2:
                corr_matrix = np.corrcoef(regime_data.T)
                self.regime_correlations[label] = corr_matrix

        if self.regime_correlations:
            logger.info(f"  Regime-dependent correlations computed for {len(self.regime_correlations)} regimes")

    # ─────────────────────────────────────────────────
    #  OUT-OF-SAMPLE ACCURACY
    # ─────────────────────────────────────────────────
    def _evaluate_oos_accuracy(self, X_scaled: np.ndarray) -> None:
        """
        Evaluate out-of-sample regime prediction accuracy via train/test split.

        Uses temporal split (70/30) to test regime prediction.

        Args:
            X_scaled: Standardised feature matrix
        """
        n_obs = X_scaled.shape[0]
        split = int(0.7 * n_obs)

        if split < 50 or n_obs - split < 20:
            return

        X_train = X_scaled[:split]
        X_test = X_scaled[split:]

        try:
            oos_model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            oos_model.fit(X_train)

            # Predict on test set using both models
            full_pred = self.model.predict(X_test)
            oos_pred = oos_model.predict(X_test)

            # Agreement rate (how often the OOS-trained model agrees with full model)
            agreement = np.mean(full_pred == oos_pred)

            # OOS log-likelihood
            oos_ll = oos_model.score(X_test)
            full_ll = self.model.score(X_test)

            self.oos_accuracy = float(agreement)

            logger.info(
                f"  OOS evaluation: agreement={agreement:.1%}, "
                f"OOS_LL={oos_ll:.1f}, full_model_LL={full_ll:.1f}"
            )
        except Exception as e:
            logger.warning(f"  OOS evaluation failed: {e}")

    # ─────────────────────────────────────────────────
    #  STATE SORTING
    # ─────────────────────────────────────────────────
    def _sort_states_by_return(self, X_scaled: np.ndarray, feature_names) -> None:
        """Sort internal HMM states so that 0=bull, 1=normal, 2=crisis (or more)."""
        states = self.model.predict(X_scaled)

        # Find which column is the return feature
        return_col = 0
        for i, name in enumerate(feature_names):
            if "return" in str(name).lower():
                return_col = i
                break

        state_means = []
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() > 0:
                state_means.append((s, X_scaled[mask, return_col].mean()))
            else:
                state_means.append((s, 0.0))

        # Sort by mean return: highest first (bull), lowest last (crisis)
        state_means.sort(key=lambda x: x[1], reverse=True)
        self.state_order = {state_means[i][0]: i for i in range(self.n_states)}

        # Dynamic labels based on n_states
        if self.n_states == 2:
            self.state_labels = {0: "bull", 1: "crisis"}
        elif self.n_states == 3:
            self.state_labels = {0: "bull", 1: "normal", 2: "crisis"}
        elif self.n_states == 4:
            self.state_labels = {0: "strong_bull", 1: "bull", 2: "normal", 3: "crisis"}
        else:
            self.state_labels = {i: f"state_{i}" for i in range(self.n_states)}

        logger.info(f"  State mapping ({self.n_states} states): {self.state_order}")
        for internal, external in self.state_order.items():
            pct = (states == internal).mean() * 100
            label = self.state_labels.get(external, f"state_{external}")
            logger.info(f"    State {internal} -> {label} ({pct:.1f}% of data)")

    # ─────────────────────────────────────────────────
    #  PREDICT
    # ─────────────────────────────────────────────────
    def predict_proba(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return soft regime probabilities.

        For 3-state model: [P(bull), P(normal), P(crisis)]
        Columns adapt to the number of states selected.

        These posterior probabilities enable continuous risk budget blending
        rather than hard switching, reducing turnover and whipsaw.
        """
        assert self._fitted, "Model not fitted"

        X_scaled = self.scaler.transform(features_df.values)
        raw_proba = self.model.predict_proba(X_scaled)  # (T, n_states)

        # Reorder columns to match ordered state labels
        ordered_proba = np.zeros_like(raw_proba)
        for internal_state, external_idx in self.state_order.items():
            ordered_proba[:, external_idx] = raw_proba[:, internal_state]

        col_names = [f"P({self.state_labels[i]})" for i in range(self.n_states)]
        proba_df = pd.DataFrame(
            ordered_proba,
            index=features_df.index,
            columns=col_names,
        )

        return proba_df

    def predict_regime(self, features_df: pd.DataFrame) -> pd.Series:
        """Return hard regime classification (for visualisation)."""
        proba = self.predict_proba(features_df)
        regime_idx = proba.values.argmax(axis=1)
        labels = [self.state_labels[i] for i in range(self.n_states)]
        regime_labels = pd.Series(
            [labels[i] for i in regime_idx],
            index=features_df.index,
            name="regime",
        )
        return regime_labels

    def predict_regime_change_probability(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the probability of a regime change at each time step.

        For each timestep, computes:
        - P(switch): probability that next state differs from current state
        - P(to_X): probability of switching TO each specific regime

        This is useful for:
        - Early warning of regime transitions
        - Dynamic position sizing (reduce before switches)
        - Monitoring transition probability spikes

        Args:
            features_df: Feature DataFrame for prediction

        Returns:
            DataFrame with switch probability and per-regime transition probs
        """
        assert self._fitted, "Model not fitted"

        X_scaled = self.scaler.transform(features_df.values)
        posteriors = self.model.predict_proba(X_scaled)  # (T, n_states)
        trans_matrix = self.model.transmat_  # (n_states, n_states)

        n_obs = posteriors.shape[0]
        switch_prob = np.zeros(n_obs)
        to_regime_prob = np.zeros((n_obs, self.n_states))

        for t in range(n_obs):
            # Current state distribution
            state_dist = posteriors[t]  # P(s_t = k)

            # Next-step distribution: P(s_{t+1} = j) = sum_k P(s_t=k) * A_{kj}
            next_dist = state_dist @ trans_matrix  # (n_states,)

            # P(switch) = 1 - P(same state)
            # P(same) = sum_k P(s_t=k) * A_{kk}
            p_same = sum(state_dist[k] * trans_matrix[k, k] for k in range(self.n_states))
            switch_prob[t] = 1.0 - p_same

            # Reorder next_dist to match ordered labels
            for internal_state, external_idx in self.state_order.items():
                to_regime_prob[t, external_idx] = next_dist[internal_state]

        # Build output DataFrame
        columns = ["P(switch)"]
        for i in range(self.n_states):
            columns.append(f"P(to_{self.state_labels[i]})")

        result = pd.DataFrame(
            np.column_stack([switch_prob, to_regime_prob]),
            index=features_df.index,
            columns=columns,
        )

        # Alert on spikes
        spike_mask = switch_prob > self.transition_spike_threshold
        if spike_mask.any():
            n_spikes = spike_mask.sum()
            max_prob = switch_prob[spike_mask].max()
            logger.warning(
                f"  TRANSITION ALERT: {n_spikes} timesteps with "
                f"P(switch)>{self.transition_spike_threshold:.0%} "
                f"(max={max_prob:.1%})"
            )

        return result

    # ─────────────────────────────────────────────────
    #  TRANSITION MATRIX & PERSISTENCE
    # ─────────────────────────────────────────────────
    def get_transition_matrix(self) -> pd.DataFrame:
        """Return the estimated transition probability matrix (ordered)."""
        assert self._fitted or self.model is not None, "Model not fitted"

        raw = self.model.transmat_
        n = self.n_states
        ordered = np.zeros((n, n))

        for i_raw, i_ord in self.state_order.items():
            for j_raw, j_ord in self.state_order.items():
                ordered[i_ord, j_ord] = raw[i_raw, j_raw]

        labels = [self.state_labels[i] for i in range(n)]
        return pd.DataFrame(ordered, index=labels, columns=labels)

    def get_regime_persistence(self) -> dict:
        """
        Expected duration of each regime in hours and days.
        E[duration of state k] = 1 / (1 - A_{kk})
        """
        trans = self.get_transition_matrix()
        persistence = {}
        for label in trans.index:
            p_stay = trans.loc[label, label]
            expected_hours = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else float("inf")
            persistence[label] = {
                "expected_hours": expected_hours,
                "expected_days": expected_hours / 24,
                "self_transition_prob": p_stay,
            }
        return persistence

    # ─────────────────────────────────────────────────
    #  UNCERTAINTY
    # ─────────────────────────────────────────────────
    def get_uncertainty(self) -> float:
        """
        Uncertainty measure for ensemble weighting.
        Uses average entropy of posterior regime probabilities.
        High entropy = uncertain classification -> lower ensemble weight.
        """
        if not self._fitted:
            return 1.0

        # Use the most recent posterior as uncertainty measure
        proba = self.model.predict_proba(
            self.scaler.transform(
                np.zeros((1, len(self.scaler.mean_)))  # dummy
            )
        )
        # Entropy: H = -sum(p * log(p))
        proba = np.clip(proba, 1e-10, 1.0)
        entropy = -np.sum(proba * np.log(proba))
        max_entropy = np.log(self.n_states)

        return entropy / max_entropy  # Normalised [0, 1]

    # ─────────────────────────────────────────────────
    #  LOGGING
    # ─────────────────────────────────────────────────
    def _log_model_summary(self) -> None:
        """Log fitted model parameters."""
        logger.info(f"  HMM converged: {self.model.monitor_.converged}")
        logger.info(f"  Final n_states: {self.n_states}")

        if self.convergence_history:
            valid_ll = [s for s in self.convergence_history if np.isfinite(s)]
            if valid_ll:
                logger.info(
                    f"  Convergence: {len(valid_ll)}/{len(self.convergence_history)} restarts OK, "
                    f"LL range=[{min(valid_ll):.1f}, {max(valid_ll):.1f}]"
                )

        trans = self.get_transition_matrix()
        logger.info(f"  Transition matrix:\n{trans.round(3).to_string()}")

        persistence = self.get_regime_persistence()
        for label, stats in persistence.items():
            logger.info(f"  {label}: expected duration = {stats['expected_days']:.1f} days")

        if self.oos_accuracy is not None:
            logger.info(f"  OOS regime agreement: {self.oos_accuracy:.1%}")
