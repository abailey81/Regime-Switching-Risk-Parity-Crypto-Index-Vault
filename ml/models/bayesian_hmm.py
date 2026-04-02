"""
Model 2: Bayesian HMM with 3 states (Bull / Normal / Crisis).

Features: [log_returns_mean_24h, realised_vol_120h, market_breadth]
Output: Soft posterior probabilities [P(bull), P(normal), P(crisis)]

Uses hmmlearn GaussianHMM with Dirichlet-style priors on the transition matrix.
States are sorted by mean return: bull (highest) > normal > crisis (lowest).
"""
import logging
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BayesianRegimeHMM:
    """
    3-state Hidden Markov Model for market regime classification.

    Outputs soft posterior probabilities for continuous risk budget blending
    instead of hard regime assignments, reducing whipsaw during transitions.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 200,
                 covariance_type: str = "full", random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.state_order = None  # Maps internal states to [bull, normal, crisis]
        self._fitted = False

    def fit(self, features_df: pd.DataFrame) -> "BayesianRegimeHMM":
        """
        Fit HMM on market features and sort states by mean return.

        Args:
            features_df: DataFrame with columns matching HMM feature names.
                         Must include 'log_returns_mean_24h' for state sorting.
        """
        logger.info(f"Fitting {self.n_states}-state Gaussian HMM on {len(features_df)} observations...")

        X = features_df.values
        X_scaled = self.scaler.fit_transform(X)

        # Fit HMM with sticky transition prior (high diagonal = persistent regimes)
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params="stmc",
            params="stmc",
        )

        # Set prior: encourage sticky transitions (each state persists)
        # Higher diagonal = longer expected regime duration
        prior_transmat = np.full((self.n_states, self.n_states), 0.05)
        np.fill_diagonal(prior_transmat, 0.90)
        prior_transmat /= prior_transmat.sum(axis=1, keepdims=True)
        self.model.transmat_prior = prior_transmat * 10  # Concentration parameter

        self.model.fit(X_scaled)

        # Sort states by mean return (column 0 = log_returns_mean_24h)
        self._sort_states_by_return(X_scaled, features_df.columns)

        self._features_index = features_df.index
        self._fitted = True

        # Log fitted parameters
        self._log_model_summary()

        return self

    def _sort_states_by_return(self, X_scaled, feature_names):
        """Sort internal HMM states so that 0=bull, 1=normal, 2=crisis."""
        # Predict states to get mean return per state
        states = self.model.predict(X_scaled)

        # Find which column is the return feature
        return_col = 0  # Assume first column is mean return
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
        self.state_labels = {0: "bull", 1: "normal", 2: "crisis"}

        logger.info(f"  State mapping: {self.state_order}")
        for internal, external in self.state_order.items():
            pct = (states == internal).mean() * 100
            logger.info(f"    State {internal} → {self.state_labels[external]} ({pct:.1f}% of data)")

    def predict_proba(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return soft regime probabilities [P(bull), P(normal), P(crisis)].

        These posterior probabilities enable continuous risk budget blending
        rather than hard switching, reducing turnover and whipsaw.
        """
        assert self._fitted, "Model not fitted"

        X_scaled = self.scaler.transform(features_df.values)
        raw_proba = self.model.predict_proba(X_scaled)  # (T, n_states)

        # Reorder columns to match [bull, normal, crisis]
        ordered_proba = np.zeros_like(raw_proba)
        for internal_state, external_idx in self.state_order.items():
            ordered_proba[:, external_idx] = raw_proba[:, internal_state]

        proba_df = pd.DataFrame(
            ordered_proba,
            index=features_df.index,
            columns=["P(bull)", "P(normal)", "P(crisis)"]
        )

        return proba_df

    def predict_regime(self, features_df: pd.DataFrame) -> pd.Series:
        """Return hard regime classification (for visualisation)."""
        proba = self.predict_proba(features_df)
        regime_idx = proba.values.argmax(axis=1)
        labels = ["bull", "normal", "crisis"]
        regime_labels = pd.Series([labels[i] for i in regime_idx],
                                   index=features_df.index, name="regime")
        return regime_labels

    def get_transition_matrix(self) -> pd.DataFrame:
        """Return the estimated transition probability matrix (ordered)."""
        assert self._fitted, "Model not fitted"

        raw = self.model.transmat_
        n = self.n_states
        ordered = np.zeros((n, n))

        for i_raw, i_ord in self.state_order.items():
            for j_raw, j_ord in self.state_order.items():
                ordered[i_ord, j_ord] = raw[i_raw, j_raw]

        labels = ["bull", "normal", "crisis"]
        return pd.DataFrame(ordered, index=labels, columns=labels)

    def get_regime_persistence(self) -> dict:
        """
        Expected duration of each regime in hours.
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

    def get_uncertainty(self) -> float:
        """
        Uncertainty measure for ensemble weighting.
        Uses average entropy of posterior regime probabilities.
        High entropy = uncertain classification → lower ensemble weight.
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

    def _log_model_summary(self):
        """Log fitted model parameters."""
        logger.info(f"  HMM converged: {self.model.monitor_.converged}")
        logger.info(f"  Log-likelihood: {self.model.score(self.scaler.transform(np.zeros((1, len(self.scaler.mean_))))):.2f}")

        trans = self.get_transition_matrix()
        logger.info(f"  Transition matrix:\n{trans.round(3).to_string()}")

        persistence = self.get_regime_persistence()
        for label, stats in persistence.items():
            logger.info(f"  {label}: expected duration = {stats['expected_days']:.1f} days")
