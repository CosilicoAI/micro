"""
Calibration methods for adjusting sample weights to match external targets.

Implements classical survey calibration algorithms:
- IPF (Iterative Proportional Fitting) / Raking
- Chi-square distance minimization
- Entropy balancing

These methods adjust weights so weighted aggregates match known population
statistics (e.g., total income from IRS SOI, population counts from Census).

Example:
    >>> from microplex.calibration import Calibrator
    >>> calibrator = Calibrator(method="entropy")
    >>> targets = {
    ...     "state": {"CA": 400, "NY": 350, "TX": 250},
    ...     "age_group": {"0-17": 250, "18-64": 550, "65+": 200},
    ... }
    >>> continuous_targets = {"income": 50_000_000}  # Total income
    >>> calibrated = calibrator.fit_transform(data, targets, continuous_targets)
"""

from typing import Dict, List, Optional, Union, Literal, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class Calibrator:
    """
    Calibrate sample weights to match external aggregate targets.

    Three calibration methods are supported:
    - IPF (Iterative Proportional Fitting): Classic raking algorithm that
      iteratively adjusts weights to match marginal totals
    - Chi-square: Minimizes chi-square distance from initial weights while
      matching targets (quadratic loss)
    - Entropy: Minimizes Kullback-Leibler divergence from initial weights
      (exponential tilting)

    Key features:
    - Supports categorical margin constraints (state, age group, etc.)
    - Supports continuous total constraints (total income, population)
    - Validation to compare weighted aggregates to targets
    - Convergence diagnostics

    Example:
        >>> from microplex.calibration import Calibrator
        >>> calibrator = Calibrator(method="ipf")
        >>> targets = {"state": {"CA": 1000, "NY": 500}}
        >>> calibrated = calibrator.fit_transform(data, targets)
    """

    def __init__(
        self,
        method: Literal["ipf", "chi2", "entropy"] = "ipf",
        tol: float = 1e-6,
        max_iter: int = 100,
        lower_bound: float = 1e-10,
        upper_bound: Optional[float] = None,
    ):
        """
        Initialize calibrator.

        Args:
            method: Calibration method ("ipf", "chi2", or "entropy")
            tol: Convergence tolerance for iterative methods
            max_iter: Maximum number of iterations
            lower_bound: Minimum allowed weight (prevents zero weights)
            upper_bound: Maximum allowed weight (None for no limit)

        Raises:
            ValueError: If method is not one of "ipf", "chi2", "entropy"
        """
        if method not in ["ipf", "chi2", "entropy"]:
            raise ValueError(
                f"Invalid method: {method}. Must be 'ipf', 'chi2', or 'entropy'"
            )

        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Set during fit
        self.weights_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False
        self.n_records_: Optional[int] = None
        self.marginal_targets_: Optional[Dict[str, Dict[str, float]]] = None
        self.continuous_targets_: Optional[Dict[str, float]] = None
        self.convergence_history_: List[Dict[str, Any]] = []
        self.n_iterations_: int = 0
        self.converged_: bool = False

    def fit(
        self,
        data: pd.DataFrame,
        marginal_targets: Dict[str, Dict[str, float]],
        continuous_targets: Optional[Dict[str, float]] = None,
        weight_col: str = "weight",
    ) -> "Calibrator":
        """
        Fit calibration weights to match targets.

        Args:
            data: DataFrame with microdata records
            marginal_targets: Dict of categorical targets {var: {category: count}}
            continuous_targets: Dict of continuous totals {var: total}
            weight_col: Name of initial weight column in data

        Returns:
            self

        Raises:
            ValueError: If margin variable not in data or category mismatch
        """
        self.n_records_ = len(data)
        self.marginal_targets_ = marginal_targets
        self.continuous_targets_ = continuous_targets or {}

        # Validate inputs
        self._validate_inputs(data, marginal_targets, continuous_targets)

        # Get initial weights
        if weight_col in data.columns:
            initial_weights = data[weight_col].values.astype(float)
        else:
            initial_weights = np.ones(len(data), dtype=float)

        # Handle zero initial weights
        initial_weights = np.maximum(initial_weights, self.lower_bound)

        # Build constraint matrices
        A_cat, b_cat = self._build_categorical_constraints(data, marginal_targets)
        A_cont, b_cont = self._build_continuous_constraints(data, continuous_targets)

        # Combine constraints
        if A_cont is not None:
            A = np.vstack([A_cat, A_cont])
            b = np.concatenate([b_cat, b_cont])
        else:
            A = A_cat
            b = b_cat

        # Solve based on method
        if self.method == "ipf":
            weights = self._fit_ipf(data, initial_weights, marginal_targets, continuous_targets)
        elif self.method == "chi2":
            weights = self._fit_chi2(A, b, initial_weights)
        else:  # entropy
            weights = self._fit_entropy(A, b, initial_weights)

        self.weights_ = weights
        self.is_fitted_ = True

        return self

    def _validate_inputs(
        self,
        data: pd.DataFrame,
        marginal_targets: Dict[str, Dict[str, float]],
        continuous_targets: Optional[Dict[str, float]],
    ):
        """Validate input data and targets."""
        # Check margin variables exist
        for var in marginal_targets:
            if var not in data.columns:
                raise ValueError(f"Margin variable '{var}' not in data columns")

            # Check all data categories are in targets
            data_categories = set(data[var].unique())
            target_categories = set(marginal_targets[var].keys())
            missing = data_categories - target_categories
            if missing:
                raise ValueError(
                    f"Data contains categories not in targets for '{var}': {missing}"
                )

        # Check continuous target variables exist
        if continuous_targets:
            for var in continuous_targets:
                if var not in data.columns:
                    raise ValueError(f"Continuous target variable '{var}' not in data columns")

    def _build_categorical_constraints(
        self,
        data: pd.DataFrame,
        marginal_targets: Dict[str, Dict[str, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build constraint matrix for categorical margins."""
        constraints = []
        targets = []

        for var, var_targets in marginal_targets.items():
            for category, target in var_targets.items():
                # Indicator: 1 if record matches category
                indicator = (data[var] == category).astype(float).values
                constraints.append(indicator)
                targets.append(target)

        A = np.vstack(constraints) if constraints else np.zeros((0, len(data)))
        b = np.array(targets)

        return A, b

    def _build_continuous_constraints(
        self,
        data: pd.DataFrame,
        continuous_targets: Optional[Dict[str, float]],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build constraint matrix for continuous totals."""
        if not continuous_targets:
            return None, None

        constraints = []
        targets = []

        for var, target in continuous_targets.items():
            constraints.append(data[var].values.astype(float))
            targets.append(target)

        A = np.vstack(constraints)
        b = np.array(targets)

        return A, b

    def _fit_ipf(
        self,
        data: pd.DataFrame,
        initial_weights: np.ndarray,
        marginal_targets: Dict[str, Dict[str, float]],
        continuous_targets: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """
        Iterative Proportional Fitting (raking).

        Classic algorithm that iteratively adjusts weights to match
        each set of marginal totals.

        Algorithm:
            1. For each margin variable:
               - Compute current weighted totals
               - Compute adjustment factor = target / current
               - Multiply weights by adjustment factor for each category
            2. Repeat until convergence

        Note: For continuous targets, IPF uses linear regression-based
        calibration within each iteration, which can disrupt categorical
        margins. For best results with continuous targets, use 'chi2' or
        'entropy' methods instead.
        """
        weights = initial_weights.copy()
        self.convergence_history_ = []

        for iteration in range(self.max_iter):
            old_weights = weights.copy()

            # Adjust for each categorical margin
            for var, var_targets in marginal_targets.items():
                for category, target in var_targets.items():
                    mask = data[var] == category
                    current_total = weights[mask].sum()

                    if current_total > self.lower_bound:
                        adjustment = target / current_total
                        weights[mask] *= adjustment

            # Adjust for continuous targets using linear calibration
            # This is a generalized raking approach for continuous variables
            if continuous_targets:
                for var, target in continuous_targets.items():
                    values = data[var].values
                    current_total = (weights * values).sum()

                    if abs(current_total) > self.lower_bound:
                        # Linear calibration: w_new = w * (1 + lambda * x)
                        # where lambda is chosen so sum(w_new * x) = target
                        # This is equivalent to: lambda = (target - current) / sum(w * x^2)
                        weighted_x2 = (weights * values ** 2).sum()
                        if weighted_x2 > self.lower_bound:
                            lam = (target - current_total) / weighted_x2
                            # Apply with damping to prevent negative weights
                            adjustment = 1 + lam * values
                            adjustment = np.maximum(adjustment, 0.1)  # Prevent negative/zero
                            weights = weights * adjustment

            # Apply bounds
            weights = np.maximum(weights, self.lower_bound)
            if self.upper_bound is not None:
                weights = np.minimum(weights, self.upper_bound)

            # Check convergence
            max_change = np.max(np.abs(weights - old_weights) / np.maximum(old_weights, self.lower_bound))

            self.convergence_history_.append({
                "iteration": iteration + 1,
                "max_error": max_change,
            })

            if max_change < self.tol:
                self.converged_ = True
                self.n_iterations_ = iteration + 1
                break

        else:
            self.converged_ = False
            self.n_iterations_ = self.max_iter

        return weights

    def _fit_chi2(
        self,
        A: np.ndarray,
        b: np.ndarray,
        initial_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Chi-square distance minimization.

        Finds weights w that minimize chi-square distance from initial weights w0:
            min sum((w - w0)^2 / w0)
        subject to:
            A @ w = b (constraints)
            w >= lower_bound

        This is a quadratic programming problem solved via Lagrange multipliers.
        """
        n = len(initial_weights)
        w0 = initial_weights

        # Lagrangian: L = sum((w - w0)^2 / w0) + lambda^T (A @ w - b)
        # Taking derivative and setting to zero:
        # 2(w - w0)/w0 + A^T @ lambda = 0
        # w = w0 * (1 - 0.5 * w0 * A^T @ lambda)
        #
        # Substituting into constraint A @ w = b:
        # A @ (w0 * (1 - 0.5 * w0 * A^T @ lambda)) = b
        #
        # We solve this iteratively using scipy.optimize

        def objective(w):
            """Chi-square distance."""
            return np.sum((w - w0) ** 2 / w0)

        def gradient(w):
            """Gradient of chi-square distance."""
            return 2 * (w - w0) / w0

        # Constraint: A @ w = b
        constraints = []
        for i in range(len(b)):
            constraints.append({
                "type": "eq",
                "fun": lambda w, i=i: A[i] @ w - b[i],
                "jac": lambda w, i=i: A[i],
            })

        # Bounds
        bounds = [(self.lower_bound, self.upper_bound) for _ in range(n)]

        # Initial guess: scale initial weights to approximately satisfy constraints
        scale = b.sum() / (A.sum(axis=0) @ w0) if (A.sum(axis=0) @ w0) > 0 else 1.0
        x0 = w0 * scale

        self.convergence_history_ = []

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            jac=gradient,
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        self.n_iterations_ = result.nit if hasattr(result, "nit") else self.max_iter
        self.converged_ = result.success

        self.convergence_history_.append({
            "iteration": self.n_iterations_,
            "max_error": result.fun if result.success else np.inf,
        })

        if not result.success:
            # Fallback: use least squares with non-negativity
            from scipy.optimize import nnls
            try:
                weights, _ = nnls(A.T, b)
                # Scale to better match initial weights
                if weights.sum() > 0:
                    weights = weights * (w0.sum() / weights.sum())
            except Exception:
                weights = w0

            return np.maximum(weights, self.lower_bound)

        return result.x

    def _fit_entropy(
        self,
        A: np.ndarray,
        b: np.ndarray,
        initial_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Entropy balancing (exponential tilting).

        Finds weights w that minimize Kullback-Leibler divergence from initial w0:
            min sum(w * log(w / w0))
        subject to:
            A @ w = b (constraints)
            w >= lower_bound

        This has closed-form solution: w = w0 * exp(A^T @ lambda)
        where lambda are Lagrange multipliers found by solving dual problem.
        """
        n = len(initial_weights)
        w0 = initial_weights
        n_constraints = len(b)

        def dual_objective(lam):
            """
            Dual function: -sum(w0 * exp(A^T @ lam)) + lam^T @ b
            We minimize the negative of this (maximize dual).
            """
            exp_term = w0 * np.exp(A.T @ lam)
            return np.sum(exp_term) - lam @ b

        def dual_gradient(lam):
            """Gradient of dual objective."""
            exp_term = w0 * np.exp(A.T @ lam)
            return A @ exp_term - b

        def dual_hessian(lam):
            """Hessian of dual objective."""
            exp_term = w0 * np.exp(A.T @ lam)
            return A @ np.diag(exp_term) @ A.T

        # Initial lambda (Lagrange multipliers)
        lam0 = np.zeros(n_constraints)

        self.convergence_history_ = []

        # Solve dual problem using Newton's method
        lam = lam0.copy()
        for iteration in range(self.max_iter):
            grad = dual_gradient(lam)
            hess = dual_hessian(lam)

            # Check convergence
            max_error = np.max(np.abs(grad))
            self.convergence_history_.append({
                "iteration": iteration + 1,
                "max_error": max_error,
            })

            if max_error < self.tol:
                self.converged_ = True
                self.n_iterations_ = iteration + 1
                break

            # Newton step with regularization for stability
            try:
                hess_reg = hess + np.eye(n_constraints) * 1e-8
                step = np.linalg.solve(hess_reg, -grad)
            except np.linalg.LinAlgError:
                # Fallback to gradient descent
                step = -grad * 0.1

            # Line search for step size
            alpha = 1.0
            for _ in range(20):
                lam_new = lam + alpha * step
                if dual_objective(lam_new) < dual_objective(lam):
                    break
                alpha *= 0.5
            else:
                alpha = 0.01

            lam = lam + alpha * step

        else:
            self.converged_ = False
            self.n_iterations_ = self.max_iter

        # Recover primal weights
        weights = w0 * np.exp(A.T @ lam)

        # Apply bounds
        weights = np.maximum(weights, self.lower_bound)
        if self.upper_bound is not None:
            weights = np.minimum(weights, self.upper_bound)

        return weights

    def transform(
        self,
        data: pd.DataFrame,
        weight_col: str = "weight",
    ) -> pd.DataFrame:
        """
        Apply fitted calibration weights to data.

        Args:
            data: DataFrame to reweight (must match fitted data length)
            weight_col: Name of weight column to update

        Returns:
            DataFrame with updated weights

        Raises:
            ValueError: If not fitted or data length doesn't match
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator not fitted. Call fit() before transform().")

        if len(data) != self.n_records_:
            raise ValueError(
                f"Data length ({len(data)}) doesn't match fitted length ({self.n_records_})"
            )

        result = data.copy()
        result[weight_col] = self.weights_

        return result

    def fit_transform(
        self,
        data: pd.DataFrame,
        marginal_targets: Dict[str, Dict[str, float]],
        continuous_targets: Optional[Dict[str, float]] = None,
        weight_col: str = "weight",
    ) -> pd.DataFrame:
        """
        Fit calibration weights and apply to data in one call.

        Args:
            data: DataFrame with microdata
            marginal_targets: Dict of categorical targets
            continuous_targets: Dict of continuous totals (optional)
            weight_col: Name of weight column

        Returns:
            DataFrame with updated weights
        """
        self.fit(data, marginal_targets, continuous_targets, weight_col=weight_col)
        return self.transform(data, weight_col=weight_col)

    def validate(
        self,
        data: pd.DataFrame,
        weight_col: str = "weight",
    ) -> Dict[str, Any]:
        """
        Validate calibrated weights against targets.

        Computes weighted aggregates and compares to targets.

        Args:
            data: DataFrame with data (weights from fit will be used)
            weight_col: Name of weight column

        Returns:
            Dict with:
                - marginal_errors: {var: {category: (actual, target, error)}}
                - continuous_errors: {var: (actual, target, error)}
                - max_error: Maximum relative error across all targets
                - converged: Whether calibration converged
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        weights = self.weights_
        marginal_errors = {}
        continuous_errors = {}
        all_errors = []

        # Check marginal targets
        if self.marginal_targets_:
            for var, var_targets in self.marginal_targets_.items():
                marginal_errors[var] = {}
                for category, target in var_targets.items():
                    mask = data[var] == category
                    actual = weights[mask].sum()
                    rel_error = abs(actual - target) / target if target > 0 else 0
                    marginal_errors[var][category] = {
                        "actual": actual,
                        "target": target,
                        "relative_error": rel_error,
                    }
                    all_errors.append(rel_error)

        # Check continuous targets
        if self.continuous_targets_:
            for var, target in self.continuous_targets_.items():
                actual = (weights * data[var].values).sum()
                rel_error = abs(actual - target) / abs(target) if target != 0 else 0
                continuous_errors[var] = {
                    "actual": actual,
                    "target": target,
                    "relative_error": rel_error,
                }
                all_errors.append(rel_error)

        max_error = max(all_errors) if all_errors else 0

        return {
            "marginal_errors": marginal_errors,
            "continuous_errors": continuous_errors,
            "max_error": max_error,
            "converged": self.converged_,
        }

    def get_weight_stats(self) -> Dict[str, float]:
        """
        Get statistics about fitted weights.

        Returns:
            Dict with min, max, mean, cv (coefficient of variation), n_iterations
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        weights = self.weights_
        mean_w = weights.mean()

        return {
            "min_weight": float(weights.min()),
            "max_weight": float(weights.max()),
            "mean_weight": float(mean_w),
            "cv": float(weights.std() / mean_w) if mean_w > 0 else 0,
            "n_iterations": self.n_iterations_,
        }

    def get_convergence_history(self) -> List[Dict[str, Any]]:
        """
        Get convergence history from fitting.

        Returns:
            List of dicts with iteration number and max_error at each step
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        return self.convergence_history_
