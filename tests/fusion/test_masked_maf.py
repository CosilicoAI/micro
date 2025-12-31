"""Tests for masked MAF model."""

import numpy as np
import pytest
import torch

from microplex.fusion.masked_maf import MaskedMAF


class TestMaskedMAF:
    """Test MaskedMAF class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with missing values."""
        np.random.seed(42)
        n_samples = 50
        n_features = 5

        # Create data
        X = np.random.randn(n_samples, n_features).astype(np.float32)

        # Create mask (80% observed)
        mask = np.random.rand(n_samples, n_features) > 0.2

        # Ensure at least one observation per feature
        for i in range(n_features):
            mask[i, i] = True

        return X, mask

    def test_init(self):
        model = MaskedMAF(n_features=10, n_layers=2, hidden_dim=16)
        assert model.n_features == 10
        assert model.n_context >= 1  # Minimum 1 for flow

    def test_fit_runs(self, sample_data):
        X, mask = sample_data
        model = MaskedMAF(n_features=5, n_layers=2, hidden_dim=16)

        # Should not raise
        model.fit(X, mask, epochs=5, batch_size=16, verbose=False)

        assert model.feature_means_ is not None
        assert model.feature_stds_ is not None

    def test_fit_decreases_loss(self, sample_data):
        X, mask = sample_data
        model = MaskedMAF(n_features=5, n_layers=2, hidden_dim=32)

        model.fit(X, mask, epochs=50, batch_size=16, lr=1e-2, verbose=False)

        # Loss should decrease overall (compare first 5 to last 5 epochs)
        early_loss = np.mean(model.training_losses_[:5])
        late_loss = np.mean(model.training_losses_[-5:])
        assert late_loss < early_loss

    def test_sample_shape(self, sample_data):
        X, mask = sample_data
        model = MaskedMAF(n_features=5, n_layers=2, hidden_dim=16)
        model.fit(X, mask, epochs=5, batch_size=16, verbose=False)

        samples = model.sample(n_samples=10)
        assert samples.shape == (10, 5)

    def test_sample_finite(self, sample_data):
        X, mask = sample_data
        model = MaskedMAF(n_features=5, n_layers=2, hidden_dim=16)
        model.fit(X, mask, epochs=5, batch_size=16, verbose=False)

        samples = model.sample(n_samples=100)
        assert np.isfinite(samples).all()

    def test_impute_shape(self, sample_data):
        X, mask = sample_data
        model = MaskedMAF(n_features=5, n_layers=2, hidden_dim=16)
        model.fit(X, mask, epochs=5, batch_size=16, verbose=False)

        imputed = model.impute(X, mask, n_samples=1)
        assert imputed.shape == X.shape

    def test_impute_preserves_observed(self, sample_data):
        X, mask = sample_data
        model = MaskedMAF(n_features=5, n_layers=2, hidden_dim=16)
        model.fit(X, mask, epochs=5, batch_size=16, verbose=False)

        # Normalize X like the model does
        X_norm = (X - model.feature_means_) / model.feature_stds_
        imputed_norm = model.impute(X_norm, mask, n_samples=1)

        # Denormalize
        imputed = imputed_norm * model.feature_stds_ + model.feature_means_

        # Observed values should be preserved (approximately)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if mask[i, j]:
                    # Should be close to original
                    assert np.abs(imputed[i, j] - X[i, j]) < 1.0

    def test_save_load(self, sample_data, tmp_path):
        X, mask = sample_data
        model = MaskedMAF(n_features=5, n_layers=2, hidden_dim=16)
        model.fit(X, mask, epochs=5, batch_size=16, verbose=False)

        # Save
        save_path = tmp_path / "model"
        model.save(str(save_path))

        # Load
        loaded = MaskedMAF.load(str(save_path))

        # Should have same parameters
        assert loaded.n_features == model.n_features
        assert np.allclose(loaded.feature_means_, model.feature_means_)

        # Should generate similar samples
        np.random.seed(0)
        torch.manual_seed(0)
        samples1 = model.sample(5)

        np.random.seed(0)
        torch.manual_seed(0)
        samples2 = loaded.sample(5)

        assert np.allclose(samples1, samples2)


class TestMaskedMAFWithWeights:
    """Test MaskedMAF with sample weights."""

    def test_fit_with_weights(self):
        np.random.seed(42)
        n_samples = 50
        n_features = 5

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        mask = np.ones((n_samples, n_features), dtype=bool)
        weights = np.random.rand(n_samples).astype(np.float32)

        model = MaskedMAF(n_features=5, n_layers=2, hidden_dim=16)
        model.fit(X, mask, sample_weights=weights, epochs=5, verbose=False)

        assert model.training_losses_[-1] < model.training_losses_[0]


class TestInverseFrequencyWeighting:
    """Test inverse frequency weighting for sparse observations."""

    def test_sparse_features_weighted_higher(self):
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features).astype(np.float32)

        # Make first feature very sparse (10% observed)
        # and last feature fully observed
        mask = np.ones((n_samples, n_features), dtype=bool)
        mask[:90, 0] = False  # Only 10 observed

        model = MaskedMAF(
            n_features=5,
            n_layers=2,
            hidden_dim=16,
            use_inverse_freq_weighting=True,
        )
        model.fit(X, mask, epochs=1, verbose=False)

        # Sparse feature should have higher weight
        assert model.dim_weights_[0] > model.dim_weights_[-1]
