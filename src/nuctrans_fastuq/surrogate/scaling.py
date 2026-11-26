"""
scaling.py - Output scaling for training, and undoing it for evaluation.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def _check_log_domain(val, scale):
    """Raise if val is outside the domain of the log scaling (val <= -scale), instead of giving NaN/-inf."""
    if np.any(np.asarray(val) <= -scale):
        raise ValueError(
            f"log scaling needs outputs above -{scale:g} (minus the smallest positive training value), "
            f"but the minimum is {np.nanmin(val):g}. It is meant for non-negative outputs: set the negative "
            f"values to 0 first (surrogate.qoi: floor > 0 with signed: false), or use another transform."
        )


def fit_scaling(train_output_at_x, scale_type=None, **kwargs):
    """Fit an output scaling on {key: array(n_samples, n_features)}.

    scale_type: None/'none', 'div_by_max', 'standardize', 'minmax' or 'log'.
    'log' is y = log1p(x / scale) (natural log), with scale the smallest positive training value.
    Returns the scaling config that is stored on the engine.
    """
    config = {"type": scale_type}
    if scale_type is None or scale_type == "none":
        config["type"] = "none"
        return config

    # one scaler for all outputs together
    all_vals = np.concatenate(
        [arr.ravel() for arr in train_output_at_x.values()]
    ).reshape(-1, 1)

    if scale_type == "div_by_max":
        max_val = np.max(np.abs(all_vals))
        config["max_val"] = float(max_val) if max_val > 0 else 1.0

    elif scale_type == "standardize":
        scaler = StandardScaler()
        scaler.fit(all_vals)
        config["scaler"] = scaler

    elif scale_type == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(all_vals)
        config["scaler"] = scaler

    elif scale_type == "log":
        positive = all_vals[all_vals > 0]
        config["scale"] = float(positive.min()) if positive.size else 1.0
        _check_log_domain(all_vals, config["scale"])

    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    return config


def scale_data(data, config):
    """Scale data (dict of arrays, array or float) with the config."""
    scale_type = config.get("type", "none")
    if scale_type == "none" or scale_type is None:
        return data

    def transform(val):
        if scale_type == "div_by_max":
            return val / config["max_val"]
        elif scale_type == "standardize" or scale_type == "minmax":
            scaler = config["scaler"]
            orig_shape = val.shape
            val_flat = val.ravel().reshape(-1, 1)
            val_scaled = scaler.transform(val_flat)
            return val_scaled.reshape(orig_shape)
        elif scale_type == "log":
            _check_log_domain(val, config["scale"])
            return np.log1p(val / config["scale"])
        return val

    if isinstance(data, dict):
        scaled = {}
        for k, v in data.items():
            scaled[k] = transform(v)
        return scaled
    else:
        return transform(data)


def descale_data(data, config):
    """Undo the scaling (dict of arrays, array or float)."""
    scale_type = config.get("type", "none")
    if scale_type == "none" or scale_type is None:
        return data

    def inv_transform(val):
        if scale_type == "div_by_max":
            return val * config["max_val"]
        elif scale_type == "standardize" or scale_type == "minmax":
            scaler = config["scaler"]
            orig_shape = val.shape
            val_flat = val.ravel().reshape(-1, 1)
            val_descaled = scaler.inverse_transform(val_flat)
            return val_descaled.reshape(orig_shape)
        elif scale_type == "log":
            return config["scale"] * np.expm1(val)
        return val

    if isinstance(data, dict):
        descaled = {}
        for k, v in data.items():
            descaled[k] = inv_transform(v)
        return descaled
    else:
        return inv_transform(data)


def descale_std(std_scaled, config):
    """Undo a linear scaling for a std: x = a*y + b gives std_x = a*std_y (the shift b drops out). Exact.

    The factor a is the slope of descale_data. Called by descale_prediction for the linear scalings only:
    the std of the log scaling also depends on the mean.
    """
    scale_type = config.get("type", "none")
    if scale_type == "none" or scale_type is None:
        return std_scaled

    def inv_transform_std(val):
        if scale_type == "div_by_max":
            return val * config["max_val"]
        elif scale_type == "standardize":
            return (
                val * config["scaler"].scale_[0]
            )  # inverse_transform: x = y * scale_ + mean_
        elif scale_type == "minmax":
            # inverse_transform: x = (y - min_) / scale_. Not data_range_: it is 0 for constant training
            # outputs, where sklearn sets scale_ = 1.
            return val / config["scaler"].scale_[0]
        return val

    if isinstance(std_scaled, dict):
        return {k: inv_transform_std(v) for k, v in std_scaled.items()}
    return inv_transform_std(std_scaled)


def descale_prediction(mean_scaled, std_scaled, config):
    """Surrogate prediction y ~ N(mean_scaled, std_scaled^2) in scaled units (prediction, std) in physical units.

    prediction = descale_data(mean_scaled); for log it is scale * expm1(mu), below the mean E[x] = scale * expm1(mu + s^2/2).
    std: the exact std of x. For log (x = scale * (exp(y) - 1), lognormal) with mu = mean_scaled, s = std_scaled
    """
    if config is None or config.get("type", "none") != "log":
        cfg = config or {"type": "none"}
        return descale_data(mean_scaled, cfg), descale_std(std_scaled, cfg)

    scale = config["scale"]

    def log_std(mu, s):
        mu, s = np.asarray(mu, dtype=float), np.asarray(s, dtype=float)
        return (
            scale * np.exp(mu + 0.5 * s**2) * np.sqrt(np.expm1(s**2))
        )  # expm1: accurate for small s; Var[x] = scale^2 * exp(2*mu + s^2) * (exp(s^2) - 1)

    if isinstance(std_scaled, dict):
        std = {k: log_std(mean_scaled[k], std_scaled[k]) for k in std_scaled}
    else:
        std = log_std(mean_scaled, std_scaled)
    return descale_data(mean_scaled, config), std


# A log-space prediction above the training range becomes orders of magnitude too large after exp,
# so it is clipped before it is turned back (in log units: a factor e above the training maximum).
LOG_CLIP_MARGIN = 1.0


def scaled_range(scaled_train):
    """(min, max) of the scaled training outputs {key: array}; keys that are not arrays (x_values) are skipped."""
    arrays = [np.asarray(v) for k, v in scaled_train.items() if k != "x_values"]
    return min(a.min() for a in arrays), max(a.max() for a in arrays)


def clip_scaled(data, config, train_range, margin=LOG_CLIP_MARGIN):
    """Clip scaled predictions (dict of arrays, array or float) to the training range, for the log scaling only.

    The lower end is the training minimum (no negative concentrations), the upper end is the training
    maximum + margin. The linear scalings are returned unchanged: their overshoot is not amplified.
    Clip the scaled mean before descale_prediction; the std is not clipped.
    """
    if config is None or config.get("type") != "log":
        return data
    lo, hi = train_range[0], train_range[1] + margin
    if isinstance(data, dict):
        return {k: np.clip(v, lo, hi) for k, v in data.items()}
    return np.clip(data, lo, hi)


def clipped_fraction(data, config, train_range, margin=LOG_CLIP_MARGIN):
    """What clip_scaled changes in the scaled predictions {key: array}: {key: (below, above)}.

    below: the fraction under the training minimum, above: over the training maximum + margin. Call it before
    clip_scaled. Empty for the linear scalings, which are not clipped. A large fraction means the surrogate
    extrapolates there, and the margin shapes the result.
    """
    if config is None or config.get("type") != "log":
        return {}
    lo, hi = train_range[0], train_range[1] + margin
    return {k: (float(np.mean(np.asarray(v) < lo)), float(np.mean(np.asarray(v) > hi))) for k, v in data.items()}
