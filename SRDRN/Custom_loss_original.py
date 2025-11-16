import tensorflow as tf

class MaskedWeightedMAEPrecip(tf.keras.losses.Loss):
    """
    Weighted MAE in *mm/day* space, computed only over land pixels.

    Inputs expected during training:
      - y_true: z-space target  (normalized log1p), shape [B,H,W] or [B,H,W,1]
      - y_pred: z-space output  (normalized log1p), shape [B,H,W] or [B,H,W,1]

    The loss inverts both tensors back to mm/day and computes:
        L = sum(mask * w * |mm_pred - mm_true|) / sum(mask * w)

    where w = clip(mm_true / scale, w_min, w_max).
    """
    def __init__(self, mean_pr, std_pr, land_mask,
                 scale=36.6, w_min=0.1, w_max=2.0, name="MaskedWeightedMAEPrecip"):
        super().__init__(name=name)
        # mean_pr/std_pr: numpy arrays of shape (H, W) used for z↔log1p denorm
        # land_mask:      numpy bool/0-1 array of shape (H, W), True/1 on land

        mean_pr = tf.convert_to_tensor(mean_pr, tf.float32)
        std_pr  = tf.convert_to_tensor(std_pr,  tf.float32)
        mask    = tf.convert_to_tensor(land_mask, tf.float32)

        # Store as [1,H,W,1] for broadcasting with [B,H,W,1]
        self.mean_pr = tf.reshape(mean_pr, (1,) + mean_pr.shape + (1,))
        self.std_pr  = tf.reshape(std_pr,  (1,) + std_pr.shape  + (1,))
        self.mask    = tf.reshape(mask,    (1,) + mask.shape    + (1,))

        self.scale = tf.constant(scale,  tf.float32)
        self.w_min = tf.constant(w_min,  tf.float32)
        self.w_max = tf.constant(w_max,  tf.float32)

    def _ensure_channel_dim(self, t):
        """Ensure last channel dim exists: [B,H,W,1]."""
        t = tf.convert_to_tensor(t, tf.float32)
        if tf.rank(t) == 3:  # [B,H,W] -> [B,H,W,1]
            t = tf.expand_dims(t, axis=-1)
        return t

    def call(self, y_true, y_pred):
        # 0) Ensure shapes are [B,H,W,1]
        y_true = self._ensure_channel_dim(y_true)
        y_pred = self._ensure_channel_dim(y_pred)

        # 1) Denormalize back to log1p(mm/day)
        #    z -> logp via logp = z*std + mean
        logp_true = y_true * self.std_pr + self.mean_pr
        logp_pred = y_pred * self.std_pr + self.mean_pr

        # 2) Invert log1p to mm/day
        mm_true = tf.math.expm1(logp_true)  # >= 0
        mm_pred = tf.math.expm1(logp_pred)

        # 3) Absolute error in *mm/day* space
        err_mm = tf.abs(mm_pred - mm_true)

        # 4) Intensity-based weight from *true* mm (optional but common)
        w = tf.clip_by_value(mm_true / self.scale, self.w_min, self.w_max)

        weighted_err = err_mm * w * self.mask
        
        numer = tf.reduce_sum(weighted_err)
        count = tf.reduce_sum(self.mask * w) + tf.constant(1e-6, tf.float32)

        return numer / count

