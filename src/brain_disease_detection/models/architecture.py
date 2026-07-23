"""Transfer-learning model construction.

The classifiers fine-tune an ImageNet-pretrained CNN backbone with a small
custom head. This is the appropriate choice for medical-imaging datasets, which
are typically far too small to train a competitive CNN from scratch: the
backbone contributes general visual features while only the lightweight head
(and, later, the top backbone layers) is trained on the domain data.

All TensorFlow imports are deferred to call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ..exceptions import ConfigurationError
from ..logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import tensorflow as tf

logger = get_logger(__name__)

# Supported ImageNet backbones, keyed by config name.
_BACKBONES: dict[str, str] = {
    "MobileNetV2": "MobileNetV2",
    "EfficientNetB0": "EfficientNetB0",
    "ResNet50": "ResNet50",
}


def _resolve_backbone_factory(name: str) -> Callable[..., "tf.keras.Model"]:
    import tensorflow as tf

    if name not in _BACKBONES:
        supported = ", ".join(sorted(_BACKBONES))
        raise ConfigurationError(
            f"Unsupported backbone '{name}'. Supported backbones: {supported}."
        )
    return getattr(tf.keras.applications, _BACKBONES[name])


def build_augmentation() -> "tf.keras.Sequential":
    """Return range-agnostic geometric augmentations applied during training.

    Augmentation layers are inert at inference time, so they can safely live
    inside the model graph and travel with the saved model.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


def build_model(
    num_classes: int,
    input_shape: tuple[int, int, int],
    backbone_name: str = "MobileNetV2",
    dropout: float = 0.3,
) -> tuple["tf.keras.Model", "tf.keras.Model"]:
    """Assemble a transfer-learning classifier.

    The backbone is frozen so the head can be trained first; call
    :func:`enable_fine_tuning` afterwards to unfreeze its top layers.

    Args:
        num_classes: Number of output classes (softmax units).
        input_shape: ``(height, width, channels)`` of the input tensor.
        backbone_name: One of the supported ImageNet backbones.
        dropout: Dropout rate applied before the classification head.

    Returns:
        A tuple ``(model, backbone)``; the backbone is returned so callers can
        toggle fine-tuning without re-searching the layer graph.
    """
    import tensorflow as tf

    backbone_factory = _resolve_backbone_factory(backbone_name)
    backbone = backbone_factory(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = False

    # The pipeline feeds images already scaled to [0, 1]; the model expects the
    # same range at inference (see brain_disease_detection.data.preprocessing).
    inputs = tf.keras.Input(shape=input_shape)
    x = build_augmentation()(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)

    model = tf.keras.Model(inputs, outputs, name=f"{backbone_name}_classifier")
    logger.info(
        "Built %s classifier: input=%s, classes=%d, params=%d",
        backbone_name,
        input_shape,
        num_classes,
        model.count_params(),
    )
    return model, backbone


def enable_fine_tuning(backbone: "tf.keras.Model", fine_tune_at: int) -> None:
    """Unfreeze the backbone from layer ``fine_tune_at`` onward.

    Lower layers (generic edges/textures) stay frozen; higher, more task-specific
    layers become trainable for the fine-tuning phase.
    """
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    trainable = sum(1 for layer in backbone.layers if layer.trainable)
    logger.info(
        "Fine-tuning enabled: %d/%d backbone layers trainable (frozen below %d).",
        trainable,
        len(backbone.layers),
        fine_tune_at,
    )
