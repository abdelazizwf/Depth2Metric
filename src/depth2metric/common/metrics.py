from prometheus_client import Counter, Histogram

# Histograms for latencies
INFERENCE_LATENCY = Histogram(
    "depth2metric_inference_latency_seconds",
    "Latency of inference components in seconds",
    ["component"],
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, float("inf")),
)

# Counter for scaling methods used
SCALING_METHOD_TOTAL = Counter(
    "depth2metric_scaling_method_total",
    "Total count of scaling methods used",
    ["method"],
)

# Histogram for payload sizes
PAYLOAD_SIZE_BYTES = Histogram(
    "depth2metric_payload_size_bytes",
    "Size of point cloud payload in bytes",
    ["type"], # compressed or uncompressed
    buckets=(100000, 500000, 1000000, 2000000, 5000000, 10000000, float("inf")),
)

# Histogram for detection confidence
DETECTION_CONFIDENCE = Histogram(
    "depth2metric_detection_confidence",
    "Confidence scores of YOLO detections used for scaling",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Counter for user manual calibrations
MANUAL_CALIBRATION_TOTAL = Counter(
    "depth2metric_manual_calibration_total",
    "Total count of manual scale corrections by users",
)
