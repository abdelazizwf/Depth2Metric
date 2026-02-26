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
