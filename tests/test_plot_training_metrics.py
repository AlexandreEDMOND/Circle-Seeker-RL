from scripts.plot_training_metrics import build_svg, metric_points


def test_metric_points_uses_update_history() -> None:
    points = metric_points(
        {
            "updates": [
                {"global_step": 8, "mean_return": -1.0, "success_rate": 0.0},
                {"global_step": 16, "mean_return": 2.0, "success_rate": 0.5},
            ]
        }
    )

    assert points == [
        {"global_step": 8.0, "mean_return": -1.0, "success_rate": 0.0},
        {"global_step": 16.0, "mean_return": 2.0, "success_rate": 0.5},
    ]


def test_build_svg_contains_expected_training_panels() -> None:
    svg = build_svg(
        [
            {
                "global_step": 8.0,
                "mean_return": -1.0,
                "success_rate": 0.0,
                "entropy": 4.0,
                "value_loss": 0.3,
                "policy_loss": -0.01,
                "approx_kl": 0.001,
                "clip_fraction": 0.0,
            },
            {
                "global_step": 16.0,
                "mean_return": 2.0,
                "success_rate": 0.5,
                "entropy": 3.5,
                "value_loss": 0.2,
                "policy_loss": -0.02,
                "approx_kl": 0.002,
                "clip_fraction": 0.1,
            },
        ]
    )

    assert "<svg" in svg
    assert "Mean return" in svg
    assert "Success rate" in svg
    assert "Approx KL" in svg
    assert "polyline" in svg
