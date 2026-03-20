from tempfile import TemporaryDirectory
import unittest

from domain.plotting.experiment_plotter import ExperimentPlotter, LineSeries


class ExperimentPlotterSamplingTests(unittest.TestCase):
    def test_prepare_line_points_downsamples_large_series_and_keeps_last_point(self):
        with TemporaryDirectory() as tmp_dir:
            plotter = ExperimentPlotter(tmp_dir)
            line = LineSeries(
                label="Dense curve",
                x_values=[float(index) for index in range(2_505)],
                y_values=[float(index % 7) for index in range(2_505)],
                color="#2563eb",
            )

            points = plotter._prepare_line_points(line)

            self.assertLessEqual(len(points), plotter._MAX_LINE_POINTS + 1)
            self.assertEqual(points[0], (0.0, 0.0))
            self.assertEqual(points[-1], (2504.0, float(2504 % 7)))

    def test_write_line_chart_limits_marker_count_for_dense_series(self):
        with TemporaryDirectory() as tmp_dir:
            plotter = ExperimentPlotter(tmp_dir)
            point_count = 2_000
            plotter._write_line_chart(
                title="Dense threshold curves",
                x_label="Threshold",
                y_label="Score",
                file_name="threshold_curves.svg",
                series=[
                    LineSeries(
                        label="F1",
                        x_values=[float(index) for index in range(point_count)],
                        y_values=[float((index % 100) / 100) for index in range(point_count)],
                        color="#059669",
                    )
                ],
            )

            svg = (plotter.experiment_dir / "threshold_curves.svg").read_text(encoding="utf-8")

            self.assertLessEqual(svg.count("<circle"), plotter._MAX_POINT_MARKERS + 1)
            self.assertIn("<polyline", svg)


if __name__ == "__main__":
    unittest.main()
