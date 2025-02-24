#!/usr/bin/env python3
import csv
from pathlib import Path
import plotext as plt
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Header, Footer, Select, Static
from textual.reactive import reactive
import pandas as pd
import numpy as np
from typing import Optional
import re

class PlotWidget(Static):
    """A widget to display plotext plots."""

    _data: Optional[pd.DataFrame] = None
    current_metric = reactive("")

    def __init__(self):
        super().__init__()
        self.styles.height = "auto"
        plt.theme('clear')  # Use a clean theme without background colors

    @property
    def data(self) -> Optional[pd.DataFrame]:
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame):
        self._data = value
        self.plot_metric()

    def clean_plot_text(self, text: str) -> str:
        """Remove ANSI escape sequences and prepare text for Textual."""
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_text = ansi_escape.sub('', text)
        # Ensure proper line endings
        clean_text = clean_text.replace('\n', '\r\n')
        return clean_text

    def plot_metric(self) -> None:
        if self._data is None or not self.current_metric:
            return

        # Clear previous plot
        plt.clf()

        # Get data for current metric
        metric_data = self._data[self._data['metric'] == self.current_metric]
        if len(metric_data.index) == 0:
            return

        samples = metric_data['sample'].values
        values = metric_data['value'].values

        # Configure plot
        plt.plotsize(100, 30)  # Set a reasonable size for the terminal
        plt.theme('clear')  # Ensure clean theme is used

        # Create the plot
        plt.plot(samples, values, marker="dot", label=self.current_metric)
        plt.title(f"Time Series for {self.current_metric}")
        plt.xlabel("Sample")
        plt.ylabel("Value")

        # Convert plot to text and clean it
        plot_text = plt.build()
        plt.xscale("linear")  # Reset scale to avoid plotext bug
        clean_text = self.clean_plot_text(plot_text)

        # Update widget content
        self.update(clean_text)

    def watch_current_metric(self, metric: str) -> None:
        """React to metric changes."""
        self.plot_metric()

class MetricsViewer(App):
    """A Textual app to visualize CUPTI metrics."""

    TITLE = "CUPTI Metrics Viewer"
    CSS = """
    PlotWidget {
        height: auto;
        min-height: 30;
        border: solid green;
        padding: 1;
        background: $surface;
        color: $text;
    }

    Select {
        margin: 1;
    }
    """

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path
        self.plot_widget = PlotWidget()
        # Load data at initialization
        self.data = pd.read_csv(self.csv_path)
        self.metrics = list(self.data['metric'].unique())

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        # Initialize Select with options from data
        yield Select(
            options=[(metric, metric) for metric in self.metrics],
            id="metric_selector",
            value=self.metrics[0] if self.metrics else None
        )
        yield self.plot_widget
        yield Footer()

    def on_mount(self) -> None:
        """Set up initial state when the app starts."""
        if self.metrics:
            # Ensure DataFrame has the correct columns
            self.plot_widget.data = self.data[['sample', 'metric', 'value']]
            self.plot_widget.current_metric = self.metrics[0]

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle metric selection changes."""
        self.plot_widget.current_metric = event.value

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python visualizer.py <csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"Error: File {csv_path} does not exist")
        sys.exit(1)

    app = MetricsViewer(csv_path)
    app.run()

if __name__ == "__main__":
    main()
