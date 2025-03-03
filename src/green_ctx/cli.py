import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
import signal
import sys
from .server.server import GPUServer
from .client.client import GPUClient



app = typer.Typer()
console = Console()

def signal_handler(sig, frame):
    console.print("\nShutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

@app.command()
def server(port: int = typer.Option(50051, help="Port to listen on")):
    """Start the GPU multiplexer server."""
    try:
        server = GPUServer(port=port)
        server.start()
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def status(host: str = typer.Option("localhost", help="Server host"),
           port: int = typer.Option(50051, help="Server port")):
    """Check the status of the GPU multiplexer server."""
    try:
        client = GPUClient(host=host, port=port)
        status = client.health_check()

        table = Table(title="GPU Multiplexer Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in status.items():
            table.add_row(key, str(value))

        console.print(table)
        client.close()
    except Exception as e:
        console.print(f"[red]Error connecting to server: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()