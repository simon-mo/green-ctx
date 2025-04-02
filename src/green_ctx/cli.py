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
def server(port: int = typer.Option(50051, help="Port to listen on"),
           kvpool_size_gb: int = typer.Option(
               30, help="Size of the KV cache pool in GB")):
    """Start the GPU multiplexer server."""
    try:
        server = GPUServer(port=port, kvpool_size_gb=kvpool_size_gb)
        server.start()
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status(host: str = typer.Option("localhost", help="Server host"),
           port: int = typer.Option(50051, help="Server port")):
    """Check the status of the GPU multiplexer server."""
    try:
        client = GPUClient(model_name="cli", host=host, port=port)
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


@app.command()
def initkvpool(size_gb: int = typer.Option(
    30, help="Size of the KV cache pool in GB")):
    """Set the size of the KV cache pool."""
    try:
        client = GPUClient(model_name="cli")
        size_bytes = size_gb * 1024**3
        client.set_kv_pool_memory_bytes(size_bytes)
        console.print(f"KV cache pool size set to {size_gb} GB")
        client.close()
    except Exception as e:
        console.print(f"[red]Error connecting to server: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
