import logging

from rich.logging import RichHandler

FORMAT = "[%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

