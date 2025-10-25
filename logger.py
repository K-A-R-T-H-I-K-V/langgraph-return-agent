import logging
from rich.logging import RichHandler

# Configure the logger
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)

# Create a logger instance
log = logging.getLogger("rich")

# Create convenience methods for different log levels
def log_agent_thought(thought):
    """Logs the agent's internal monologue."""
    log.info(f"[bold_cyan]🧠 AGENT THOUGHT[/bold_cyan]\n{thought}\n")

def log_tool_call(tool_name, args):
    """Logs a tool call with color."""
    log.info(f"[bold_yellow]🛠️ CALLING TOOL[/bold_yellow]: [yellow]{tool_name}[/yellow]({args})")
    
def log_error(error):
    """Logs an error."""
    log.error(f"[bold_red]🔥 ERROR[/bold_red]: {error}")