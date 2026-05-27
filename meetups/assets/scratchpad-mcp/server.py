# server.py
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scratchpad")
_notes: list[dict] = []

@mcp.tool()
def add_note(text: str, tag: str = "general") -> dict:
    """Append a note to the scratchpad."""
    note = {"id": len(_notes), "text": text, "tag": tag,
            "created": datetime.utcnow().isoformat()}
    _notes.append(note)
    return note

@mcp.tool()
def list_notes(tag: str | None = None) -> list[dict]:
    """Return notes, optionally filtered by tag."""
    return [n for n in _notes if tag is None or n["tag"] == tag]

@mcp.resource("scratchpad://all")
def all_notes() -> str:
    """All notes as a plain-text dump."""
    return "\n".join(f"[{n['tag']}] {n['text']}" for n in _notes) or "(empty)"

@mcp.prompt()
def summarise_by_tag(tag: str) -> str:
    """Ask the model to summarise notes for a given tag."""
    return f"Summarise the scratchpad notes tagged '{tag}'. Be concise."

if __name__ == "__main__":
    mcp.run()
