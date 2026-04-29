import sys
from pathlib import Path

# Ensure the project root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.chatbot_langgraph import get_compiled_app


def generate_graph_png():
    app = get_compiled_app()
    try:
        # Generate the PNG bytes
        png_bytes = app.get_graph().draw_mermaid_png()

        # Save to file
        output_path = Path(__file__).resolve().parent.parent / "langgraph_workflow.png"
        with output_path.open("wb") as f:
            f.write(png_bytes)
        print(f"✅ Graph successfully generated at: {output_path}")
    except Exception as e:
        print(f"❌ Failed to generate graph: {e}")
        print("Note: This often requires 'pygraphviz' or an active internet connection for mermaid.ink")

if __name__ == "__main__":
    generate_graph_png()
