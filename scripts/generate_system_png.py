import base64
from pathlib import Path

import requests


def generate_system_png():
    root = Path(__file__).resolve().parent.parent
    arch_file = root / "ARCHITECTURE.md"

    if not arch_file.exists():
        print("❌ ARCHITECTURE.md not found.")
        return

    # Extract the mermaid block
    with arch_file.open() as f:
        content = f.read()

    start_marker = "```mermaid"
    end_marker = "```"

    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("❌ No Mermaid block found.")
        return

    # Get the actual graph code
    start_idx += len(start_marker)
    end_idx = content.find(end_marker, start_idx)
    mermaid_code = content[start_idx:end_idx].strip()

    # Encode for mermaid.ink
    # Note: We use base64 encoding for the URL
    encoded_string = base64.b64encode(mermaid_code.encode("ascii")).decode("ascii")
    url = f"https://mermaid.ink/img/{encoded_string}"

    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            output_path = root / "images" / "system_architecture_upd.png"
            with output_path.open("wb") as f:
                f.write(response.content)
            print(f"✅ System architecture PNG saved to: {output_path}")
        else:
            print(f"❌ Failed to fetch image: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Error during generation: {e}")

if __name__ == "__main__":
    generate_system_png()
