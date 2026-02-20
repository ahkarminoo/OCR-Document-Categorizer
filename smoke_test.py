import os
from pathlib import Path


def main():
    # Configure outputs for this run before importing the app.
    out_dir = (Path(__file__).parent / ".tmp" / "smoke_outputs").resolve()
    os.environ["OUTPUT_DIR"] = str(out_dir)
    os.environ.setdefault("TESS_LANG", "eng")

    from fastapi.testclient import TestClient
    from PIL import Image, ImageDraw
    import main as app_module

    (Path(__file__).parent / ".tmp").mkdir(parents=True, exist_ok=True)
    img_path = Path(__file__).parent / ".tmp" / "smoke_invoice.jpg"

    img = Image.new("RGB", (1200, 800), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([80, 80, 1120, 720], outline="black", width=6)
    d.text(
        (140, 140),
        "INVOICE\nInvoice No: 12345\nDate: 02/20/2026\nTotal: $42.50\nThank you!",
        fill="black",
    )
    img.save(img_path)

    client = TestClient(app_module.app)
    with open(img_path, "rb") as f:
        resp = client.post("/api/scan", files={"file": ("smoke_invoice.jpg", f, "image/jpeg")})

    print("status:", resp.status_code)
    data = resp.json()
    print("scan_id:", data.get("scan_id"))
    print("category/subcategory:", data.get("results", {}).get("category"), "/", data.get("results", {}).get("subcategory"))
    print("meta:", data.get("meta"))
    print("artifacts:", data.get("artifacts"))

    scan_id = data.get("scan_id")
    if scan_id:
        scan_dir = out_dir / scan_id
        print("saved_dir_exists:", scan_dir.exists(), str(scan_dir))
        for name in ["cropped.jpg", "ocr_ready.png", "ocr.txt", "result.json"]:
            p = scan_dir / name
            print(name, "exists:", p.exists(), "size:", p.stat().st_size if p.exists() else 0)


if __name__ == "__main__":
    main()

