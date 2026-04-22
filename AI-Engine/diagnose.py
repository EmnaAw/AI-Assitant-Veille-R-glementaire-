# diagnose.py  —  run this to inspect your actual file structure
import os

print("=" * 60)
print("DIRECTORY SCAN")
print("=" * 60)

# Scan up to 3 levels from current directory
for root, dirs, files in os.walk("."):
    # Skip hidden folders and common noise
    dirs[:] = [d for d in dirs if not d.startswith(".") 
               and d not in ["__pycache__", "node_modules"]]
    
    depth = root.count(os.sep)
    if depth > 3:
        continue
    
    indent = "  " * depth
    print(f"{indent}📁 {root}/")
    
    txt_files = [f for f in files if f.endswith(".txt")]
    pdf_files = [f for f in files if f.endswith(".pdf")]
    other     = [f for f in files if not f.endswith((".txt", ".pdf", ".pyc"))]
    
    if txt_files:
        print(f"{indent}  TXT files ({len(txt_files)}): {txt_files}")
    if pdf_files:
        print(f"{indent}  PDF files ({len(pdf_files)}): {pdf_files[:5]}"
              f"{'...' if len(pdf_files) > 5 else ''}")
    if other:
        print(f"{indent}  Other    ({len(other)}): {other}")

print("\n" + "=" * 60)
print("PDF FILENAME SAMPLE vs TXT RECORD PDF FIELD")
print("=" * 60)

# Find TXT files anywhere
import glob
for txt in glob.glob("**/*.txt", recursive=True):
    with open(txt, encoding="utf-8") as f:
        content = f.read()
    # Extract PDF: field values
    import re
    pdf_refs = re.findall(r"^PDF:\s*(.+)$", content, re.MULTILINE)
    pdf_refs = [p.strip() for p in pdf_refs if p.strip()]
    if pdf_refs:
        print(f"\nIn {txt} — PDF references ({len(pdf_refs)} total):")
        print(f"  First 5: {pdf_refs[:5]}")
        print(f"  Last  5: {pdf_refs[-5:]}")