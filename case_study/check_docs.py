import json
import os

file_path = r"e:\HyperRAG\PoincaréRAG\eval\datasets\legal\work_dir_Hicluster_5_hiem\kv_store_full_docs.json"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    found_ocugen = False
    found_section = False
    
    print(f"Total documents: {len(data)}")
    
    for doc_id, doc_data in data.items():
        content = doc_data.get('content', '')
        
        if "Ocugen" in content:
            print(f"Found 'Ocugen' in {doc_id}")
            found_ocugen = True
            # Print a snippet around Ocugen
            idx = content.find("Ocugen")
            start = max(0, idx - 100)
            end = min(len(content), idx + 100)
            print(f"Snippet: ...{content[start:end]}...")
            
        if "Section 7(g)" in content:
            print(f"Found 'Section 7(g)' in {doc_id}")
            found_section = True
            idx = content.find("Section 7(g)")
            start = max(0, idx - 100)
            end = min(len(content), idx + 100)
            print(f"Snippet: ...{content[start:end]}...")

        if "Placement Shares" in content and "use of proceeds" in content.lower():
             print(f"Found 'Placement Shares' and 'use of proceeds' in {doc_id}")
             
    if not found_ocugen:
        print("Did not find 'Ocugen'")
    if not found_section:
        print("Did not find 'Section 7(g)'")
        
except Exception as e:
    print(f"Error: {e}")
