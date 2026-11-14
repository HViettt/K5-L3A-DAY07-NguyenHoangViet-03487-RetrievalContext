"""Align evidence quotes in questions.json to match exact substrings in chunks.json."""
import json
import re

def main():
    chunks = json.load(open('data/converted/chunks.json', encoding='utf-8'))
    questions = json.load(open('data/converted/questions.json', encoding='utf-8'))
    
    aligned_count = 0
    for q in questions:
        gold = q['gold'][0]
        ev = gold['evidence']
        page = gold['page']
        page_chunks = [c for c in chunks if c['page'] == page]
        
        # Look for regex match with flexible whitespace
        words = re.findall(r'\S+', ev)
        # build regex pattern allowing any whitespace between tokens
        pattern = r'\s+'.join(re.escape(w) for w in words)
        
        matched_chunk = None
        matched_span = None
        for c in page_chunks:
            m = re.search(pattern, c['text'])
            if m:
                matched_chunk = c
                matched_span = m.group(0)
                break
        
        if matched_span:
            # Exact substring in c['text']!
            gold['evidence'] = matched_span
            aligned_count += 1
        else:
            # Let's try matching partial prefix if ev spans across boundaries
            for n_words in range(len(words), 4, -1):
                sub_pattern = r'\s+'.join(re.escape(w) for w in words[:n_words])
                for c in page_chunks:
                    m = re.search(sub_pattern, c['text'])
                    if m:
                        matched_chunk = c
                        matched_span = m.group(0)
                        break
                if matched_span:
                    break
            if matched_span:
                print(f"Partial match for {q['id']} ({len(matched_span)} chars): {repr(matched_span[:60])}...")
                gold['evidence'] = matched_span
                aligned_count += 1
            else:
                print(f"FAILED to match {q['id']}")

    print(f"Successfully aligned {aligned_count}/{len(questions)} questions.")
    
    # Save back
    with open('data/converted/questions.json', 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
